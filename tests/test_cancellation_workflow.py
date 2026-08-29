import importlib
import json
import sys
import types

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

import src.tools.order_cancellation as cancellation
import src.tools.order_status as order_status
import src.tools.support_docs as support_docs


class FakeRuntime:
    def __init__(self, thread_id: str = "test-thread") -> None:
        self.state = {
            "messages": [HumanMessage(content="Please cancel order order-1.")]
        }
        self.config = {"configurable": {"thread_id": thread_id}}


def build_cancellation_tool_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode([cancellation.cancel_order]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    return builder.compile(checkpointer=InMemorySaver())


class ChatGraphAdapter:
    def __init__(self) -> None:
        self.graph = build_cancellation_tool_graph()

    def invoke(self, graph_input, config):
        if isinstance(graph_input, Command):
            return self.graph.invoke(graph_input, config=config)
        return self.graph.invoke(cancellation_request(), config=config)


def cancellation_request() -> dict:
    return {
        "messages": [
            HumanMessage(content="Please cancel order order-1."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "cancel_order",
                        "args": {"order_id": "order-1"},
                        "id": "cancel-call-1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    }


def parsed_tool_result(result: dict) -> dict:
    message = result["messages"][-1]
    assert isinstance(message, ToolMessage)
    return json.loads(message.content)


def persist_review(result: dict, thread_id: str) -> dict:
    ticket, _ = cancellation.create_pending_review_after_interrupt(
        result["__interrupt__"][0].value,
        thread_id,
    )
    return ticket


@pytest.fixture
def review_database(monkeypatch):
    state = {
        "order_status": "invoiced",
        "ticket": None,
        "tickets_created": 0,
        "cancel_calls": 0,
    }

    def fetch_order(order_id: str) -> dict | None:
        if state["order_status"] is None:
            return None
        return {"order_id": order_id, "order_status": state["order_status"]}

    def fetch_pending(order_id: str) -> dict | None:
        ticket = state["ticket"]
        if ticket and ticket["status"] == cancellation.PENDING_REVIEW:
            return dict(ticket)
        return None

    def fetch_pending_by_thread(thread_id: str) -> dict | None:
        ticket = state["ticket"]
        if (
            ticket
            and ticket["thread_id"] == thread_id
            and ticket["status"] == cancellation.PENDING_REVIEW
        ):
            return dict(ticket)
        return None

    def get_or_create(
        order_id: str,
        thread_id: str,
        review_reason: str,
        request_details: str,
    ) -> tuple[dict, bool]:
        existing = fetch_pending(order_id)
        if existing:
            return existing, False

        state["tickets_created"] += 1
        state["ticket"] = {
            "ticket_id": 100 + state["tickets_created"],
            "order_id": order_id,
            "issue_type": "order_cancellation",
            "status": cancellation.PENDING_REVIEW,
            "thread_id": thread_id,
            "review_reason": review_reason,
            "request_details": request_details,
        }
        return dict(state["ticket"]), True

    def update_ticket(
        ticket_id: int,
        status: str,
        expected_status: str | None = None,
    ) -> dict | None:
        ticket = state["ticket"]
        if ticket is None or ticket["ticket_id"] != ticket_id:
            return None
        if expected_status and ticket["status"] != expected_status:
            return None
        ticket["status"] = status
        return dict(ticket)

    def cancel_order(
        order_id: str,
        allowed_statuses,
        review_ticket_id: int | None = None,
    ) -> dict | None:
        state["cancel_calls"] += 1
        if state["order_status"] not in allowed_statuses:
            return None
        state["order_status"] = "canceled"
        if review_ticket_id is not None:
            state["ticket"]["status"] = cancellation.APPROVED
        return {"order_id": order_id, "order_status": "canceled"}

    monkeypatch.setattr(cancellation, "fetch_order_by_id", fetch_order)
    monkeypatch.setattr(
        cancellation,
        "fetch_pending_cancellation_ticket",
        fetch_pending,
    )
    monkeypatch.setattr(
        cancellation,
        "get_or_create_pending_cancellation_ticket",
        get_or_create,
    )
    monkeypatch.setattr(
        cancellation,
        "update_support_ticket_status",
        update_ticket,
    )
    monkeypatch.setattr(cancellation, "cancel_order_by_id", cancel_order)
    state["fetch_pending_by_thread"] = fetch_pending_by_thread
    state["update_ticket"] = update_ticket
    return state


@pytest.mark.parametrize("status", ["created", "approved", "processing"])
def test_auto_cancellable_statuses_cancel_without_ticket(monkeypatch, status):
    monkeypatch.setattr(
        cancellation,
        "fetch_order_by_id",
        lambda order_id: {"order_id": order_id, "order_status": status},
    )
    monkeypatch.setattr(
        cancellation,
        "cancel_order_by_id",
        lambda order_id, allowed_statuses: {
            "order_id": order_id,
            "order_status": "canceled",
        },
    )
    monkeypatch.setattr(
        cancellation,
        "get_or_create_pending_cancellation_ticket",
        lambda **kwargs: pytest.fail("Auto-cancellation created a ticket."),
    )

    result = cancellation.cancel_order.func("order-1", FakeRuntime())

    assert result["outcome"] == "cancelled"
    assert result["ticket_id"] is None


def test_invoiced_order_creates_one_ticket_and_interrupts(review_database):
    graph = build_cancellation_tool_graph()
    config = {"configurable": {"thread_id": "review-thread"}}

    result = graph.invoke(cancellation_request(), config=config)
    ticket = persist_review(result, "review-thread")

    assert result["__interrupt__"][0].value["ticket_id"] is None
    assert ticket["ticket_id"] == 101
    assert result["__interrupt__"][0].value["status"] == "PENDING_REVIEW"
    assert review_database["tickets_created"] == 1
    assert review_database["order_status"] == "invoiced"


def test_admin_approve_resumes_and_cancels(review_database):
    graph = build_cancellation_tool_graph()
    config = {"configurable": {"thread_id": "approve-thread"}}
    interrupted = graph.invoke(cancellation_request(), config=config)
    persist_review(interrupted, "approve-thread")

    result = graph.invoke(
        Command(resume={"decision": "approve", "ticket_id": 101}),
        config=config,
    )
    tool_result = parsed_tool_result(result)

    assert tool_result["outcome"] == "approved_and_cancelled"
    assert review_database["order_status"] == "canceled"
    assert review_database["ticket"]["status"] == "APPROVED"
    assert review_database["cancel_calls"] == 1
    assert review_database["tickets_created"] == 1


def test_admin_deny_resumes_without_cancelling(review_database):
    graph = build_cancellation_tool_graph()
    config = {"configurable": {"thread_id": "deny-thread"}}
    interrupted = graph.invoke(cancellation_request(), config=config)
    persist_review(interrupted, "deny-thread")

    result = graph.invoke(
        Command(resume={"decision": "deny", "ticket_id": 101}),
        config=config,
    )
    tool_result = parsed_tool_result(result)

    assert tool_result["outcome"] == "admin_denied"
    assert review_database["order_status"] == "invoiced"
    assert review_database["ticket"]["status"] == "DENIED"
    assert review_database["cancel_calls"] == 0


@pytest.mark.parametrize("status", ["shipped", "delivered"])
def test_shipped_and_delivered_orders_are_rejected(monkeypatch, status):
    monkeypatch.setattr(
        cancellation,
        "fetch_order_by_id",
        lambda order_id: {"order_id": order_id, "order_status": status},
    )
    monkeypatch.setattr(
        cancellation,
        "fetch_pending_cancellation_ticket",
        lambda order_id: None,
    )
    monkeypatch.setattr(
        cancellation,
        "cancel_order_by_id",
        lambda *args, **kwargs: pytest.fail("Hard-rejected order was changed."),
    )

    result = cancellation.cancel_order.func("order-1", FakeRuntime())

    assert result["outcome"] == "rejected"
    assert result["status"] == status


def test_nonexistent_order_is_safe(monkeypatch):
    monkeypatch.setattr(cancellation, "fetch_order_by_id", lambda order_id: None)
    monkeypatch.setattr(
        cancellation,
        "fetch_pending_cancellation_ticket",
        lambda order_id: None,
    )
    monkeypatch.setattr(
        cancellation,
        "cancel_order_by_id",
        lambda *args, **kwargs: pytest.fail("Missing order was changed."),
    )

    result = cancellation.cancel_order.func("missing", FakeRuntime())

    assert result["outcome"] == "not_found"


def test_repeated_pending_request_does_not_create_duplicate(review_database):
    first_graph = build_cancellation_tool_graph()
    first_config = {"configurable": {"thread_id": "original-thread"}}
    interrupted = first_graph.invoke(cancellation_request(), config=first_config)
    persist_review(interrupted, "original-thread")

    second_graph = build_cancellation_tool_graph()
    second_config = {"configurable": {"thread_id": "retry-thread"}}
    result = second_graph.invoke(cancellation_request(), config=second_config)
    tool_result = parsed_tool_result(result)

    assert "__interrupt__" not in result
    assert tool_result["outcome"] == "pending_admin_review"
    assert tool_result["ticket_id"] == 101
    assert review_database["tickets_created"] == 1


def test_llm_cannot_supply_policy_or_admin_decision():
    tool_schema = cancellation.cancel_order.tool_call_schema.model_json_schema()

    assert set(tool_schema["properties"]) == {"order_id"}
    assert cancellation.cancellation_policy("created") == "auto_cancel"
    assert cancellation.cancellation_policy("invoiced") == "admin_review"
    assert cancellation.cancellation_policy("shipped") == "hard_reject"
    assert cancellation.cancellation_policy("unknown") == "hard_reject"


def load_api_with_fake_graph(monkeypatch, fake_graph):
    fake_graph_module = types.ModuleType("src.agent.graph")
    fake_graph_module.graph = fake_graph
    monkeypatch.setitem(sys.modules, "src.agent.graph", fake_graph_module)
    sys.modules.pop("src.api.app", None)
    return importlib.import_module("src.api.app")


def configure_review_api(api_module, monkeypatch, review_database) -> None:
    monkeypatch.setattr(api_module, "ADMIN_API_KEY", "test-admin-key")
    monkeypatch.setattr(
        api_module,
        "fetch_pending_cancellation_ticket_by_thread_id",
        review_database["fetch_pending_by_thread"],
    )
    monkeypatch.setattr(
        api_module,
        "fetch_support_ticket_by_id",
        lambda ticket_id: (
            dict(review_database["ticket"])
            if review_database["ticket"]
            and review_database["ticket"]["ticket_id"] == ticket_id
            else None
        ),
    )
    monkeypatch.setattr(
        api_module,
        "update_support_ticket_status",
        review_database["update_ticket"],
    )


def test_chat_retry_after_deny_reinterrupts_on_same_thread(
    monkeypatch,
    review_database,
):
    api_module = load_api_with_fake_graph(monkeypatch, ChatGraphAdapter())
    configure_review_api(api_module, monkeypatch, review_database)
    request = api_module.ChatRequest(
        message="Cancel order order-1",
        thread_id="same-thread",
    )

    first = api_module.chat(request)
    denied = api_module._resume_admin_review(101, "deny")
    retried = api_module.chat(request)

    assert first.approval_required is True
    assert denied.ticket_status == "DENIED"
    assert retried.approval_required is True
    assert retried.interrupt["ticket_id"] == 102
    assert review_database["tickets_created"] == 2


def test_interrupt_failure_cannot_create_pending_ticket(monkeypatch):
    monkeypatch.setattr(
        cancellation,
        "fetch_order_by_id",
        lambda order_id: {"order_id": order_id, "order_status": "invoiced"},
    )
    monkeypatch.setattr(
        cancellation,
        "fetch_pending_cancellation_ticket",
        lambda order_id: None,
    )
    monkeypatch.setattr(
        cancellation,
        "get_or_create_pending_cancellation_ticket",
        lambda **kwargs: pytest.fail("Ticket was created before checkpointing."),
    )
    monkeypatch.setattr(
        cancellation,
        "interrupt",
        lambda payload: (_ for _ in ()).throw(RuntimeError("checkpoint failed")),
    )

    with pytest.raises(RuntimeError, match="checkpoint failed"):
        cancellation.cancel_order.func("order-1", FakeRuntime())


def test_existing_pending_review_reports_approval_required(
    monkeypatch,
    review_database,
):
    graph = build_cancellation_tool_graph()
    config = {"configurable": {"thread_id": "pending-thread"}}
    interrupted = graph.invoke(cancellation_request(), config=config)
    persist_review(interrupted, "pending-thread")

    api_module = load_api_with_fake_graph(monkeypatch, ChatGraphAdapter())
    configure_review_api(api_module, monkeypatch, review_database)
    response = api_module.chat(
        api_module.ChatRequest(
            message="Cancel order order-1",
            thread_id="new-thread",
        )
    )

    assert response.approval_required is True
    assert response.interrupt["ticket_id"] == 101
    assert review_database["tickets_created"] == 1


def test_admin_endpoint_rejects_missing_and_invalid_api_keys(monkeypatch):
    api_module = load_api_with_fake_graph(monkeypatch, object())
    monkeypatch.setattr(api_module, "ADMIN_API_KEY", "test-admin-key")
    client = TestClient(api_module.app)

    missing = client.post("/admin/reviews/101/approve")
    invalid = client.post(
        "/admin/reviews/101/approve",
        headers={"X-Admin-Key": "wrong-key"},
    )

    assert missing.status_code == 401
    assert invalid.status_code == 403


@pytest.mark.parametrize(
    ("route", "decision", "ticket_status"),
    [
        ("approve", "approve", "APPROVED"),
        ("deny", "deny", "DENIED"),
    ],
)
def test_admin_endpoint_resumes_ticket_thread(
    monkeypatch,
    route,
    decision,
    ticket_status,
):
    ticket = {
        "ticket_id": 101,
        "order_id": "order-1",
        "issue_type": "order_cancellation",
        "status": "PENDING_REVIEW",
        "thread_id": "saved-review-thread",
    }

    class FakeGraph:
        def invoke(self, command, config):
            assert command.resume == {"decision": decision, "ticket_id": 101}
            assert config["configurable"]["thread_id"] == "saved-review-thread"
            ticket["status"] = ticket_status
            return {"messages": [AIMessage(content=f"Review {decision}d.")]}

    api_module = load_api_with_fake_graph(monkeypatch, FakeGraph())
    monkeypatch.setattr(api_module, "ADMIN_API_KEY", "test-admin-key")
    monkeypatch.setattr(
        api_module,
        "fetch_support_ticket_by_id",
        lambda ticket_id: dict(ticket),
    )
    client = TestClient(api_module.app)

    response = client.post(
        f"/admin/reviews/101/{route}",
        headers={"X-Admin-Key": "test-admin-key"},
    )

    assert response.status_code == 200
    assert response.json()["decision"] == decision
    assert response.json()["ticket_status"] == ticket_status


def test_order_status_and_rag_tools_still_work(monkeypatch):
    monkeypatch.setattr(
        order_status,
        "fetch_order_by_id",
        lambda order_id: {"order_id": order_id, "order_status": "processing"},
    )
    monkeypatch.setattr(
        support_docs,
        "retrieve_support_chunks",
        lambda query: [
            Document(
                page_content="Returns are accepted within 30 days.",
                metadata={"source": "returns.md"},
            )
        ],
    )

    status_result = order_status.get_order_status.invoke({"order_id": "order-1"})
    rag_result = support_docs.search_support_docs.invoke({"query": "returns"})

    assert status_result == "Order order-1 is processing."
    assert "Source: returns.md" in rag_result
