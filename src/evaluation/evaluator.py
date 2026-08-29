import json
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

from src.agent.graph import graph


DATASET_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "evaluation" / "cases.jsonl"
)


def load_cases() -> list[dict]:
    with DATASET_PATH.open(encoding="utf-8") as dataset_file:
        return [json.loads(line) for line in dataset_file if line.strip()]


def fake_insert_support_ticket(
    description: str,
    issue_type: str,
    order_id: str | None = None,
) -> dict:
    return {
        "ticket_id": "evaluation-ticket",
        "order_id": order_id,
        "issue_type": issue_type,
        "description": description,
        "status": "open",
    }


def fake_cancel_order(
    order_id: str,
    allowed_statuses=None,
    review_ticket_id: int | None = None,
) -> dict:
    return {"order_id": order_id, "order_status": "cancelled"}


def fake_pending_cancellation_ticket(
    order_id: str,
    thread_id: str,
    review_reason: str,
    request_details: str,
) -> tuple[dict, bool]:
    return (
        {
            "ticket_id": 1,
            "order_id": order_id,
            "issue_type": "order_cancellation",
            "status": "PENDING_REVIEW",
            "thread_id": thread_id,
            "review_reason": review_reason,
            "request_details": request_details,
        },
        True,
    )


def fake_update_ticket_status(
    ticket_id: int,
    status: str,
    expected_status: str | None = None,
) -> dict:
    return {"ticket_id": ticket_id, "status": status}


def get_called_tools(messages: list) -> list[str]:
    called_tools = []

    for message in messages:
        for tool_call in getattr(message, "tool_calls", []):
            called_tools.append(tool_call["name"])

    return called_tools


def get_returned_sources(messages: list) -> list[str]:
    sources = []

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue

        for line in str(message.content).splitlines():
            if line.startswith("Source: "):
                sources.append(line.removeprefix("Source: "))

    return sources


def run_case(case: dict) -> dict:
    config = {
        "configurable": {
            "thread_id": f"evaluation-{case['id']}-{uuid4().hex}",
        }
    }
    result = graph.invoke(
        {"messages": [HumanMessage(content=case["user_input"])]},
        config=config,
    )

    while result.get("__interrupt__"):
        interrupt_payload = result["__interrupt__"][0].value
        result = graph.invoke(
            Command(
                resume={
                    "decision": "deny",
                    "ticket_id": interrupt_payload["ticket_id"],
                }
            ),
            config=config,
        )

    messages = result["messages"]
    called_tools = get_called_tools(messages)
    returned_sources = get_returned_sources(messages)
    expected_tool = case["expected_tool"]
    expected_source = case["expected_source"]

    tool_passed = (
        expected_tool in called_tools if expected_tool else not called_tools
    )
    source_passed = (
        expected_source in returned_sources if expected_source else None
    )

    return {
        "id": case["id"],
        "expected_behavior": case["expected_behavior"],
        "expected_tool": expected_tool,
        "called_tools": called_tools,
        "tool_passed": tool_passed,
        "expected_source": expected_source,
        "returned_sources": returned_sources,
        "source_passed": source_passed,
    }


def print_summary(results: list[dict]) -> None:
    total_cases = len(results)
    tool_passes = sum(result["tool_passed"] for result in results)
    rag_results = [
        result for result in results if result["expected_source"] is not None
    ]
    rag_passes = sum(result["source_passed"] for result in rag_results)
    tool_accuracy = tool_passes / total_cases if total_cases else 0
    rag_accuracy = rag_passes / len(rag_results) if rag_results else 0
    failures = [
        result
        for result in results
        if not result["tool_passed"] or result["source_passed"] is False
    ]

    print("\nEvaluation Summary")
    print(f"Total cases: {total_cases}")
    print(
        "Tool-routing accuracy: "
        f"{tool_passes}/{total_cases} ({tool_accuracy:.1%})"
    )
    print(
        "RAG source-hit accuracy: "
        f"{rag_passes}/{len(rag_results)} ({rag_accuracy:.1%})"
    )

    if not failures:
        print("Failures: none")
        return

    print("Failures:")
    for failure in failures:
        if failure.get("error"):
            print(f"- {failure['id']}: {failure['error']}")
            print(f"  expected behavior: {failure['expected_behavior']}")
            continue

        expected_tool = failure["expected_tool"] or "no tool"
        called_tools = failure["called_tools"] or ["no tool"]
        print(
            f"- {failure['id']}: expected tool {expected_tool}; "
            f"called {', '.join(called_tools)}"
        )
        if failure["expected_source"]:
            returned_sources = failure["returned_sources"] or ["no sources"]
            print(
                f"  expected source {failure['expected_source']}; "
                f"returned {', '.join(returned_sources)}"
            )
        print(f"  expected behavior: {failure['expected_behavior']}")


def main() -> None:
    cases = load_cases()

    # Exercise action routing without writing tickets or changing orders.
    with patch(
        "src.tools.support_tickets.insert_support_ticket",
        fake_insert_support_ticket,
    ), patch(
        "src.tools.order_cancellation.cancel_order_by_id",
        fake_cancel_order,
    ), patch(
        "src.tools.order_cancellation.get_or_create_pending_cancellation_ticket",
        fake_pending_cancellation_ticket,
    ), patch(
        "src.tools.order_cancellation.update_support_ticket_status",
        fake_update_ticket_status,
    ):
        results = []
        for case in cases:
            print(f"Running {case['id']}...")
            try:
                results.append(run_case(case))
            except Exception as error:
                results.append(
                    {
                        "id": case["id"],
                        "expected_behavior": case["expected_behavior"],
                        "expected_tool": case["expected_tool"],
                        "called_tools": [],
                        "tool_passed": False,
                        "expected_source": case["expected_source"],
                        "returned_sources": [],
                        "source_passed": (
                            False if case["expected_source"] else None
                        ),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    print_summary(results)


if __name__ == "__main__":
    main()
