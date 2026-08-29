import json
import secrets
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel

from src.agent.graph import graph
from src.config import ADMIN_API_KEY
from src.db.database import (
    fetch_pending_cancellation_ticket_by_thread_id,
    fetch_support_ticket_by_id,
    update_support_ticket_status,
)
from src.tools.order_cancellation import (
    PENDING_REVIEW,
    REJECTED,
    create_pending_review_after_interrupt,
)


app = FastAPI(title="Customer Support Agent")


class ChatRequest(BaseModel):
    message: str
    thread_id: str


class ChatResponse(BaseModel):
    response: str | None = None
    approval_required: bool = False
    interrupt: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str


class AdminReviewResponse(BaseModel):
    ticket_id: int
    order_id: str
    decision: str
    ticket_status: str
    response: str


def require_admin_api_key(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> None:
    if x_admin_key is None:
        raise HTTPException(status_code=401, detail="X-Admin-Key is required.")
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="ADMIN_API_KEY is not configured.")
    if not secrets.compare_digest(x_admin_key, ADMIN_API_KEY):
        raise HTTPException(status_code=403, detail="Invalid admin API key.")


def _pending_review_payload(ticket: dict) -> dict[str, Any]:
    return {
        "outcome": "pending_admin_review",
        "requested_action": "cancel_order",
        "order_id": ticket["order_id"],
        "reason": ticket.get("review_reason"),
        "ticket_id": ticket["ticket_id"],
        "status": PENDING_REVIEW,
    }


def _pending_review_from_messages(messages: list) -> dict | None:
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == "cancel_order":
            try:
                result = json.loads(message.content)
            except (TypeError, json.JSONDecodeError):
                continue
            if result.get("outcome") == "pending_admin_review":
                return result
    return None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    config = {"configurable": {"thread_id": request.thread_id}}
    pending_ticket = fetch_pending_cancellation_ticket_by_thread_id(
        request.thread_id
    )
    if pending_ticket is not None:
        return ChatResponse(
            approval_required=True,
            interrupt=_pending_review_payload(pending_ticket),
        )

    result = graph.invoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config,
    )

    if result.get("__interrupt__"):
        interrupt_payload = dict(result["__interrupt__"][0].value)
        try:
            ticket, _ = create_pending_review_after_interrupt(
                interrupt_payload,
                request.thread_id,
            )
        except Exception as error:
            pending_ticket = fetch_pending_cancellation_ticket_by_thread_id(
                request.thread_id
            )
            if pending_ticket is not None:
                update_support_ticket_status(
                    pending_ticket["ticket_id"],
                    REJECTED,
                    expected_status=PENDING_REVIEW,
                )
            try:
                graph.invoke(
                    Command(
                        resume={"decision": "setup_failed", "ticket_id": None}
                    ),
                    config=config,
                )
            except Exception:
                pass
            raise HTTPException(
                status_code=503,
                detail="The admin review could not be created safely.",
            ) from error

        return ChatResponse(
            approval_required=True,
            interrupt=_pending_review_payload(ticket),
        )

    newest_response = result["messages"][-1].content
    pending_result = _pending_review_from_messages(result["messages"])
    if pending_result is not None:
        return ChatResponse(
            response=str(newest_response),
            approval_required=True,
            interrupt=pending_result,
        )

    return ChatResponse(response=str(newest_response))


def _resume_admin_review(ticket_id: int, decision: str) -> AdminReviewResponse:
    ticket = fetch_support_ticket_by_id(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Support ticket not found.")
    if ticket["issue_type"] != "order_cancellation":
        raise HTTPException(
            status_code=400,
            detail="This support ticket is not a cancellation review.",
        )
    if ticket["status"] != PENDING_REVIEW:
        raise HTTPException(
            status_code=409,
            detail=f"This review is already {ticket['status']}.",
        )
    if not ticket.get("thread_id"):
        raise HTTPException(
            status_code=409,
            detail="This review does not have a LangGraph thread_id.",
        )

    config = {"configurable": {"thread_id": ticket["thread_id"]}}
    result = graph.invoke(
        Command(resume={"decision": decision, "ticket_id": ticket_id}),
        config=config,
    )

    if result.get("__interrupt__"):
        raise HTTPException(
            status_code=409,
            detail="The graph is still waiting for an admin decision.",
        )

    updated_ticket = fetch_support_ticket_by_id(ticket_id)
    newest_response = str(result["messages"][-1].content)

    return AdminReviewResponse(
        ticket_id=ticket_id,
        order_id=ticket["order_id"],
        decision=decision,
        ticket_status=updated_ticket["status"],
        response=newest_response,
    )


@app.post(
    "/admin/reviews/{ticket_id}/approve",
    response_model=AdminReviewResponse,
    dependencies=[Depends(require_admin_api_key)],
)
def approve_cancellation(ticket_id: int) -> AdminReviewResponse:
    return _resume_admin_review(ticket_id, "approve")


@app.post(
    "/admin/reviews/{ticket_id}/deny",
    response_model=AdminReviewResponse,
    dependencies=[Depends(require_admin_api_key)],
)
def deny_cancellation(ticket_id: int) -> AdminReviewResponse:
    return _resume_admin_review(ticket_id, "deny")
