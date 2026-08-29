from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from src.db.database import (
    cancel_order_by_id,
    fetch_denied_cancellation_ticket,
    fetch_order_by_id,
    fetch_pending_cancellation_ticket,
    get_or_create_pending_cancellation_ticket,
    update_support_ticket_status,
)


AUTO_CANCELLABLE_STATUSES = {"created", "approved", "processing"}
ADMIN_REVIEW_STATUSES = {"invoiced"}
HARD_REJECT_STATUSES = {
    "shipped",
    "delivered",
    "canceled",
    "cancelled",
    "unavailable",
}

PENDING_REVIEW = "PENDING_REVIEW"
APPROVED = "APPROVED"
DENIED = "DENIED"
REJECTED = "REJECTED"


def cancellation_policy(order_status: str) -> str:
    """Return the deterministic cancellation branch for an order status."""
    normalized_status = order_status.lower()

    if normalized_status in AUTO_CANCELLABLE_STATUSES:
        return "auto_cancel"
    if normalized_status in ADMIN_REVIEW_STATUSES:
        return "admin_review"
    if normalized_status in HARD_REJECT_STATUSES:
        return "hard_reject"
    return "hard_reject"


def _latest_user_request(runtime: ToolRuntime) -> str:
    for message in reversed(runtime.state["messages"]):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return "Customer requested cancellation."


def _thread_id(runtime: ToolRuntime) -> str:
    thread_id = runtime.config.get("configurable", {}).get("thread_id")
    return str(thread_id) if thread_id else ""


def create_pending_review_after_interrupt(
    interrupt_payload: dict[str, Any],
    thread_id: str,
) -> tuple[dict, bool]:
    """Create the ticket after LangGraph has saved the interrupt checkpoint."""
    return get_or_create_pending_cancellation_ticket(
        order_id=interrupt_payload["order_id"],
        thread_id=thread_id,
        review_reason=interrupt_payload["reason"],
        request_details=interrupt_payload["request_details"],
    )


def _close_pending_review(order_id: str, status: str) -> int | None:
    ticket = fetch_pending_cancellation_ticket(order_id)
    if ticket is None:
        return None

    update_support_ticket_status(
        ticket_id=ticket["ticket_id"],
        status=status,
        expected_status=PENDING_REVIEW,
    )
    return ticket["ticket_id"]


def _result(
    outcome: str,
    order_id: str,
    reason: str,
    status: str | None,
    ticket_id: int | None = None,
    ticket_status: str | None = None,
) -> dict[str, Any]:
    return {
        "outcome": outcome,
        "order_id": order_id,
        "reason": reason,
        "ticket_id": ticket_id,
        "status": status,
        "ticket_status": ticket_status,
    }


@tool
def cancel_order(order_id: str, runtime: ToolRuntime) -> dict[str, Any]:
    """
    Handle any explicit user request to cancel an order.

    Use this tool whenever the user asks to cancel an order. Python applies the
    cancellation policy; do not decide eligibility or approval yourself.
    """
    order = fetch_order_by_id(order_id)

    if order is None:
        ticket_id = _close_pending_review(order_id, REJECTED)
        return _result(
            outcome="not_found",
            order_id=order_id,
            reason="The order was not found, so no cancellation was performed.",
            status=None,
            ticket_id=ticket_id,
            ticket_status=REJECTED if ticket_id else None,
        )

    current_status = order["order_status"].lower()
    policy = cancellation_policy(current_status)

    if policy == "hard_reject":
        ticket_id = _close_pending_review(order_id, REJECTED)
        return _result(
            outcome="rejected",
            order_id=order_id,
            reason=(
                f"Orders with status '{current_status}' cannot be cancelled "
                "through this workflow."
            ),
            status=current_status,
            ticket_id=ticket_id,
            ticket_status=REJECTED if ticket_id else None,
        )

    if policy == "auto_cancel":
        cancelled_order = cancel_order_by_id(
            order_id,
            allowed_statuses=AUTO_CANCELLABLE_STATUSES,
        )
        if cancelled_order is None:
            latest_order = fetch_order_by_id(order_id)
            latest_status = (
                latest_order["order_status"].lower() if latest_order else None
            )
            return _result(
                outcome="rejected",
                order_id=order_id,
                reason="The order status changed before cancellation completed.",
                status=latest_status,
            )

        return _result(
            outcome="cancelled",
            order_id=order_id,
            reason="The order status allowed automatic cancellation.",
            status=cancelled_order["order_status"],
        )

    thread_id = _thread_id(runtime)
    if not thread_id:
        return _result(
            outcome="rejected",
            order_id=order_id,
            reason="Admin review requires a LangGraph thread_id.",
            status=current_status,
        )

    review_reason = (
        f"Orders with status '{current_status}' require admin review."
    )
    denied_ticket = fetch_denied_cancellation_ticket(order_id)
    if denied_ticket is not None:
        stale_ticket = fetch_pending_cancellation_ticket(order_id)
        if stale_ticket is not None:
            update_support_ticket_status(
                stale_ticket["ticket_id"],
                REJECTED,
                expected_status=PENDING_REVIEW,
            )
        return _result(
            outcome="previously_denied",
            order_id=order_id,
            reason=(
                "Your cancellation request was previously denied by an "
                "administrator."
            ),
            status=current_status,
            ticket_id=denied_ticket["ticket_id"],
            ticket_status=DENIED,
        )

    ticket = fetch_pending_cancellation_ticket(order_id)

    if ticket is not None and ticket.get("thread_id") != thread_id:
        return _result(
            outcome="pending_admin_review",
            order_id=order_id,
            reason="A cancellation review is already pending.",
            status=current_status,
            ticket_id=ticket["ticket_id"],
            ticket_status=PENDING_REVIEW,
        )

    decision = interrupt(
        {
            "outcome": "pending_admin_review",
            "requested_action": "cancel_order",
            "order_id": order_id,
            "reason": review_reason,
            "request_details": _latest_user_request(runtime),
            "ticket_id": ticket["ticket_id"] if ticket else None,
            "status": PENDING_REVIEW,
            "question": (
                f"Should an authorized admin approve cancellation of order "
                f"{order_id}?"
            ),
        }
    )

    if isinstance(decision, dict) and decision.get("decision") == "setup_failed":
        return _result(
            outcome="review_setup_failed",
            order_id=order_id,
            reason="The admin review could not be created safely.",
            status=current_status,
        )

    ticket = fetch_pending_cancellation_ticket(order_id)
    if ticket is None:
        return _result(
            outcome="review_setup_failed",
            order_id=order_id,
            reason="No active admin review ticket was found.",
            status=current_status,
        )

    approved = (
        isinstance(decision, dict)
        and decision.get("decision") == "approve"
        and decision.get("ticket_id") == ticket["ticket_id"]
    )

    if not approved:
        update_support_ticket_status(
            ticket_id=ticket["ticket_id"],
            status=DENIED,
            expected_status=PENDING_REVIEW,
        )
        return _result(
            outcome="admin_denied",
            order_id=order_id,
            reason="An admin denied the cancellation request.",
            status=current_status,
            ticket_id=ticket["ticket_id"],
            ticket_status=DENIED,
        )

    latest_order = fetch_order_by_id(order_id)
    latest_status = (
        latest_order["order_status"].lower() if latest_order else None
    )
    if latest_status not in ADMIN_REVIEW_STATUSES:
        update_support_ticket_status(
            ticket_id=ticket["ticket_id"],
            status=REJECTED,
            expected_status=PENDING_REVIEW,
        )
        return _result(
            outcome="rejected" if latest_order else "not_found",
            order_id=order_id,
            reason=(
                "The order was no longer eligible when the admin decision "
                "was applied."
            ),
            status=latest_status,
            ticket_id=ticket["ticket_id"],
            ticket_status=REJECTED,
        )

    cancelled_order = cancel_order_by_id(
        order_id,
        allowed_statuses=ADMIN_REVIEW_STATUSES,
        review_ticket_id=ticket["ticket_id"],
    )
    if cancelled_order is None:
        update_support_ticket_status(
            ticket_id=ticket["ticket_id"],
            status=REJECTED,
            expected_status=PENDING_REVIEW,
        )
        return _result(
            outcome="rejected",
            order_id=order_id,
            reason="The order status changed before cancellation completed.",
            status=latest_status,
            ticket_id=ticket["ticket_id"],
            ticket_status=REJECTED,
        )

    return _result(
        outcome="approved_and_cancelled",
        order_id=order_id,
        reason="An admin approved the request and the order was cancelled.",
        status=cancelled_order["order_status"],
        ticket_id=ticket["ticket_id"],
        ticket_status=APPROVED,
    )
