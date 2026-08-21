from langchain_core.tools import tool
from langgraph.types import interrupt

from src.db.database import cancel_order_by_id, fetch_order_by_id

@tool
def cancel_order(order_id: str) -> str:
    """
    Handle any explicit user request to cancel an order.

    Use this tool whenever the user asks to cancel an order,
    regardless of the order's current status.

    This tool performs the deterministic eligibility checks itself
    and safely rejects orders that cannot be cancelled.
    """

    order = fetch_order_by_id(order_id)

    if order is None:
        return f"Order {order_id} was not found, so it was not cancelled."

    current_status = order["order_status"].lower()

    if current_status in {"delivered", "cancelled", "canceled"}:
        return f"Order {order_id} is already {current_status} and cannot be cancelled."

    if current_status != "processing":
        return (
            f"Order {order_id} is {current_status}. "
            "Only processing orders can be cancelled."
        )

    approved = interrupt(
        {
            "requested_action": "cancel order",
            "order_id": order_id,
            "question": f"Do you approve cancelling order {order_id}?",
        }
    )

    if approved is not True:
        return f"Cancellation of order {order_id} was rejected. No changes were made."

    cancelled_order = cancel_order_by_id(order_id)

    if cancelled_order is None:
        return (
            f"Order {order_id} was not cancelled because its status changed "
            "before the update."
        )

    return f"Order {order_id} was successfully cancelled."
