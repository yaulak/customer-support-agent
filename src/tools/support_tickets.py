from langchain_core.tools import tool

from src.db.database import insert_support_ticket


@tool
def create_support_ticket(
    description: str,
    issue_type: str,
    order_id: str | None = None,
) -> str:
    """Create a support ticket for an issue that needs customer-support follow-up."""
    ticket = insert_support_ticket(
        description=description,
        issue_type=issue_type,
        order_id=order_id,
    )
    ticket_order_id = ticket["order_id"] or "not provided"

    return (
        f"Created support ticket {ticket['ticket_id']}. "
        f"Issue type: {ticket['issue_type']}. "
        f"Order ID: {ticket_order_id}. "
        f"Status: {ticket['status']}. "
        f"Description: {ticket['description']}"
    )
