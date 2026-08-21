from langchain_core.tools import tool

from src.db.database import fetch_order_by_id

@tool
def get_order_status(order_id: str) -> str:
    """
    Look up the current status of an order.

    Use this tool for order-status or order-information questions.
    Do not use this tool to handle an explicit request to cancel an order.
    """
    order = fetch_order_by_id(order_id)

    if order is None:
        return f"Order {order_id} was not found."

    return f"Order {order_id} is {order['order_status']}."
