from langchain_core.tools import tool


# Temporary learning data. This will be replaced by a real e-commerce dataset
# stored in PostgreSQL on AWS RDS in a later project step.
ORDERS = {
    "1001": {"order_id": "1001", "status": "processing"},
    "1002": {"order_id": "1002", "status": "shipped"},
    "1003": {"order_id": "1003", "status": "delivered"},
    "1004": {"order_id": "1004", "status": "cancelled"},
    "1005": {"order_id": "1005", "status": "shipped"},
}


@tool
def get_order_status(order_id: str) -> str:
    """Look up the current status of an order by its order ID."""
    order = ORDERS.get(order_id)

    if order is None:
        return f"Order {order_id} was not found."

    return f"Order {order_id} is {order['status']}."
