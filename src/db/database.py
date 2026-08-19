from sqlalchemy import Column, DateTime, MetaData, String, Table, create_engine, select

from src.config import DATABASE_URL


if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set. Add it to your .env file.")


engine = create_engine(DATABASE_URL)
metadata = MetaData()

orders_table = Table(
    "orders",
    metadata,
    Column("order_id", String(32), primary_key=True),
    Column("customer_id", String(32), nullable=False),
    Column("order_status", String(32), nullable=False),
    Column("order_purchase_timestamp", DateTime, nullable=False),
    Column("order_approved_at", DateTime),
    Column("order_delivered_carrier_date", DateTime),
    Column("order_delivered_customer_date", DateTime),
    Column("order_estimated_delivery_date", DateTime, nullable=False),
)


def fetch_order_by_id(order_id: str) -> dict | None:
    query = select(orders_table).where(orders_table.c.order_id == order_id)

    with engine.connect() as connection:
        order = connection.execute(query).mappings().first()

    return dict(order) if order else None
