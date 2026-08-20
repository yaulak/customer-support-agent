from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    insert,
    select,
    update,
)

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

support_tickets_table = Table(
    "support_tickets",
    metadata,
    Column("ticket_id", Integer, primary_key=True, autoincrement=True),
    Column("order_id", String(32)),
    Column("issue_type", String(64), nullable=False),
    Column("description", Text, nullable=False),
    Column("status", String(32), nullable=False, server_default="open"),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    ),
)


def fetch_order_by_id(order_id: str) -> dict | None:
    query = select(orders_table).where(orders_table.c.order_id == order_id)

    with engine.connect() as connection:
        order = connection.execute(query).mappings().first()

    return dict(order) if order else None


def insert_support_ticket(
    description: str,
    issue_type: str,
    order_id: str | None = None,
) -> dict:
    statement = (
        insert(support_tickets_table)
        .values(
            order_id=order_id,
            issue_type=issue_type,
            description=description,
        )
        .returning(*support_tickets_table.c)
    )

    with engine.begin() as connection:
        ticket = connection.execute(statement).mappings().one()

    return dict(ticket)


def cancel_order_by_id(order_id: str) -> dict | None:
    statement = (
        update(orders_table)
        .where(
            orders_table.c.order_id == order_id,
            orders_table.c.order_status == "processing",
        )
        .values(order_status="cancelled")
        .returning(*orders_table.c)
    )

    with engine.begin() as connection:
        order = connection.execute(statement).mappings().first()

    return dict(order) if order else None
