from collections.abc import Collection

from sqlalchemy import (
    Index,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    create_engine,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import insert as postgresql_insert

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
    Column("thread_id", String(255)),
    Column("review_reason", Text),
    Column("request_details", Text),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    ),
)

pending_cancellation_ticket_index = Index(
    "uq_pending_cancellation_ticket_per_order",
    support_tickets_table.c.order_id,
    unique=True,
    postgresql_where=and_(
        support_tickets_table.c.issue_type == "order_cancellation",
        support_tickets_table.c.status == "PENDING_REVIEW",
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
    status: str = "open",
    thread_id: str | None = None,
    review_reason: str | None = None,
    request_details: str | None = None,
) -> dict:
    statement = (
        insert(support_tickets_table)
        .values(
            order_id=order_id,
            issue_type=issue_type,
            description=description,
            status=status,
            thread_id=thread_id,
            review_reason=review_reason,
            request_details=request_details,
        )
        .returning(*support_tickets_table.c)
    )

    with engine.begin() as connection:
        ticket = connection.execute(statement).mappings().one()

    return dict(ticket)


def fetch_support_ticket_by_id(ticket_id: int) -> dict | None:
    query = select(support_tickets_table).where(
        support_tickets_table.c.ticket_id == ticket_id
    )

    with engine.connect() as connection:
        ticket = connection.execute(query).mappings().first()

    return dict(ticket) if ticket else None


def fetch_pending_cancellation_ticket(order_id: str) -> dict | None:
    query = select(support_tickets_table).where(
        support_tickets_table.c.order_id == order_id,
        support_tickets_table.c.issue_type == "order_cancellation",
        support_tickets_table.c.status == "PENDING_REVIEW",
    )

    with engine.connect() as connection:
        ticket = connection.execute(query).mappings().first()

    return dict(ticket) if ticket else None


def get_or_create_pending_cancellation_ticket(
    order_id: str,
    thread_id: str,
    review_reason: str,
    request_details: str,
) -> tuple[dict, bool]:
    existing_ticket = fetch_pending_cancellation_ticket(order_id)
    if existing_ticket is not None:
        return existing_ticket, False

    description = (
        f"Cancellation review for order {order_id}. "
        f"Reason: {review_reason} Request: {request_details}"
    )
    statement = (
        postgresql_insert(support_tickets_table)
        .values(
            order_id=order_id,
            issue_type="order_cancellation",
            description=description,
            status="PENDING_REVIEW",
            thread_id=thread_id,
            review_reason=review_reason,
            request_details=request_details,
        )
        .on_conflict_do_nothing(
            index_elements=[support_tickets_table.c.order_id],
            index_where=and_(
                support_tickets_table.c.issue_type == "order_cancellation",
                support_tickets_table.c.status == "PENDING_REVIEW",
            ),
        )
        .returning(*support_tickets_table.c)
    )

    with engine.begin() as connection:
        ticket = connection.execute(statement).mappings().first()

    if ticket is not None:
        return dict(ticket), True

    # Another request created the pending ticket at the same time.
    existing_ticket = fetch_pending_cancellation_ticket(order_id)
    if existing_ticket is None:
        raise RuntimeError("Could not create or find the cancellation review ticket.")

    return existing_ticket, False


def update_support_ticket_status(
    ticket_id: int,
    status: str,
    expected_status: str | None = None,
) -> dict | None:
    statement = update(support_tickets_table).where(
        support_tickets_table.c.ticket_id == ticket_id
    )
    if expected_status is not None:
        statement = statement.where(
            support_tickets_table.c.status == expected_status
        )

    statement = statement.values(status=status).returning(
        *support_tickets_table.c
    )

    with engine.begin() as connection:
        ticket = connection.execute(statement).mappings().first()

    return dict(ticket) if ticket else None


def cancel_order_by_id(
    order_id: str,
    allowed_statuses: Collection[str] = ("processing",),
    review_ticket_id: int | None = None,
) -> dict | None:
    normalized_statuses = [status.lower() for status in allowed_statuses]
    statement = (
        update(orders_table)
        .where(
            orders_table.c.order_id == order_id,
            func.lower(orders_table.c.order_status).in_(normalized_statuses),
        )
        .values(order_status="canceled")
        .returning(*orders_table.c)
    )

    with engine.begin() as connection:
        order = connection.execute(statement).mappings().first()

        if order is not None and review_ticket_id is not None:
            ticket_statement = (
                update(support_tickets_table)
                .where(
                    support_tickets_table.c.ticket_id == review_ticket_id,
                    support_tickets_table.c.order_id == order_id,
                    support_tickets_table.c.issue_type == "order_cancellation",
                    support_tickets_table.c.status == "PENDING_REVIEW",
                )
                .values(status="APPROVED")
                .returning(support_tickets_table.c.ticket_id)
            )
            updated_ticket_id = connection.execute(
                ticket_statement
            ).scalar_one_or_none()
            if updated_ticket_id is None:
                raise RuntimeError(
                    "The pending cancellation ticket could not be approved."
                )

    return dict(order) if order else None


def initialize_database() -> None:
    metadata.create_all(engine)

    # create_all() does not add new columns to an existing table.
    with engine.begin() as connection:
        connection.execute(
            text(
                "ALTER TABLE support_tickets "
                "ADD COLUMN IF NOT EXISTS thread_id VARCHAR(255)"
            )
        )
        connection.execute(
            text(
                "ALTER TABLE support_tickets "
                "ADD COLUMN IF NOT EXISTS review_reason TEXT"
            )
        )
        connection.execute(
            text(
                "ALTER TABLE support_tickets "
                "ADD COLUMN IF NOT EXISTS request_details TEXT"
            )
        )
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "uq_pending_cancellation_ticket_per_order "
                "ON support_tickets (order_id) "
                "WHERE issue_type = 'order_cancellation' "
                "AND status = 'PENDING_REVIEW'"
            )
        )
