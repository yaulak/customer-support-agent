import csv
from datetime import datetime
from pathlib import Path

from src.db.database import engine, metadata, orders_table


CSV_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "olist"
    / "olist_orders_dataset.csv"
)


def parse_timestamp(value: str) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def load_orders() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    orders = []

    with CSV_PATH.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            orders.append(
                {
                    "order_id": row["order_id"],
                    "customer_id": row["customer_id"],
                    "order_status": row["order_status"],
                    "order_purchase_timestamp": parse_timestamp(
                        row["order_purchase_timestamp"]
                    ),
                    "order_approved_at": parse_timestamp(row["order_approved_at"]),
                    "order_delivered_carrier_date": parse_timestamp(
                        row["order_delivered_carrier_date"]
                    ),
                    "order_delivered_customer_date": parse_timestamp(
                        row["order_delivered_customer_date"]
                    ),
                    "order_estimated_delivery_date": parse_timestamp(
                        row["order_estimated_delivery_date"]
                    ),
                }
            )

    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(orders_table.insert(), orders)

    print(f"Loaded {len(orders)} orders into PostgreSQL.")


if __name__ == "__main__":
    load_orders()
