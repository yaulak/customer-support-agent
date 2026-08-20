from src.db.database import engine, metadata


def main() -> None:
    metadata.create_all(engine)
    print("Database tables are ready.")


if __name__ == "__main__":
    main()
