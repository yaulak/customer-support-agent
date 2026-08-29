from src.db.database import initialize_database


def main() -> None:
    initialize_database()
    print("Database tables are ready.")


if __name__ == "__main__":
    main()
