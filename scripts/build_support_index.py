from src.rag.documents import load_support_documents, split_documents
from src.rag.vector_store import build_vector_store


def main() -> None:
    documents = load_support_documents()
    chunks = split_documents(documents)
    build_vector_store(chunks)
    print(f"Indexed {len(chunks)} support-policy chunks.")


if __name__ == "__main__":
    main()
