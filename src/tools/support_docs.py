from langchain_core.tools import tool

from src.rag.vector_store import retrieve_support_chunks


@tool
def search_support_docs(query: str) -> str:
    """Search company policies about returns, refunds, shipping, and cancellations."""
    chunks = retrieve_support_chunks(query)

    if not chunks:
        return "No relevant support policy was found."

    results = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        results.append(f"Source: {source}\n{chunk.page_content}")

    return "\n\n---\n\n".join(results)
