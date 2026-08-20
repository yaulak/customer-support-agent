from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORT_DOCS_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "data" / "support_docs"
)


def load_support_documents() -> list[Document]:
    documents = []

    for path in sorted(SUPPORT_DOCS_DIRECTORY.glob("*.md")):
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": path.name},
            )
        )

    if not documents:
        raise FileNotFoundError(f"No Markdown files found in {SUPPORT_DOCS_DIRECTORY}")

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
    )
    return text_splitter.split_documents(documents)
