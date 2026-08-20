import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import OPENAI_API_KEY


CHROMA_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "chroma"
)
COLLECTION_NAME = "support-policies"


def create_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )


def build_vector_store(chunks: list[Document]) -> None:
    if CHROMA_DIRECTORY.exists():
        shutil.rmtree(CHROMA_DIRECTORY)

    Chroma.from_documents(
        documents=chunks,
        embedding=create_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIRECTORY),
    )


def retrieve_support_chunks(query: str, number_of_results: int = 3) -> list[Document]:
    if not CHROMA_DIRECTORY.exists():
        raise FileNotFoundError(
            "Support document index not found. Run the index-building script first."
        )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=create_embeddings(),
        persist_directory=str(CHROMA_DIRECTORY),
    )
    return vector_store.similarity_search(query, k=number_of_results)
