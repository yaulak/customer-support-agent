from pathlib import PurePosixPath

import boto3
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    AWS_ACCESS_KEY_ID,
    AWS_REGION,
    AWS_SECRET_ACCESS_KEY,
    S3_BUCKET_NAME,
)


def load_support_documents() -> list[Document]:
    if not S3_BUCKET_NAME:
        raise ValueError("S3_BUCKET_NAME is not set. Add it to your .env file.")

    s3_client = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    paginator = s3_client.get_paginator("list_objects_v2")
    markdown_keys = []

    for page in paginator.paginate(Bucket=S3_BUCKET_NAME):
        for s3_object in page.get("Contents", []):
            key = s3_object["Key"]
            if key.lower().endswith(".md"):
                markdown_keys.append(key)

    documents = []

    for key in sorted(markdown_keys):
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        text = response["Body"].read().decode("utf-8")
        documents.append(
            Document(
                page_content=text,
                metadata={"source": PurePosixPath(key).name},
            )
        )

    if not documents:
        raise FileNotFoundError(
            f"No Markdown files found in S3 bucket {S3_BUCKET_NAME}."
        )

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
    )
    return text_splitter.split_documents(documents)
