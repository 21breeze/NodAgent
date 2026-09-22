from pathlib import Path
from typing import List

from langchain_core.documents import Document as LangChainDocument
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_document(file_path: str) -> List[LangChainDocument]:
    path = Path(file_path)

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))

    elif suffix in {".txt", ".md"}:
        loader = TextLoader(
            str(path),
            encoding="utf-8",
        )

    else:
        raise ValueError(
            f"Unsupported file type: {suffix}"
        )

    return loader.load()


def split_documents(
    documents: List[LangChainDocument],
) -> List[LangChainDocument]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return text_splitter.split_documents(
        documents
    )


def parse_and_split_document(
    file_path: str,
) -> List[LangChainDocument]:
    documents = load_document(
        file_path=file_path
    )

    chunks = split_documents(
        documents=documents
    )

    return chunks