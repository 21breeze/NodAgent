from pathlib import Path
from typing import List

from fastapi import (
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from backend.app.models.document import Document
from backend.app.models.document_chunk import (
    DocumentChunk,
)
from backend.app.models.workspace import Workspace
from backend.app.services.document_parser import (
    parse_and_split_document,
)
from backend.app.services.embedding_service import (
    embedding_model,
)


UPLOAD_DIR = Path("data/uploads")


def get_documents(
    db: Session,
    workspace_id: int,
) -> List[Document]:
    workspace = (
        db.query(Workspace)
        .filter(
            Workspace.id == workspace_id
        )
        .first()
    )

    if workspace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found",
        )

    return (
        db.query(Document)
        .filter(
            Document.workspace_id == workspace_id
        )
        .order_by(Document.id)
        .all()
    )


def create_document(
    db: Session,
    workspace_id: int,
    file: UploadFile,
) -> Document:
    workspace = (
        db.query(Workspace)
        .filter(
            Workspace.id == workspace_id
        )
        .first()
    )

    if workspace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found",
        )

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = (
        UPLOAD_DIR
        / file.filename
    )

    with file_path.open("wb") as buffer:
        while True:
            chunk = file.file.read(
                1024 * 1024
            )

            if not chunk:
                break

            buffer.write(chunk)

    document = Document(
        workspace_id=workspace_id,
        filename=file.filename,
        file_path=str(file_path),
        content_type=file.content_type,
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    return document


def process_document(
    db: Session,
    workspace_id: int,
    document_id: int,
) -> int:
    document = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.workspace_id == workspace_id,
        )
        .first()
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    try:
        chunks = parse_and_split_document(
            file_path=document.file_path
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document parsing failed: {exc}",
        )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No text content found in document",
        )

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    try:
        embeddings = (
            embedding_model.embed_documents(
                texts
            )
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding failed: {exc}",
        )

    (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.document_id
            == document.id
        )
        .delete(
            synchronize_session=False
        )
    )

    for index, (
        chunk,
        embedding,
    ) in enumerate(
        zip(
            chunks,
            embeddings,
        )
    ):
        db_chunk = DocumentChunk(
            workspace_id=workspace_id,
            document_id=document.id,
            chunk_index=index,
            content=chunk.page_content,
            metadata_json=chunk.metadata,
            embedding=embedding,
        )

        db.add(db_chunk)

    db.commit()

    return len(chunks)


def get_document_chunks(
    db: Session,
    workspace_id: int,
    document_id: int,
) -> List[DocumentChunk]:
    document = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.workspace_id == workspace_id,
        )
        .first()
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.document_id
            == document_id
        )
        .order_by(
            DocumentChunk.chunk_index
        )
        .all()
    )