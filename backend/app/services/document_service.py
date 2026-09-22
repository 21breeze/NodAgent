from pathlib import Path
from typing import List

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.app.models.document import Document
from backend.app.models.workspace import Workspace


UPLOAD_DIR = Path("data/uploads")


def get_documents(
    db: Session,
    workspace_id: int,
) -> List[Document]:
    workspace = (
        db.query(Workspace)
        .filter(Workspace.id == workspace_id)
        .first()
    )

    if workspace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found",
        )

    return (
        db.query(Document)
        .filter(Document.workspace_id == workspace_id)
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
        .filter(Workspace.id == workspace_id)
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

    file_path = UPLOAD_DIR / file.filename

    with file_path.open("wb") as buffer:
        while True:
            chunk = file.file.read(1024 * 1024)

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