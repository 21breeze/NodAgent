from typing import List

from fastapi import (
    APIRouter,
    Depends,
    File,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.schemas.document import (
    DocumentChunkResponse,
    DocumentProcessResponse,
    DocumentResponse,
)
from backend.app.services import document_service


router = APIRouter(
    prefix="/workspaces/{workspace_id}/documents",
    tags=["documents"],
)


@router.get(
    "",
    response_model=List[DocumentResponse],
)
def get_documents(
    workspace_id: int,
    db: Session = Depends(get_db),
):
    return document_service.get_documents(
        db=db,
        workspace_id=workspace_id,
    )


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    workspace_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    return document_service.create_document(
        db=db,
        workspace_id=workspace_id,
        file=file,
    )


@router.post(
    "/{document_id}/process",
    response_model=DocumentProcessResponse,
)
def process_document(
    workspace_id: int,
    document_id: int,
    db: Session = Depends(get_db),
):
    chunk_count = (
        document_service.process_document(
            db=db,
            workspace_id=workspace_id,
            document_id=document_id,
        )
    )

    return DocumentProcessResponse(
        document_id=document_id,
        chunk_count=chunk_count,
    )


@router.get(
    "/{document_id}/chunks",
    response_model=List[DocumentChunkResponse],
)
def get_document_chunks(
    workspace_id: int,
    document_id: int,
    db: Session = Depends(get_db),
):
    return (
        document_service.get_document_chunks(
            db=db,
            workspace_id=workspace_id,
            document_id=document_id,
        )
    )