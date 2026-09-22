from typing import List

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.schemas.document import DocumentResponse
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