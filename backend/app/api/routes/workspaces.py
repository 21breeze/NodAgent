from typing import List

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.schemas.workspace import (
    WorkspaceCreate,
    WorkspaceResponse,
)
from backend.app.services import workspace_service


router = APIRouter(
    prefix="/workspaces",
    tags=["workspaces"],
)


@router.get(
    "",
    response_model=List[WorkspaceResponse],
)
def get_workspaces(
    db: Session = Depends(get_db),
):
    return workspace_service.get_workspaces(db)


@router.post(
    "",
    response_model=WorkspaceResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_workspace(
    workspace: WorkspaceCreate,
    db: Session = Depends(get_db),
):
    return workspace_service.create_workspace(
        db=db,
        workspace=workspace,
    )