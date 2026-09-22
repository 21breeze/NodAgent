from fastapi import APIRouter

from backend.app.schemas.workspace import WorkspaceCreate


router = APIRouter(
    prefix="/workspaces",
    tags=["workspaces"],
)


@router.get("")
def get_workspaces():
    return {
        "workspaces": []
    }


@router.post("")
def create_workspace(workspace: WorkspaceCreate):
    return {
        "message": "workspace created",
        "workspace": workspace
    }