from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)
from sqlalchemy.orm import Session

from backend.app.db.database import (
    get_db,
)
from backend.app.models.workspace import (
    Workspace,
)
from backend.app.schemas.memory import (
    MemoryContextResponse,
    MemoryResponse,
    MemoryUpsertRequest,
)
from backend.app.services import (
    memory_service,
)


router = APIRouter(
    prefix=(
        "/workspaces/"
        "{workspace_id}/memories"
    ),
    tags=["memories"],
)


def ensure_workspace_exists(
    db: Session,
    workspace_id: int,
) -> None:
    workspace = (
        db.query(Workspace)
        .filter(
            Workspace.id
            == workspace_id
        )
        .first()
    )

    if workspace is None:
        raise HTTPException(
            status_code=404,
            detail="Workspace not found",
        )


@router.put(
    "",
    response_model=MemoryResponse,
)
def upsert_memory(
    workspace_id: int,
    request: MemoryUpsertRequest,
    db: Session = Depends(get_db),
):
    ensure_workspace_exists(
        db=db,
        workspace_id=workspace_id,
    )

    return memory_service.upsert_memory(
        db=db,
        workspace_id=workspace_id,
        memory_scope=(
            request.memory_scope
        ),
        user_id=request.user_id,
        memory_key=request.memory_key,
        memory_value=(
            request.memory_value
        ),
    )


@router.get(
    "",
    response_model=MemoryContextResponse,
)
def get_memories(
    workspace_id: int,
    user_id: str = Query(
        min_length=1
    ),
    db: Session = Depends(get_db),
):
    ensure_workspace_exists(
        db=db,
        workspace_id=workspace_id,
    )

    workspace_memories = (
        memory_service
        .get_workspace_memories(
            db=db,
            workspace_id=workspace_id,
        )
    )

    user_memories = (
        memory_service
        .get_user_memories(
            db=db,
            workspace_id=workspace_id,
            user_id=user_id,
        )
    )

    return MemoryContextResponse(
        workspace_id=workspace_id,
        user_id=user_id,
        workspace_memories=(
            workspace_memories
        ),
        user_memories=user_memories,
    )


@router.delete(
    "/{memory_id}",
    status_code=(
        status.HTTP_204_NO_CONTENT
    ),
)
def delete_memory(
    workspace_id: int,
    memory_id: int,
    db: Session = Depends(get_db),
):
    deleted = (
        memory_service.delete_memory(
            db=db,
            workspace_id=workspace_id,
            memory_id=memory_id,
        )
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Memory not found",
        )

    return None
