from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.models.workspace import (
    Workspace,
)
from backend.app.schemas.chat import (
    ChatRequest,
    ChatResponse,
)
from backend.app.services.rag_service import (
    chat_with_workspace,
)


router = APIRouter(
    prefix=(
        "/workspaces/"
        "{workspace_id}/chat"
    ),
    tags=["chat"],
)


@router.post(
    "",
    response_model=ChatResponse,
)
async def chat(
    workspace_id: int,
    chat_request: ChatRequest,
    http_request: Request,
    db: Session = Depends(get_db),
):
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
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="Workspace not found",
        )

    graph = (
        http_request
        .app
        .state
        .rag_graph
    )

    try:
        return await chat_with_workspace(
            graph=graph,
            user_id=(
                chat_request.user_id
            ),
            thread_id=(
                chat_request.thread_id
            ),
            workspace_id=workspace_id,
            message=(
                chat_request.message
            ),
            top_k=(
                chat_request.top_k
            ),
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=str(exc),
        )