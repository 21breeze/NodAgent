from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)
from sqlalchemy.orm import Session

from backend.app.db.database import (
    get_db,
)
from backend.app.schemas.chat import (
    ChatRequest,
    ChatResponse,
)
from backend.app.services import (
    chat_history_service,
    rag_service,
)


router = APIRouter(
    prefix="/workspaces/{workspace_id}/chat",
    tags=["chat"],
)


@router.post(
    "",
    response_model=ChatResponse,
)
async def chat(
    workspace_id: int,
    data: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    workspace = (
        chat_history_service.get_workspace(
            db=db,
            workspace_id=workspace_id,
        )
    )

    if workspace is None:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="Workspace not found",
        )

    graph = request.app.state.rag_graph

    try:
        return await rag_service.chat(
            db=db,
            graph=graph,
            workspace_id=workspace_id,
            user_id=data.user_id,
            thread_id=data.thread_id,
            message=data.message,
            top_k=data.top_k,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Failed to generate "
                "chat response"
            ),
        ) from exc