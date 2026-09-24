from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)
from fastapi.sse import (
    EventSourceResponse,
    ServerSentEvent,
)
from sqlalchemy.orm import Session

from backend.app.db.database import (
    SessionLocal,
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


@dataclass(frozen=True)
class StreamChatContext:
    data: ChatRequest
    chat_thread_id: int


def get_stream_chat_context(
    workspace_id: int,
    data: ChatRequest,
) -> StreamChatContext:
    with SessionLocal() as db:
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

        thread = (
            chat_history_service.get_thread(
                db=db,
                workspace_id=workspace_id,
                user_id=data.user_id,
                thread_id=data.thread_id,
            )
        )

        if thread is None:
            raise HTTPException(
                status_code=(
                    status.HTTP_404_NOT_FOUND
                ),
                detail="Chat thread not found",
            )

        return StreamChatContext(
            data=data,
            chat_thread_id=thread.id,
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


@router.post(
    "/stream",
    response_class=EventSourceResponse,
)
async def stream_chat(
    workspace_id: int,
    request: Request,
    stream_context: StreamChatContext = Depends(
        get_stream_chat_context
    ),
) -> AsyncIterator[ServerSentEvent]:
    data = stream_context.data

    graph = request.app.state.rag_graph

    async for stream_event in (
        rag_service.stream_chat(
            graph=graph,
            chat_thread_id=(
                stream_context.chat_thread_id
            ),
            workspace_id=workspace_id,
            user_id=data.user_id,
            thread_id=data.thread_id,
            message=data.message,
            top_k=data.top_k,
        )
    ):
        yield ServerSentEvent(
            event=stream_event["event"],
            data=stream_event["data"],
        )