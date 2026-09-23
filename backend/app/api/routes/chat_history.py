from typing import List

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from sqlalchemy.orm import Session

from backend.app.db.database import (
    get_db,
)
from backend.app.schemas.chat_history import (
    ChatMessageResponse,
    ChatThreadCreate,
    ChatThreadResponse,
)
from backend.app.services import (
    chat_history_service,
)


router = APIRouter(
    prefix="/workspaces/{workspace_id}/threads",
    tags=["chat-history"],
)


@router.post(
    "",
    response_model=ChatThreadResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_thread(
    workspace_id: int,
    data: ChatThreadCreate,
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found",
        )

    existing_thread = (
        chat_history_service.get_thread(
            db=db,
            workspace_id=workspace_id,
            user_id=data.user_id,
            thread_id=data.thread_id,
        )
    )

    if existing_thread is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Thread already exists",
        )

    return chat_history_service.create_thread(
        db=db,
        workspace_id=workspace_id,
        data=data,
    )


@router.get(
    "",
    response_model=List[ChatThreadResponse],
)
def list_threads(
    workspace_id: int,
    user_id: str = Query(
        min_length=1,
        max_length=100,
    ),
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found",
        )

    return chat_history_service.list_threads(
        db=db,
        workspace_id=workspace_id,
        user_id=user_id,
    )


@router.get(
    "/{thread_id}/messages",
    response_model=List[ChatMessageResponse],
)
def list_messages(
    workspace_id: int,
    thread_id: str,
    user_id: str = Query(
        min_length=1,
        max_length=100,
    ),
    db: Session = Depends(get_db),
):
    thread = (
        chat_history_service.get_thread(
            db=db,
            workspace_id=workspace_id,
            user_id=user_id,
            thread_id=thread_id,
        )
    )

    if thread is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found",
        )

    return chat_history_service.list_messages(
        db=db,
        chat_thread_id=thread.id,
    )


@router.delete(
    "/{thread_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_thread(
    workspace_id: int,
    thread_id: str,
    request: Request,
    user_id: str = Query(
        min_length=1,
        max_length=100,
    ),
    db: Session = Depends(get_db),
):
    thread = (
        chat_history_service.get_thread(
            db=db,
            workspace_id=workspace_id,
            user_id=user_id,
            thread_id=thread_id,
        )
    )

    if thread is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found",
        )

    internal_thread_id = (
        chat_history_service
        .build_internal_thread_id(
            user_id=user_id,
            workspace_id=workspace_id,
            thread_id=thread_id,
        )
    )

    checkpointer = (
        request.app.state.checkpointer
    )

    try:
        await checkpointer.adelete_thread(
            internal_thread_id
        )

        chat_history_service.delete_thread(
            db=db,
            thread=thread,
        )

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="Failed to delete thread",
        ) from exc

    return Response(
        status_code=(
            status.HTTP_204_NO_CONTENT
        )
    )