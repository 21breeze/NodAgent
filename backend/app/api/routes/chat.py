from collections.abc import (
    AsyncIterator,
)
from dataclasses import dataclass
from typing import Union

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

from backend.app.db.database import (
    SessionLocal,
)
from backend.app.schemas.chat import (
    ChatApiResponse,
    ChatRequest,
    ChatResumeRequest,
)
from backend.app.services import (
    agent_service,
    chat_history_service,
)
from backend.app.services.agent_service import (
    NoPendingInterruptError,
    PendingInterruptError,
)


router = APIRouter(
    prefix=(
        "/workspaces/"
        "{workspace_id}/chat"
    ),
    tags=[
        "chat",
    ],
)


@dataclass(
    frozen=True
)
class ChatContext:
    data: Union[
        ChatRequest,
        ChatResumeRequest,
    ]

    chat_thread_id: int


def validate_chat_thread(
    workspace_id: int,
    user_id: str,
    thread_id: str,
) -> int:
    with SessionLocal() as db:
        workspace = (
            chat_history_service
            .get_workspace(
                db=db,
                workspace_id=(
                    workspace_id
                ),
            )
        )

        if workspace is None:
            raise HTTPException(
                status_code=(
                    status
                    .HTTP_404_NOT_FOUND
                ),
                detail=(
                    "Workspace not found"
                ),
            )

        thread = (
            chat_history_service
            .get_thread(
                db=db,
                workspace_id=(
                    workspace_id
                ),
                user_id=user_id,
                thread_id=thread_id,
            )
        )

        if thread is None:
            raise HTTPException(
                status_code=(
                    status
                    .HTTP_404_NOT_FOUND
                ),
                detail=(
                    "Chat thread "
                    "not found"
                ),
            )

        return thread.id


def get_chat_context(
    workspace_id: int,
    data: ChatRequest,
) -> ChatContext:
    chat_thread_id = (
        validate_chat_thread(
            workspace_id=workspace_id,
            user_id=data.user_id,
            thread_id=data.thread_id,
        )
    )

    return ChatContext(
        data=data,
        chat_thread_id=(
            chat_thread_id
        ),
    )


def get_resume_context(
    workspace_id: int,
    data: ChatResumeRequest,
) -> ChatContext:
    chat_thread_id = (
        validate_chat_thread(
            workspace_id=workspace_id,
            user_id=data.user_id,
            thread_id=data.thread_id,
        )
    )

    return ChatContext(
        data=data,
        chat_thread_id=(
            chat_thread_id
        ),
    )


@router.post(
    "",
    response_model=ChatApiResponse,
)
async def chat(
    workspace_id: int,
    request: Request,
    chat_context: ChatContext = Depends(
        get_chat_context
    ),
):
    data = chat_context.data

    agent = (
        request.app.state.agent_graph
    )

    try:
        return await agent_service.chat(
            agent=agent,
            chat_thread_id=(
                chat_context
                .chat_thread_id
            ),
            workspace_id=(
                workspace_id
            ),
            user_id=data.user_id,
            thread_id=(
                data.thread_id
            ),
            message=data.message,
            top_k=data.top_k,
        )

    except PendingInterruptError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_409_CONFLICT
            ),
            detail=(
                "This thread is waiting "
                "for human approval. "
                "Use /chat/resume."
            ),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=(
                status
                .HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Failed to execute "
                "agent graph"
            ),
        ) from exc


@router.post(
    "/resume",
    response_model=ChatApiResponse,
)
async def resume_chat(
    workspace_id: int,
    request: Request,
    chat_context: ChatContext = Depends(
        get_resume_context
    ),
):
    data = chat_context.data

    agent = (
        request.app.state.agent_graph
    )

    try:
        return await (
            agent_service.resume_chat(
                agent=agent,
                chat_thread_id=(
                    chat_context
                    .chat_thread_id
                ),
                workspace_id=(
                    workspace_id
                ),
                user_id=(
                    data.user_id
                ),
                thread_id=(
                    data.thread_id
                ),
                decision=(
                    data.decision
                ),
                top_k=data.top_k,
            )
        )

    except (
        NoPendingInterruptError
    ) as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_409_CONFLICT
            ),
            detail=(
                "This thread has no "
                "pending approval."
            ),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=(
                status
                .HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Failed to resume "
                "agent graph"
            ),
        ) from exc


@router.post(
    "/stream",
    response_class=(
        EventSourceResponse
    ),
)
async def stream_chat(
    workspace_id: int,
    request: Request,
    chat_context: ChatContext = Depends(
        get_chat_context
    ),
) -> AsyncIterator[
    ServerSentEvent
]:
    data = chat_context.data

    agent = (
        request.app.state.agent_graph
    )

    async for stream_event in (
        agent_service.stream_chat(
            agent=agent,
            chat_thread_id=(
                chat_context
                .chat_thread_id
            ),
            workspace_id=(
                workspace_id
            ),
            user_id=data.user_id,
            thread_id=(
                data.thread_id
            ),
            message=data.message,
            top_k=data.top_k,
        )
    ):
        yield ServerSentEvent(
            event=(
                stream_event[
                    "event"
                ]
            ),
            data=(
                stream_event[
                    "data"
                ]
            ),
        )


@router.post(
    "/stream/resume",
    response_class=(
        EventSourceResponse
    ),
)
async def stream_resume_chat(
    workspace_id: int,
    request: Request,
    chat_context: ChatContext = Depends(
        get_resume_context
    ),
) -> AsyncIterator[
    ServerSentEvent
]:
    data = chat_context.data

    agent = (
        request.app.state.agent_graph
    )

    async for stream_event in (
        agent_service
        .stream_resume_chat(
            agent=agent,
            chat_thread_id=(
                chat_context
                .chat_thread_id
            ),
            workspace_id=(
                workspace_id
            ),
            user_id=data.user_id,
            thread_id=(
                data.thread_id
            ),
            decision=(
                data.decision
            ),
            top_k=data.top_k,
        )
    ):
        yield ServerSentEvent(
            event=(
                stream_event[
                    "event"
                ]
            ),
            data=(
                stream_event[
                    "data"
                ]
            ),
        )