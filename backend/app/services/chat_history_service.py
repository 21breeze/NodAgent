from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from sqlalchemy.orm import Session

from backend.app.models.chat_message import (
    ChatMessage,
)
from backend.app.models.chat_thread import (
    ChatThread,
)
from backend.app.models.workspace import (
    Workspace,
)
from backend.app.schemas.chat_history import (
    ChatThreadCreate,
)


DEFAULT_THREAD_TITLE = "新对话"


def get_workspace(
    db: Session,
    workspace_id: int,
) -> Optional[Workspace]:
    return (
        db.query(Workspace)
        .filter(
            Workspace.id == workspace_id
        )
        .first()
    )


def build_internal_thread_id(
    user_id: str,
    workspace_id: int,
    thread_id: str,
) -> str:
    return (
        f"user:{user_id}:"
        f"workspace:{workspace_id}:"
        f"thread:{thread_id}"
    )


def generate_thread_title(
    message: str,
    max_length: int = 30,
) -> str:
    normalized_message = " ".join(
        message.strip().split()
    )

    if not normalized_message:
        return DEFAULT_THREAD_TITLE

    if len(normalized_message) <= max_length:
        return normalized_message

    return (
        normalized_message[:max_length]
        + "..."
    )


def create_thread(
    db: Session,
    workspace_id: int,
    data: ChatThreadCreate,
) -> ChatThread:
    thread = ChatThread(
        workspace_id=workspace_id,
        user_id=data.user_id,
        thread_id=data.thread_id,
        title=(
            data.title
            or DEFAULT_THREAD_TITLE
        ),
    )

    db.add(thread)
    db.commit()
    db.refresh(thread)

    return thread


def get_thread(
    db: Session,
    workspace_id: int,
    user_id: str,
    thread_id: str,
) -> Optional[ChatThread]:
    return (
        db.query(ChatThread)
        .filter(
            ChatThread.workspace_id
            == workspace_id,
            ChatThread.user_id
            == user_id,
            ChatThread.thread_id
            == thread_id,
        )
        .first()
    )


def list_threads(
    db: Session,
    workspace_id: int,
    user_id: str,
) -> List[ChatThread]:
    return (
        db.query(ChatThread)
        .filter(
            ChatThread.workspace_id
            == workspace_id,
            ChatThread.user_id
            == user_id,
        )
        .order_by(
            ChatThread.updated_at.desc()
        )
        .all()
    )


def list_messages(
    db: Session,
    chat_thread_id: int,
) -> List[ChatMessage]:
    return (
        db.query(ChatMessage)
        .filter(
            ChatMessage.chat_thread_id
            == chat_thread_id
        )
        .order_by(
            ChatMessage.id.asc()
        )
        .all()
    )


def create_message(
    db: Session,
    chat_thread_id: int,
    role: str,
    content: str,
    sources: Optional[
        List[Dict[str, Any]]
    ] = None,
) -> ChatMessage:
    message = ChatMessage(
        chat_thread_id=chat_thread_id,
        role=role,
        content=content,
        sources=sources or [],
    )

    db.add(message)

    return message


def touch_thread(
    thread: ChatThread,
) -> None:
    thread.updated_at = datetime.utcnow()


def update_thread_title_from_message(
    thread: ChatThread,
    message: str,
) -> None:
    if thread.title != DEFAULT_THREAD_TITLE:
        return

    thread.title = generate_thread_title(
        message=message
    )


def delete_thread(
    db: Session,
    thread: ChatThread,
) -> None:
    db.delete(thread)
    db.commit()