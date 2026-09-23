from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from backend.app.db.database import Base


class ChatThread(Base):
    __tablename__ = "chat_threads"

    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "user_id",
            "thread_id",
            name="uq_chat_threads_workspace_user_thread",
        ),
    )

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    workspace_id = Column(
        Integer,
        ForeignKey("workspaces.id"),
        nullable=False,
        index=True,
    )

    user_id = Column(
        String(100),
        nullable=False,
        index=True,
    )

    thread_id = Column(
        String(100),
        nullable=False,
        index=True,
    )

    title = Column(
        String(200),
        nullable=False,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    workspace = relationship(
        "Workspace",
        back_populates="chat_threads",
    )

    messages = relationship(
        "ChatMessage",
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="ChatMessage.id",
    )