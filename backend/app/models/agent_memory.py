from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)

from backend.app.db.database import Base


class AgentMemory(Base):
    __tablename__ = "agent_memories"

    __table_args__ = (
        CheckConstraint(
            """
            (
                memory_scope = 'user'
                AND user_id IS NOT NULL
            )
            OR
            (
                memory_scope = 'workspace'
                AND user_id IS NULL
            )
            """,
            name="ck_agent_memories_scope_user",
        ),

        Index(
            "uq_agent_memories_user",
            "workspace_id",
            "user_id",
            "memory_key",
            unique=True,
            postgresql_where=text(
                "memory_scope = 'user'"
            ),
        ),

        Index(
            "uq_agent_memories_workspace",
            "workspace_id",
            "memory_key",
            unique=True,
            postgresql_where=text(
                "memory_scope = 'workspace'"
            ),
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

    memory_scope = Column(
        String(20),
        nullable=False,
        index=True,
    )

    user_id = Column(
        String(100),
        nullable=True,
        index=True,
    )

    memory_key = Column(
        String(100),
        nullable=False,
    )

    memory_value = Column(
        Text,
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
