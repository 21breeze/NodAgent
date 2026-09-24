from typing import (
    Dict,
    List,
    Optional,
)

from sqlalchemy.orm import Session

from backend.app.db.database import (
    SessionLocal,
)
from backend.app.models.agent_memory import (
    AgentMemory,
)


def upsert_memory(
    db: Session,
    workspace_id: int,
    memory_scope: str,
    memory_key: str,
    memory_value: str,
    user_id: Optional[str] = None,
) -> AgentMemory:
    query = (
        db.query(AgentMemory)
        .filter(
            AgentMemory.workspace_id
            == workspace_id,
            AgentMemory.memory_scope
            == memory_scope,
            AgentMemory.memory_key
            == memory_key,
        )
    )

    if memory_scope == "user":
        query = query.filter(
            AgentMemory.user_id
            == user_id
        )
    else:
        query = query.filter(
            AgentMemory.user_id.is_(None)
        )

    memory = query.first()

    if memory is None:
        memory = AgentMemory(
            workspace_id=workspace_id,
            memory_scope=memory_scope,
            user_id=user_id,
            memory_key=memory_key,
            memory_value=memory_value,
        )

        db.add(memory)

    else:
        memory.memory_value = (
            memory_value
        )

    db.commit()

    db.refresh(memory)

    return memory


def get_workspace_memories(
    db: Session,
    workspace_id: int,
) -> List[AgentMemory]:
    return (
        db.query(AgentMemory)
        .filter(
            AgentMemory.workspace_id
            == workspace_id,
            AgentMemory.memory_scope
            == "workspace",
        )
        .order_by(
            AgentMemory.memory_key
        )
        .all()
    )


def get_user_memories(
    db: Session,
    workspace_id: int,
    user_id: str,
) -> List[AgentMemory]:
    return (
        db.query(AgentMemory)
        .filter(
            AgentMemory.workspace_id
            == workspace_id,
            AgentMemory.memory_scope
            == "user",
            AgentMemory.user_id
            == user_id,
        )
        .order_by(
            AgentMemory.memory_key
        )
        .all()
    )


def delete_memory(
    db: Session,
    workspace_id: int,
    memory_id: int,
) -> bool:
    memory = (
        db.query(AgentMemory)
        .filter(
            AgentMemory.id
            == memory_id,
            AgentMemory.workspace_id
            == workspace_id,
        )
        .first()
    )

    if memory is None:
        return False

    db.delete(memory)

    db.commit()

    return True


def load_memory_context(
    workspace_id: int,
    user_id: str,
) -> Dict[str, Dict[str, str]]:
    with SessionLocal() as db:
        workspace_memories = (
            get_workspace_memories(
                db=db,
                workspace_id=workspace_id,
            )
        )

        user_memories = (
            get_user_memories(
                db=db,
                workspace_id=workspace_id,
                user_id=user_id,
            )
        )

        return {
            "workspace": {
                memory.memory_key:
                memory.memory_value
                for memory
                in workspace_memories
            },

            "user": {
                memory.memory_key:
                memory.memory_value
                for memory
                in user_memories
            },
        }


def build_memory_prompt(
    memory_context: Dict[
        str,
        Dict[str, str],
    ],
) -> str:
    workspace_memories = (
        memory_context.get(
            "workspace",
            {},
        )
    )

    user_memories = (
        memory_context.get(
            "user",
            {},
        )
    )

    lines: List[str] = []

    if workspace_memories:
        lines.append(
            "Workspace long-term memory:"
        )

        for key, value in (
            workspace_memories.items()
        ):
            lines.append(
                f"- {key}: {value}"
            )

    if user_memories:
        if lines:
            lines.append("")

        lines.append(
            "User long-term memory:"
        )

        for key, value in (
            user_memories.items()
        ):
            lines.append(
                f"- {key}: {value}"
            )

    return "\n".join(lines)
