from typing import Literal

from langchain.tools import (
    ToolRuntime,
)
from langchain_core.tools import (
    tool,
)

from backend.app.agents.context import (
    AgentContext,
)
from backend.app.db.database import (
    SessionLocal,
)
from backend.app.services import (
    memory_service,
)


MemoryScope = Literal[
    "user",
    "workspace",
]


@tool("save_memory")
def save_memory(
    memory_scope: MemoryScope,
    memory_key: str,
    memory_value: str,
    runtime: ToolRuntime[
        AgentContext
    ],
) -> str:
    """
    Save or update one long-term memory.

    Use user scope for preferences or facts
    that belong to the current user.

    Use workspace scope for stable project,
    business, or workspace-level information.

    Saving an existing key updates its value
    instead of creating a duplicate.
    """

    workspace_id = (
        runtime.context[
            "workspace_id"
        ]
    )

    current_user_id = (
        runtime.context[
            "user_id"
        ]
    )

    if memory_scope == "user":
        user_id = current_user_id
    else:
        user_id = None

    normalized_key = (
        memory_key.strip()
    )

    normalized_value = (
        memory_value.strip()
    )

    if not normalized_key:
        return (
            "Memory key cannot be empty."
        )

    if not normalized_value:
        return (
            "Memory value cannot be empty."
        )

    with SessionLocal() as db:
        memory = (
            memory_service.upsert_memory(
                db=db,
                workspace_id=(
                    workspace_id
                ),
                memory_scope=(
                    memory_scope
                ),
                memory_key=(
                    normalized_key
                ),
                memory_value=(
                    normalized_value
                ),
                user_id=user_id,
            )
        )

    return (
        "Memory saved successfully. "
        f"scope={memory.memory_scope}, "
        f"key={memory.memory_key}, "
        f"value={memory.memory_value}"
    )


@tool("list_memories")
def list_memories(
    runtime: ToolRuntime[
        AgentContext
    ],
) -> str:
    """
    List the long-term memories available to
    the current user and workspace.
    """

    workspace_id = (
        runtime.context[
            "workspace_id"
        ]
    )

    user_id = (
        runtime.context[
            "user_id"
        ]
    )

    with SessionLocal() as db:
        workspace_memories = (
            memory_service
            .get_workspace_memories(
                db=db,
                workspace_id=(
                    workspace_id
                ),
            )
        )

        user_memories = (
            memory_service
            .get_user_memories(
                db=db,
                workspace_id=(
                    workspace_id
                ),
                user_id=user_id,
            )
        )

    lines = []

    if workspace_memories:
        lines.append(
            "Workspace memories:"
        )

        for memory in (
            workspace_memories
        ):
            lines.append(
                "- "
                f"{memory.memory_key}: "
                f"{memory.memory_value}"
            )

    if user_memories:
        if lines:
            lines.append("")

        lines.append(
            "User memories:"
        )

        for memory in user_memories:
            lines.append(
                "- "
                f"{memory.memory_key}: "
                f"{memory.memory_value}"
            )

    if not lines:
        return (
            "No long-term memories found."
        )

    return "\n".join(lines)


@tool("delete_memory")
def delete_memory(
    memory_scope: MemoryScope,
    memory_key: str,
    runtime: ToolRuntime[
        AgentContext
    ],
) -> str:
    """
    Delete one long-term memory by its key.

    Use user scope for the current user's
    memory.

    Use workspace scope for workspace-level
    memory.
    """

    workspace_id = (
        runtime.context[
            "workspace_id"
        ]
    )

    current_user_id = (
        runtime.context[
            "user_id"
        ]
    )

    normalized_key = (
        memory_key.strip()
    )

    if not normalized_key:
        return (
            "Memory key cannot be empty."
        )

    if memory_scope == "user":
        user_id = current_user_id
    else:
        user_id = None

    with SessionLocal() as db:
        deleted = (
            memory_service
            .delete_memory_by_key(
                db=db,
                workspace_id=(
                    workspace_id
                ),
                memory_scope=(
                    memory_scope
                ),
                memory_key=(
                    normalized_key
                ),
                user_id=user_id,
            )
        )

    if not deleted:
        return (
            "Memory not found. "
            f"scope={memory_scope}, "
            f"key={normalized_key}"
        )

    return (
        "Memory deleted successfully. "
        f"scope={memory_scope}, "
        f"key={normalized_key}"
    )
