from typing import Any, Dict

from backend.app.services import (
    memory_service,
)


def build_agent_context(
    user_id: str,
    workspace_id: int,
    top_k: int,
) -> Dict[str, Any]:
    memory_context = (
        memory_service
        .load_memory_context(
            workspace_id=workspace_id,
            user_id=user_id,
        )
    )

    memory_prompt = (
        memory_service
        .build_memory_prompt(
            memory_context
        )
    )

    return {
        "user_id": user_id,
        "workspace_id": workspace_id,
        "top_k": top_k,
        "memory_prompt": memory_prompt,
    }
