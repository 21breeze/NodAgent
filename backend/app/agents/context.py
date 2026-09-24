from typing_extensions import TypedDict


class AgentContext(TypedDict):
    user_id: str

    workspace_id: int

    top_k: int