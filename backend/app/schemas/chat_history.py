from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class ChatThreadCreate(BaseModel):
    user_id: str = Field(
        min_length=1,
        max_length=100,
    )

    thread_id: str = Field(
        min_length=1,
        max_length=100,
    )

    title: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=200,
    )


class ChatThreadResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True
    )

    id: int
    workspace_id: int
    user_id: str
    thread_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True
    )

    id: int
    chat_thread_id: int
    role: str
    content: str
    sources: List[Dict[str, Any]]
    created_at: datetime