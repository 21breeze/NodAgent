from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(
        min_length=1,
        max_length=100,
    )

    thread_id: str = Field(
        min_length=1,
        max_length=100,
    )

    message: str = Field(
        min_length=1,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )


class ChatSource(BaseModel):
    source_number: int

    chunk_id: int

    document_id: int

    chunk_index: int

    content: str

    metadata_json: Dict[str, Any]

    distance: float


class ChatResponse(BaseModel):
    thread_id: str

    answer: str

    sources: List[ChatSource]