from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="用户的检索问题",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="返回最相关的 Chunk 数量",
    )


class SearchResult(BaseModel):
    chunk_id: int
    document_id: int
    chunk_index: int
    content: str
    metadata_json: Dict[str, Any]

    distance: Optional[float] = None
    similarity: Optional[float] = None

    keyword_score: Optional[float] = None

    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None

    rrf_score: float


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[SearchResult]
