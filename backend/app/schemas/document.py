from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    id: int
    workspace_id: int
    filename: str
    file_path: str
    content_type: Optional[str] = None

    model_config = ConfigDict(
        from_attributes=True
    )


class DocumentProcessResponse(BaseModel):
    document_id: int
    chunk_count: int


class DocumentChunkResponse(BaseModel):
    id: int
    document_id: int
    chunk_index: int
    content: str
    metadata_json: Dict[str, Any]

    model_config = ConfigDict(
        from_attributes=True
    )