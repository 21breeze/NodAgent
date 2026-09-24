from typing import (
    Any,
    Dict,
    Optional,
)

from pydantic import (
    BaseModel,
    ConfigDict,
)


class DocumentResponse(
    BaseModel
):
    model_config = ConfigDict(
        from_attributes=True
    )

    id: int

    workspace_id: int

    filename: str

    file_path: str

    content_type: Optional[
        str
    ]

    processing_status: str

    processing_task_id: Optional[
        str
    ]

    processing_error: Optional[
        str
    ]


class DocumentProcessingResponse(
    BaseModel
):
    document_id: int

    status: str

    task_id: Optional[
        str
    ]

    error: Optional[
        str
    ]


class DocumentChunkResponse(
    BaseModel
):
    model_config = ConfigDict(
        from_attributes=True
    )

    id: int

    workspace_id: int

    document_id: int

    chunk_index: int

    content: str

    metadata_json: Dict[
        str,
        Any,
    ]