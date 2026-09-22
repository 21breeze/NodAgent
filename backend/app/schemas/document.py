from typing import Optional

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