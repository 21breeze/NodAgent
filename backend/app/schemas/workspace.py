from typing import Optional

from pydantic import BaseModel, ConfigDict


class WorkspaceCreate(BaseModel):
    name: str
    description: Optional[str] = None


class WorkspaceResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None

    model_config = ConfigDict(
        from_attributes=True
    )