from datetime import datetime
from typing import (
    List,
    Literal,
    Optional,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


MemoryScope = Literal[
    "user",
    "workspace",
]


class MemoryUpsertRequest(BaseModel):
    memory_scope: MemoryScope

    user_id: Optional[str] = None

    memory_key: str = Field(
        min_length=1,
        max_length=100,
    )

    memory_value: str = Field(
        min_length=1,
    )

    @model_validator(
        mode="after"
    )
    def validate_scope(
        self,
    ):
        if (
            self.memory_scope == "user"
            and not self.user_id
        ):
            raise ValueError(
                "user scope requires user_id"
            )

        if (
            self.memory_scope
            == "workspace"
            and self.user_id is not None
        ):
            raise ValueError(
                "workspace scope must not "
                "contain user_id"
            )

        return self


class MemoryResponse(BaseModel):
    id: int

    workspace_id: int

    memory_scope: MemoryScope

    user_id: Optional[str]

    memory_key: str

    memory_value: str

    created_at: datetime

    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )


class MemoryContextResponse(BaseModel):
    workspace_id: int

    user_id: str

    workspace_memories: List[
        MemoryResponse
    ]

    user_memories: List[
        MemoryResponse
    ]
