from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Union,
)

from pydantic import (
    BaseModel,
    Field,
)


class ChatSource(BaseModel):
    source_number: int

    chunk_id: int

    document_id: int

    chunk_index: int

    content: str

    metadata_json: Dict[
        str,
        Any,
    ]

    distance: float


class ChatRequest(BaseModel):
    user_id: str

    thread_id: str

    message: str

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )


class ChatResumeRequest(BaseModel):
    user_id: str

    thread_id: str

    decision: Literal[
        "approve",
        "reject",
    ]

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )


class ChatResponse(BaseModel):
    status: Literal[
        "completed"
    ] = "completed"

    thread_id: str

    answer: str

    sources: List[
        ChatSource
    ] = Field(
        default_factory=list
    )


class ChatInterruptResponse(BaseModel):
    status: Literal[
        "interrupted"
    ] = "interrupted"

    thread_id: str

    interrupt_id: str

    interrupt: Dict[
        str,
        Any,
    ]


ChatApiResponse = Annotated[
    Union[
        ChatResponse,
        ChatInterruptResponse,
    ],
    Field(
        discriminator="status"
    ),
]