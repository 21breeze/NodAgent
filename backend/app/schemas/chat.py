from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

from pydantic import (
    BaseModel,
    Field,
)


class ChatSource(BaseModel):
    source_number: int

    citation: str

    chunk_id: int

    document_id: int

    filename: str

    chunk_index: int

    page_number: Optional[int] = None

    content: str

    metadata_json: Dict[
        str,
        Any,
    ]

    distance: Optional[float] = None

    similarity: Optional[float] = None

    keyword_score: Optional[float] = None

    vector_rank: Optional[int] = None

    keyword_rank: Optional[int] = None

    rrf_score: Optional[float] = None


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
