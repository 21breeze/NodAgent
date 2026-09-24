from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
)

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
)
from langgraph.graph import (
    END,
    START,
    StateGraph,
    add_messages,
)
from langgraph.runtime import Runtime
from typing_extensions import (
    TypedDict,
)

from backend.app.prompts.rag_prompt import (
    RAG_PROMPT,
)
from backend.app.services import (
    retrieval_service,
)
from backend.app.services.llm_service import (
    get_chat_model,
)


class RetrievedChunk(
    TypedDict
):
    chunk_id: int

    document_id: int

    chunk_index: int

    content: str

    metadata_json: Dict[
        str,
        Any,
    ]

    distance: float


class RAGState(
    TypedDict
):
    messages: Annotated[
        List[AnyMessage],
        add_messages,
    ]

    top_k: int

    retrieved_chunks: List[
        RetrievedChunk
    ]


class RAGContext(
    TypedDict
):
    user_id: str

    workspace_id: int


def get_latest_question(
    state: RAGState,
) -> str:
    messages = state.get(
        "messages",
        [],
    )

    for message in reversed(
        messages
    ):
        if isinstance(
            message,
            HumanMessage,
        ):
            return str(
                message.content
            )

    return ""


async def retrieve_node(
    state: RAGState,
    runtime: Runtime[
        RAGContext
    ],
):
    question = (
        get_latest_question(
            state
        )
    )

    if not question:
        return {
            "retrieved_chunks": []
        }

    workspace_id = (
        runtime.context[
            "workspace_id"
        ]
    )

    top_k = state.get(
        "top_k",
        5,
    )

    results = (
        await retrieval_service
        .search_workspace_async(
            workspace_id=(
                workspace_id
            ),
            query=question,
            top_k=top_k,
        )
    )

    retrieved_chunks: List[
        RetrievedChunk
    ] = []

    for result in results:
        retrieved_chunks.append(
            RetrievedChunk(
                chunk_id=(
                    result.chunk_id
                ),
                document_id=(
                    result.document_id
                ),
                chunk_index=(
                    result.chunk_index
                ),
                content=(
                    result.content
                ),
                metadata_json=(
                    result.metadata_json
                ),
                distance=(
                    result.distance
                ),
            )
        )

    return {
        "retrieved_chunks": (
            retrieved_chunks
        )
    }


def route_after_retrieve(
    state: RAGState,
) -> Literal[
    "generate",
    "no_context",
]:
    retrieved_chunks = (
        state.get(
            "retrieved_chunks",
            [],
        )
    )

    if retrieved_chunks:
        return "generate"

    return "no_context"


def format_context(
    retrieved_chunks: List[
        RetrievedChunk
    ],
) -> str:
    context_parts: List[
        str
    ] = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        metadata = (
            chunk[
                "metadata_json"
            ]
        )

        page = metadata.get(
            "page"
        )

        source_info = [
            (
                f"document_id="
                f"{chunk['document_id']}"
            ),
            (
                f"chunk_index="
                f"{chunk['chunk_index']}"
            ),
        ]

        if page is not None:
            source_info.append(
                f"page={page}"
            )

        source_text = ", ".join(
            source_info
        )

        context_parts.append(
            (
                f"[{index}]\n"
                f"{source_text}\n"
                f"{chunk['content']}"
            )
        )

    return "\n\n".join(
        context_parts
    )


async def generate_node(
    state: RAGState,
):
    question = (
        get_latest_question(
            state
        )
    )

    retrieved_chunks = (
        state.get(
            "retrieved_chunks",
            [],
        )
    )

    context = format_context(
        retrieved_chunks
    )

    messages = state.get(
        "messages",
        [],
    )

    history = (
        messages[:-1]
        if messages
        else []
    )

    prompt_value = (
        RAG_PROMPT.invoke(
            {
                "context": context,
                "history": history,
                "question": question,
            }
        )
    )

    model = get_chat_model()

    response = (
        await model.ainvoke(
            prompt_value
        )
    )

    return {
        "messages": [
            response
        ]
    }


async def no_context_node(
    state: RAGState,
):
    return {
        "messages": [
            AIMessage(
                content=(
                    "当前知识库中没有检索到"
                    "足够相关的已完成文档内容，"
                    "暂时无法基于知识库回答"
                    "这个问题。"
                )
            )
        ]
    }


def build_rag_graph(
    checkpointer,
):
    graph = StateGraph(
        RAGState,
        context_schema=RAGContext,
    )

    graph.add_node(
        "retrieve",
        retrieve_node,
    )

    graph.add_node(
        "generate",
        generate_node,
    )

    graph.add_node(
        "no_context",
        no_context_node,
    )

    graph.add_edge(
        START,
        "retrieve",
    )

    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "generate": (
                "generate"
            ),
            "no_context": (
                "no_context"
            ),
        },
    )

    graph.add_edge(
        "generate",
        END,
    )

    graph.add_edge(
        "no_context",
        END,
    )

    return graph.compile(
        checkpointer=checkpointer
    )