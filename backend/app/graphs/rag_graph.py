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
from typing_extensions import TypedDict

from backend.app.prompts.rag_prompt import (
    RAG_PROMPT,
)
from backend.app.services.llm_service import (
    get_chat_model,
)
from backend.app.services.vector_store_service import (
    vector_store,
)


class RetrievedChunk(TypedDict):
    chunk_id: int
    document_id: int
    chunk_index: int
    content: str
    metadata_json: Dict[str, Any]
    distance: float


class RAGState(TypedDict):
    messages: Annotated[
        List[AnyMessage],
        add_messages,
    ]

    top_k: int

    retrieved_chunks: List[
        RetrievedChunk
    ]


class RAGContext(TypedDict):
    user_id: str
    workspace_id: int


def get_latest_question(
    state: RAGState,
) -> str:
    for message in reversed(
        state["messages"]
    ):
        if isinstance(
            message,
            HumanMessage,
        ):
            if isinstance(
                message.content,
                str,
            ):
                return message.content

            return str(
                message.content
            )

    raise RuntimeError(
        "No human message found"
    )


async def retrieve_node(
    state: RAGState,
    runtime: Runtime[RAGContext],
):
    question = get_latest_question(
        state
    )

    workspace_id = (
        runtime.context["workspace_id"]
    )

    documents_with_scores = (
        await vector_store
        .asimilarity_search_with_score(
            query=question,
            k=state["top_k"],
            filter={
                "workspace_id": workspace_id,
            },
        )
    )

    retrieved_chunks = []

    for document, distance in (
        documents_with_scores
    ):
        metadata = dict(
            document.metadata
        )

        document_id = int(
            metadata.pop(
                "document_id"
            )
        )

        chunk_index = int(
            metadata.pop(
                "chunk_index"
            )
        )

        metadata.pop(
            "workspace_id",
            None,
        )

        chunk_id = 0

        if document.id is not None:
            chunk_id = int(
                document.id
            )

        retrieved_chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=chunk_index,
                content=(
                    document.page_content
                ),
                metadata_json=metadata,
                distance=float(
                    distance
                ),
            )
        )

    return {
        "retrieved_chunks":
            retrieved_chunks
    }


def route_after_retrieve(
    state: RAGState,
) -> Literal[
    "generate",
    "no_context",
]:
    if state["retrieved_chunks"]:
        return "generate"

    return "no_context"


def format_context(
    chunks: List[RetrievedChunk],
) -> str:
    sections = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        metadata = (
            chunk["metadata_json"]
        )

        page = metadata.get(
            "page"
        )

        source_info = (
            f"[{index}] "
            f"document_id="
            f"{chunk['document_id']}, "
            f"chunk_index="
            f"{chunk['chunk_index']}"
        )

        if page is not None:
            source_info += (
                f", page={page}"
            )

        section = (
            f"{source_info}\n"
            f"{chunk['content']}"
        )

        sections.append(
            section
        )

    return "\n\n".join(
        sections
    )


async def generate_node(
    state: RAGState,
):
    question = get_latest_question(
        state
    )

    context = format_context(
        state["retrieved_chunks"]
    )

    history = (
        state["messages"][:-1]
    )

    prompt_value = (
        RAG_PROMPT.invoke(
            {
                "question": question,
                "context": context,
                "history": history,
            }
        )
    )

    model = get_chat_model()

    response = await model.ainvoke(
        prompt_value
    )

    return {
        "messages": [
            response
        ]
    }


def no_context_node(
    state: RAGState,
):
    return {
        "messages": [
            AIMessage(
                content=(
                    "当前知识库中没有检索到"
                    "可以用于回答这个问题的内容。"
                )
            )
        ]
    }


def build_rag_graph(
    checkpointer,
):
    builder = StateGraph(
        RAGState,
        context_schema=RAGContext,
    )

    builder.add_node(
        "retrieve",
        retrieve_node,
    )

    builder.add_node(
        "generate",
        generate_node,
    )

    builder.add_node(
        "no_context",
        no_context_node,
    )

    builder.add_edge(
        START,
        "retrieve",
    )

    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "generate": "generate",
            "no_context": "no_context",
        },
    )

    builder.add_edge(
        "generate",
        END,
    )

    builder.add_edge(
        "no_context",
        END,
    )

    return builder.compile(
        checkpointer=checkpointer
    )