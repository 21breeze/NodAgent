from typing import (
    Any,
    Dict,
    List,
    Tuple,
)

from langchain.tools import (
    ToolRuntime,
)
from langchain_core.tools import (
    tool,
)

from backend.app.agents.context import (
    AgentContext,
)
from backend.app.services import (
    retrieval_service,
)


@tool(
    "search_knowledge_base",
    response_format=(
        "content_and_artifact"
    ),
)
async def search_knowledge_base(
    query: str,
    runtime: ToolRuntime[
        AgentContext
    ],
) -> Tuple[
    str,
    Dict[str, Any],
]:
    """
    Search the current workspace knowledge base.

    Use this tool when the user's question
    depends on uploaded documents or knowledge
    stored in the current workspace.

    The query should contain enough semantic
    context to retrieve the relevant document
    chunks.
    """

    workspace_id = (
        runtime.context[
            "workspace_id"
        ]
    )

    top_k = (
        runtime.context[
            "top_k"
        ]
    )

    results = (
        await retrieval_service
        .search_workspace_async(
            workspace_id=(
                workspace_id
            ),
            query=query,
            top_k=top_k,
        )
    )

    if not results:
        return (
            (
                "No relevant completed "
                "documents were found in "
                "the current workspace."
            ),
            {
                "sources": [],
            },
        )

    context_parts: List[
        str
    ] = []

    sources: List[
        Dict[str, Any]
    ] = []

    for index, result in enumerate(
        results,
        start=1,
    ):
        metadata = dict(
            result.metadata_json
        )

        page = metadata.get(
            "page"
        )

        source_info = [
            (
                f"document_id="
                f"{result.document_id}"
            ),
            (
                f"chunk_index="
                f"{result.chunk_index}"
            ),
        ]

        if page is not None:
            source_info.append(
                f"page={page}"
            )

        context_parts.append(
            (
                f"[{index}]\n"
                f"{', '.join(source_info)}\n"
                f"{result.content}"
            )
        )

        sources.append(
            {
                "source_number": (
                    index
                ),
                "chunk_id": (
                    result.chunk_id
                ),
                "document_id": (
                    result.document_id
                ),
                "chunk_index": (
                    result.chunk_index
                ),
                "content": (
                    result.content
                ),
                "metadata_json": (
                    result.metadata_json
                ),
                "distance": (
                    result.distance
                ),
                "similarity": (
                    result.similarity
                ),
                "keyword_score": (
                    result.keyword_score
                ),
                "vector_rank": (
                    result.vector_rank
                ),
                "keyword_rank": (
                    result.keyword_rank
                ),
                "rrf_score": (
                    result.rrf_score
                ),
            }
        )

    tool_content = "\n\n".join(
        context_parts
    )

    artifact = {
        "sources": sources,
    }

    return (
        tool_content,
        artifact,
    )