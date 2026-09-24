import asyncio
from typing import (
    Any,
    Dict,
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
    citation_service,
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

    sources = await asyncio.to_thread(
        citation_service
        .build_citation_sources,
        results,
    )

    tool_content = (
        citation_service
        .build_citation_context(
            sources
        )
    )

    artifact = {
        "sources": sources,
    }

    return (
        tool_content,
        artifact,
    )
