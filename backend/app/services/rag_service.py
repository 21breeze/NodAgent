from typing import Any

from langchain_core.messages import (
    HumanMessage,
)

from backend.app.schemas.chat import (
    ChatResponse,
    ChatSource,
)


def message_to_text(
    message,
) -> str:
    content = message.content

    if isinstance(
        content,
        str,
    ):
        return content

    text_parts = []

    for block in content:
        if isinstance(
            block,
            str,
        ):
            text_parts.append(
                block
            )

        elif isinstance(
            block,
            dict,
        ):
            text = block.get(
                "text"
            )

            if text:
                text_parts.append(
                    str(text)
                )

    return "\n".join(
        text_parts
    )


async def chat_with_workspace(
    graph: Any,
    user_id: str,
    thread_id: str,
    workspace_id: int,
    message: str,
    top_k: int,
) -> ChatResponse:
    internal_thread_id = (
        f"user:{user_id}:"
        f"workspace:{workspace_id}:"
        f"thread:{thread_id}"
    )

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=message
                )
            ],
            "top_k": top_k,
        },
        config={
            "configurable": {
                "thread_id":
                    internal_thread_id,
            }
        },
        context={
            "user_id": user_id,
            "workspace_id":
                workspace_id,
        },
    )

    answer_message = (
        result["messages"][-1]
    )

    answer = message_to_text(
        answer_message
    )

    sources = []

    for index, chunk in enumerate(
        result.get(
            "retrieved_chunks",
            [],
        ),
        start=1,
    ):
        sources.append(
            ChatSource(
                source_number=index,
                chunk_id=(
                    chunk["chunk_id"]
                ),
                document_id=(
                    chunk["document_id"]
                ),
                chunk_index=(
                    chunk["chunk_index"]
                ),
                content=(
                    chunk["content"]
                ),
                metadata_json=(
                    chunk["metadata_json"]
                ),
                distance=(
                    chunk["distance"]
                ),
            )
        )

    return ChatResponse(
        thread_id=thread_id,
        answer=answer,
        sources=sources,
    )