from typing import List

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langgraph.graph.state import (
    CompiledStateGraph,
)
from sqlalchemy.orm import Session

from backend.app.schemas.chat import (
    ChatResponse,
    ChatSource,
)
from backend.app.services import (
    chat_history_service,
)


async def chat(
    db: Session,
    graph: CompiledStateGraph,
    workspace_id: int,
    user_id: str,
    thread_id: str,
    message: str,
    top_k: int,
) -> ChatResponse:
    thread = (
        chat_history_service.get_thread(
            db=db,
            workspace_id=workspace_id,
            user_id=user_id,
            thread_id=thread_id,
        )
    )

    if thread is None:
        raise ValueError(
            "Chat thread not found"
        )

    chat_history_service.create_message(
        db=db,
        chat_thread_id=thread.id,
        role="user",
        content=message,
    )

    db.flush()

    internal_thread_id = (
        chat_history_service
        .build_internal_thread_id(
            user_id=user_id,
            workspace_id=workspace_id,
            thread_id=thread_id,
        )
    )

    try:
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
                    "thread_id": (
                        internal_thread_id
                    ),
                }
            },
            context={
                "user_id": user_id,
                "workspace_id": (
                    workspace_id
                ),
            },
        )

        messages = result.get(
            "messages",
            [],
        )

        answer = ""

        for current_message in reversed(
            messages
        ):
            if isinstance(
                current_message,
                AIMessage,
            ):
                answer = str(
                    current_message.content
                )
                break

        if not answer:
            raise RuntimeError(
                "RAG graph returned no "
                "assistant answer"
            )

        retrieved_chunks = result.get(
            "retrieved_chunks",
            [],
        )

        sources: List[ChatSource] = []

        source_payload = []

        for index, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            source = ChatSource(
                source_number=index,
                chunk_id=chunk[
                    "chunk_id"
                ],
                document_id=chunk[
                    "document_id"
                ],
                chunk_index=chunk[
                    "chunk_index"
                ],
                content=chunk["content"],
                metadata_json=chunk[
                    "metadata_json"
                ],
                distance=chunk[
                    "distance"
                ],
            )

            sources.append(source)

            source_payload.append(
                source.model_dump()
            )

        chat_history_service.create_message(
            db=db,
            chat_thread_id=thread.id,
            role="assistant",
            content=answer,
            sources=source_payload,
        )

        (
            chat_history_service
            .update_thread_title_from_message(
                thread=thread,
                message=message,
            )
        )

        chat_history_service.touch_thread(
            thread=thread
        )

        db.commit()

        return ChatResponse(
            thread_id=thread_id,
            answer=answer,
            sources=sources,
        )

    except Exception:
        db.rollback()
        raise