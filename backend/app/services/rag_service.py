from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Tuple,
)

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
)
from langgraph.graph.state import (
    CompiledStateGraph,
)
from sqlalchemy.orm import Session

from backend.app.db.database import (
    SessionLocal,
)
from backend.app.models.chat_thread import (
    ChatThread,
)
from backend.app.schemas.chat import (
    ChatResponse,
    ChatSource,
)
from backend.app.services import (
    chat_history_service,
)


def build_sources(
    retrieved_chunks: List[
        Dict[str, Any]
    ],
) -> Tuple[
    List[ChatSource],
    List[Dict[str, Any]],
]:
    sources: List[ChatSource] = []

    source_payload: List[
        Dict[str, Any]
    ] = []

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
            content=chunk[
                "content"
            ],
            metadata_json=chunk[
                "metadata_json"
            ],
            distance=chunk[
                "distance"
            ],
        )

        sources.append(
            source
        )

        source_payload.append(
            source.model_dump()
        )

    return (
        sources,
        source_payload,
    )


def get_message_text(
    message: Any,
) -> str:
    content = getattr(
        message,
        "content",
        "",
    )

    if isinstance(
        content,
        str,
    ):
        return content

    if isinstance(
        content,
        list,
    ):
        text_parts: List[str] = []

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

        return "".join(
            text_parts
        )

    return ""


def get_latest_ai_answer(
    messages: List[Any],
) -> str:
    for message in reversed(
        messages
    ):
        if isinstance(
            message,
            AIMessage,
        ):
            return get_message_text(
                message
            )

    return ""


def save_user_message(
    db: Session,
    thread: ChatThread,
    message: str,
) -> None:
    chat_history_service.create_message(
        db=db,
        chat_thread_id=thread.id,
        role="user",
        content=message,
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


def save_assistant_message(
    db: Session,
    thread: ChatThread,
    answer: str,
    source_payload: List[
        Dict[str, Any]
    ],
):
    assistant_message = (
        chat_history_service
        .create_message(
            db=db,
            chat_thread_id=thread.id,
            role="assistant",
            content=answer,
            sources=source_payload,
        )
    )

    chat_history_service.touch_thread(
        thread=thread
    )

    db.commit()

    db.refresh(
        assistant_message
    )

    return assistant_message


def save_stream_user_message(
    chat_thread_id: int,
    message: str,
) -> None:
    with SessionLocal() as db:
        thread = (
            db.query(ChatThread)
            .filter(
                ChatThread.id
                == chat_thread_id
            )
            .first()
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


def save_stream_assistant_message(
    chat_thread_id: int,
    answer: str,
    source_payload: List[
        Dict[str, Any]
    ],
) -> int:
    with SessionLocal() as db:
        thread = (
            db.query(ChatThread)
            .filter(
                ChatThread.id
                == chat_thread_id
            )
            .first()
        )

        if thread is None:
            raise ValueError(
                "Chat thread not found"
            )

        assistant_message = (
            chat_history_service
            .create_message(
                db=db,
                chat_thread_id=thread.id,
                role="assistant",
                content=answer,
                sources=source_payload,
            )
        )

        chat_history_service.touch_thread(
            thread=thread
        )

        db.commit()

        db.refresh(
            assistant_message
        )

        return assistant_message.id


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

    save_user_message(
        db=db,
        thread=thread,
        message=message,
    )

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

        answer = (
            get_latest_ai_answer(
                messages=messages
            )
        )

        if not answer:
            raise RuntimeError(
                "RAG graph returned no "
                "assistant answer"
            )

        retrieved_chunks = (
            result.get(
                "retrieved_chunks",
                [],
            )
        )

        (
            sources,
            source_payload,
        ) = build_sources(
            retrieved_chunks
        )

        save_assistant_message(
            db=db,
            thread=thread,
            answer=answer,
            source_payload=(
                source_payload
            ),
        )

        return ChatResponse(
            thread_id=thread_id,
            answer=answer,
            sources=sources,
        )

    except Exception:
        db.rollback()
        raise


async def stream_chat(
    graph: CompiledStateGraph,
    chat_thread_id: int,
    workspace_id: int,
    user_id: str,
    thread_id: str,
    message: str,
    top_k: int,
) -> AsyncIterator[
    Dict[str, Any]
]:
    answer_parts: List[str] = []

    source_payload: List[
        Dict[str, Any]
    ] = []

    try:
        save_stream_user_message(
            chat_thread_id=(
                chat_thread_id
            ),
            message=message,
        )

        yield {
            "event": "start",
            "data": {
                "thread_id": (
                    thread_id
                ),
            },
        }

        internal_thread_id = (
            chat_history_service
            .build_internal_thread_id(
                user_id=user_id,
                workspace_id=(
                    workspace_id
                ),
                thread_id=thread_id,
            )
        )

        async for part in graph.astream(
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
            stream_mode=[
                "updates",
                "messages",
            ],
            version="v2",
        ):
            part_type = part[
                "type"
            ]

            if (
                part_type
                == "updates"
            ):
                update_data = part[
                    "data"
                ]

                retrieve_update = (
                    update_data.get(
                        "retrieve"
                    )
                )

                if retrieve_update:
                    retrieved_chunks = (
                        retrieve_update.get(
                            "retrieved_chunks",
                            [],
                        )
                    )

                    (
                        sources,
                        source_payload,
                    ) = build_sources(
                        retrieved_chunks
                    )

                    yield {
                        "event": "sources",
                        "data": {
                            "sources": [
                                source.model_dump()
                                for source
                                in sources
                            ],
                        },
                    }

                no_context_update = (
                    update_data.get(
                        "no_context"
                    )
                )

                if no_context_update:
                    no_context_messages = (
                        no_context_update.get(
                            "messages",
                            [],
                        )
                    )

                    no_context_answer = (
                        get_latest_ai_answer(
                            messages=(
                                no_context_messages
                            )
                        )
                    )

                    if no_context_answer:
                        answer_parts.append(
                            no_context_answer
                        )

                        yield {
                            "event": "token",
                            "data": {
                                "content": (
                                    no_context_answer
                                ),
                            },
                        }

            elif (
                part_type
                == "messages"
            ):
                (
                    message_chunk,
                    metadata,
                ) = part[
                    "data"
                ]

                if (
                    metadata.get(
                        "langgraph_node"
                    )
                    != "generate"
                ):
                    continue

                if not isinstance(
                    message_chunk,
                    AIMessageChunk,
                ):
                    continue

                token = get_message_text(
                    message_chunk
                )

                if not token:
                    continue

                answer_parts.append(
                    token
                )

                yield {
                    "event": "token",
                    "data": {
                        "content": token,
                    },
                }

        answer = "".join(
            answer_parts
        ).strip()

        if not answer:
            raise RuntimeError(
                "RAG graph returned no "
                "assistant answer"
            )

        assistant_message_id = (
            save_stream_assistant_message(
                chat_thread_id=(
                    chat_thread_id
                ),
                answer=answer,
                source_payload=(
                    source_payload
                ),
            )
        )

        yield {
            "event": "done",
            "data": {
                "thread_id": (
                    thread_id
                ),
                "message_id": (
                    assistant_message_id
                ),
            },
        }

    except Exception as exc:
        print(
            "Stream chat error:",
            repr(exc),
        )

        yield {
            "event": "error",
            "data": {
                "message": (
                    "Failed to generate "
                    "chat response"
                ),
            },
        }