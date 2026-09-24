from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Union,
)

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
)
from langgraph.graph.state import (
    CompiledStateGraph,
)
from langgraph.types import (
    Command,
)

from backend.app.db.database import (
    SessionLocal,
)
from backend.app.models.chat_thread import (
    ChatThread,
)
from backend.app.schemas.chat import (
    ChatInterruptResponse,
    ChatResponse,
    ChatSource,
)
from backend.app.services import (
    agent_context_service,
    chat_history_service,
)


class PendingInterruptError(
    RuntimeError
):
    pass


class NoPendingInterruptError(
    RuntimeError
):
    pass


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


def get_current_turn_messages(
    messages: List[Any],
) -> List[Any]:
    for index in range(
        len(messages) - 1,
        -1,
        -1,
    ):
        if isinstance(
            messages[index],
            HumanMessage,
        ):
            return messages[
                index + 1:
            ]

    return messages


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
            text = get_message_text(
                message
            )

            if text:
                return text

    return ""


def parse_sources(
    raw_sources: Any,
) -> List[ChatSource]:
    if not isinstance(
        raw_sources,
        list,
    ):
        return []

    sources_by_chunk: Dict[
        int,
        ChatSource,
    ] = {}

    for raw_source in raw_sources:
        if not isinstance(
            raw_source,
            dict,
        ):
            continue

        try:
            source = ChatSource(
                **raw_source
            )

        except Exception:
            continue

        sources_by_chunk[
            source.chunk_id
        ] = source

    return list(
        sources_by_chunk.values()
    )


def save_user_message(
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

        (
            chat_history_service
            .create_message(
                db=db,
                chat_thread_id=(
                    thread.id
                ),
                role="user",
                content=message,
            )
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
    chat_thread_id: int,
    answer: str,
    sources: List[
        ChatSource
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

        source_payload = [
            source.model_dump()
            for source in sources
        ]

        assistant_message = (
            chat_history_service
            .create_message(
                db=db,
                chat_thread_id=(
                    thread.id
                ),
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


def build_thread_config(
    user_id: str,
    workspace_id: int,
    thread_id: str,
):
    internal_thread_id = (
        chat_history_service
        .build_internal_thread_id(
            user_id=user_id,
            workspace_id=workspace_id,
            thread_id=thread_id,
        )
    )

    return {
        "configurable": {
            "thread_id": (
                internal_thread_id
            )
        }
    }


def build_agent_context(
    user_id: str,
    workspace_id: int,
    top_k: int,
):
    return (
        agent_context_service
        .build_agent_context(
            user_id=user_id,
            workspace_id=workspace_id,
            top_k=top_k,
        )
    )


def build_interrupt_response(
    thread_id: str,
    snapshot,
) -> ChatInterruptResponse:
    pending_interrupt = (
        snapshot.interrupts[0]
    )

    value = (
        pending_interrupt.value
    )

    if not isinstance(
        value,
        dict,
    ):
        value = {
            "message": str(
                value
            )
        }

    return ChatInterruptResponse(
        thread_id=thread_id,
        interrupt_id=(
            pending_interrupt.id
        ),
        interrupt=value,
    )


async def ensure_no_pending_interrupt(
    agent: CompiledStateGraph,
    config,
) -> None:
    snapshot = await agent.aget_state(
        config
    )

    if snapshot.interrupts:
        raise PendingInterruptError(
            "This thread has a "
            "pending human approval"
        )


async def chat(
    agent: CompiledStateGraph,
    chat_thread_id: int,
    workspace_id: int,
    user_id: str,
    thread_id: str,
    message: str,
    top_k: int,
) -> Union[
    ChatResponse,
    ChatInterruptResponse,
]:
    config = build_thread_config(
        user_id=user_id,
        workspace_id=workspace_id,
        thread_id=thread_id,
    )

    context = build_agent_context(
        user_id=user_id,
        workspace_id=workspace_id,
        top_k=top_k,
    )

    await ensure_no_pending_interrupt(
        agent=agent,
        config=config,
    )

    save_user_message(
        chat_thread_id=(
            chat_thread_id
        ),
        message=message,
    )

    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=message
                )
            ]
        },
        config=config,
        context=context,
    )

    snapshot = await agent.aget_state(
        config
    )

    if snapshot.interrupts:
        return build_interrupt_response(
            thread_id=thread_id,
            snapshot=snapshot,
        )

    messages = result.get(
        "messages",
        [],
    )

    current_turn_messages = (
        get_current_turn_messages(
            messages
        )
    )

    answer = get_latest_ai_answer(
        current_turn_messages
    )

    if not answer:
        raise RuntimeError(
            "Main graph returned no "
            "assistant answer"
        )

    sources = parse_sources(
        result.get(
            "sources",
            [],
        )
    )

    save_assistant_message(
        chat_thread_id=(
            chat_thread_id
        ),
        answer=answer,
        sources=sources,
    )

    return ChatResponse(
        thread_id=thread_id,
        answer=answer,
        sources=sources,
    )


async def resume_chat(
    agent: CompiledStateGraph,
    chat_thread_id: int,
    workspace_id: int,
    user_id: str,
    thread_id: str,
    decision: str,
    top_k: int,
) -> Union[
    ChatResponse,
    ChatInterruptResponse,
]:
    config = build_thread_config(
        user_id=user_id,
        workspace_id=workspace_id,
        thread_id=thread_id,
    )

    context = build_agent_context(
        user_id=user_id,
        workspace_id=workspace_id,
        top_k=top_k,
    )

    snapshot = await agent.aget_state(
        config
    )

    if not snapshot.interrupts:
        raise NoPendingInterruptError(
            "This thread has no "
            "pending human approval"
        )

    result = await agent.ainvoke(
        Command(
            resume={
                "decision": decision
            }
        ),
        config=config,
        context=context,
    )

    snapshot = await agent.aget_state(
        config
    )

    if snapshot.interrupts:
        return build_interrupt_response(
            thread_id=thread_id,
            snapshot=snapshot,
        )

    messages = result.get(
        "messages",
        [],
    )

    current_turn_messages = (
        get_current_turn_messages(
            messages
        )
    )

    answer = get_latest_ai_answer(
        current_turn_messages
    )

    if not answer:
        raise RuntimeError(
            "Main graph returned no "
            "assistant answer"
        )

    sources = parse_sources(
        result.get(
            "sources",
            [],
        )
    )

    save_assistant_message(
        chat_thread_id=(
            chat_thread_id
        ),
        answer=answer,
        sources=sources,
    )

    return ChatResponse(
        thread_id=thread_id,
        answer=answer,
        sources=sources,
    )


async def _stream_graph_run(
    agent: CompiledStateGraph,
    graph_input,
    chat_thread_id: int,
    thread_id: str,
    config,
    context,
) -> AsyncIterator[
    Dict[str, Any]
]:
    answer_parts: List[str] = []

    final_answer = ""

    sources_by_chunk: Dict[
        int,
        ChatSource,
    ] = {}

    try:
        async for part in agent.astream(
            graph_input,
            config=config,
            context=context,
            stream_mode=[
                "updates",
                "messages",
            ],
            version="v2",
        ):
            part_type = part[
                "type"
            ]

            if part_type == "updates":
                update_data = part[
                    "data"
                ]

                knowledge_update = (
                    update_data.get(
                        "knowledge"
                    )
                )

                if isinstance(
                    knowledge_update,
                    dict,
                ):
                    new_sources = (
                        parse_sources(
                            knowledge_update.get(
                                "sources",
                                [],
                            )
                        )
                    )

                    for source in new_sources:
                        sources_by_chunk[
                            source.chunk_id
                        ] = source

                    if new_sources:
                        yield {
                            "event": (
                                "sources"
                            ),
                            "data": {
                                "sources": [
                                    source.model_dump()
                                    for source
                                    in (
                                        sources_by_chunk
                                        .values()
                                    )
                                ]
                            },
                        }

                chat_update = (
                    update_data.get(
                        "chat"
                    )
                )

                if isinstance(
                    chat_update,
                    dict,
                ):
                    chat_messages = (
                        chat_update.get(
                            "messages",
                            [],
                        )
                    )

                    for chat_message in (
                        chat_messages
                    ):
                        if not isinstance(
                            chat_message,
                            AIMessage,
                        ):
                            continue

                        text = (
                            get_message_text(
                                chat_message
                            )
                        )

                        if text:
                            final_answer = text

            elif part_type == "messages":
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
                    != "chat"
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
                        "content": token
                    },
                }

        snapshot = await agent.aget_state(
            config
        )

        if snapshot.interrupts:
            interrupt_response = (
                build_interrupt_response(
                    thread_id=thread_id,
                    snapshot=snapshot,
                )
            )

            yield {
                "event": "interrupt",
                "data": (
                    interrupt_response
                    .model_dump()
                ),
            }

            return

        answer = (
            final_answer.strip()
            or "".join(
                answer_parts
            ).strip()
        )

        if not answer:
            snapshot_messages = (
                snapshot.values.get(
                    "messages",
                    [],
                )
            )

            answer = (
                get_latest_ai_answer(
                    snapshot_messages
                )
            )

        if not answer:
            raise RuntimeError(
                "Main graph returned no "
                "assistant answer"
            )

        snapshot_sources = (
            snapshot.values.get(
                "sources",
                [],
            )
        )

        sources = parse_sources(
            snapshot_sources
        )

        message_id = (
            save_assistant_message(
                chat_thread_id=(
                    chat_thread_id
                ),
                answer=answer,
                sources=sources,
            )
        )

        yield {
            "event": "done",
            "data": {
                "thread_id": thread_id,
                "message_id": (
                    message_id
                ),
            },
        }

    except Exception as exc:
        print(
            "Main graph stream error:",
            repr(exc),
        )

        yield {
            "event": "error",
            "data": {
                "message": (
                    "Failed to execute "
                    "agent graph"
                )
            },
        }


async def stream_chat(
    agent: CompiledStateGraph,
    chat_thread_id: int,
    workspace_id: int,
    user_id: str,
    thread_id: str,
    message: str,
    top_k: int,
) -> AsyncIterator[
    Dict[str, Any]
]:
    config = build_thread_config(
        user_id=user_id,
        workspace_id=workspace_id,
        thread_id=thread_id,
    )

    context = build_agent_context(
        user_id=user_id,
        workspace_id=workspace_id,
        top_k=top_k,
    )

    snapshot = await agent.aget_state(
        config
    )

    if snapshot.interrupts:
        yield {
            "event": "error",
            "data": {
                "message": (
                    "Thread has a pending "
                    "human approval"
                )
            },
        }

        return

    save_user_message(
        chat_thread_id=(
            chat_thread_id
        ),
        message=message,
    )

    yield {
        "event": "start",
        "data": {
            "thread_id": thread_id,
        },
    }

    async for event in (
        _stream_graph_run(
            agent=agent,
            graph_input={
                "messages": [
                    HumanMessage(
                        content=message
                    )
                ]
            },
            chat_thread_id=(
                chat_thread_id
            ),
            thread_id=thread_id,
            config=config,
            context=context,
        )
    ):
        yield event


async def stream_resume_chat(
    agent: CompiledStateGraph,
    chat_thread_id: int,
    workspace_id: int,
    user_id: str,
    thread_id: str,
    decision: str,
    top_k: int,
) -> AsyncIterator[
    Dict[str, Any]
]:
    config = build_thread_config(
        user_id=user_id,
        workspace_id=workspace_id,
        thread_id=thread_id,
    )

    context = build_agent_context(
        user_id=user_id,
        workspace_id=workspace_id,
        top_k=top_k,
    )

    snapshot = await agent.aget_state(
        config
    )

    if not snapshot.interrupts:
        yield {
            "event": "error",
            "data": {
                "message": (
                    "Thread has no pending "
                    "human approval"
                )
            },
        }

        return

    yield {
        "event": "resume",
        "data": {
            "thread_id": thread_id,
            "decision": decision,
        },
    }

    async for event in (
        _stream_graph_run(
            agent=agent,
            graph_input=Command(
                resume={
                    "decision": decision
                }
            ),
            chat_thread_id=(
                chat_thread_id
            ),
            thread_id=thread_id,
            config=config,
            context=context,
        )
    ):
        yield event