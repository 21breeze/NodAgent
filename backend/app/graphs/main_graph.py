import asyncio
from functools import lru_cache
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
)

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import (
    END,
    START,
    StateGraph,
    add_messages,
)
from langgraph.runtime import Runtime
from langgraph.types import (
    interrupt,
)
from pydantic import (
    BaseModel,
    Field,
)
from typing_extensions import TypedDict

from backend.app.agents.context import (
    AgentContext,
)
from backend.app.agents.document_agent import (
    get_document_agent,
)
from backend.app.agents.knowledge_agent import (
    get_knowledge_agent,
)
from backend.app.services import (
    document_action_service,
)
from backend.app.services.document_action_service import (
    DocumentActionError,
)
from backend.app.services.llm_service import (
    get_chat_model,
)


RouteName = Literal[
    "direct",
    "knowledge",
    "document",
    "delete_document",
]


class MainGraphState(
    TypedDict,
    total=False,
):
    messages: Annotated[
        List[AnyMessage],
        add_messages,
    ]

    route: RouteName

    specialist_task: str

    specialist_answer: str

    specialist_agent: str

    sources: List[
        Dict[str, Any]
    ]

    documents: List[
        Dict[str, Any]
    ]

    action_document_id: Optional[
        int
    ]

    action_ready: bool

    action_approved: bool

    pending_action: Dict[
        str,
        Any,
    ]


class RouteDecision(BaseModel):
    route: RouteName = Field(
        description=(
            "Destination for the "
            "current request."
        )
    )

    task: str = Field(
        description=(
            "Standalone task with "
            "references resolved from "
            "conversation history."
        )
    )

    document_id: Optional[int] = Field(
        default=None,
        description=(
            "Document ID when the "
            "request targets a specific "
            "document. Do not invent "
            "an ID."
        ),
    )


class DeleteApproval(BaseModel):
    decision: Literal[
        "approve",
        "reject",
    ]


ROUTER_SYSTEM_PROMPT = """
你是 NodAgent 的 Router。

你不负责回答问题。

你只负责根据当前问题和完整聊天历史，
决定应该进入哪个处理节点。

只能选择：

direct
knowledge
document
delete_document

==================================================
direct
==================================================

普通聊天、通用知识问题，
以及已经能够直接从长期记忆中回答的问题。

例如：

Python 字典是什么？
什么是二分查找？
FastAPI Depends 是什么？

我之前对代码修改有什么要求？
当前 Workspace 主要是什么项目？

如果当前用户或 Workspace 的长期记忆
已经足以回答问题，应选择 direct，
不要为了重复确认而进入 knowledge。

==================================================
knowledge
==================================================

只有当回答确实需要读取 Workspace
中的文档正文、Chunk 或知识库证据时，
才选择 knowledge。

例如：

这篇论文主要讲了什么？
论文有什么创新？
总结知识库里的文档。
根据我的简历介绍项目。

如果长期记忆已经包含足够的信息，
不要进入 knowledge。

==================================================
document
==================================================

查询文档管理信息。

例如：

我有哪些文档？
哪些文档处理完成？
7 号文档状态是什么？
某个文档有多少 Chunk？

==================================================
delete_document
==================================================

用户明确要求删除某个文档。

例如：

删除 7 号文档。
把刚才那个文件删掉。
删除 document_id=12。

删除属于有副作用的高风险操作，
必须进入 delete_document，
不能进入普通 document。

==================================================

如果用户说：

它
这个
这个文件
刚才那个文档
这篇论文

必须结合聊天历史确定具体对象。

如果能够从历史明确得到 document_id，
请写入 document_id。

如果无法可靠确定 document_id：

document_id 必须为 null。

绝对不要猜 document_id。

task 必须是一个完整、独立可理解的任务。

例如：

用户：
“把它删掉”

历史中明确知道它是 document_id=7。

则：

route = delete_document
document_id = 7
task = 删除 document_id=7 的文档。
""".strip()


CHAT_SYSTEM_PROMPT = """
你是 NodAgent 最终负责与用户交流的 Chat Agent。

你的职责：

1. 根据用户问题、聊天历史以及专业节点结果，
   生成最终面向用户的回答。

2. 如果问题涉及 Workspace 文档正文、
   论文、简历或知识库证据，
   必须以专业节点结果为准。

   如果问题只是询问长期记忆中已经明确保存的
   用户偏好或 Workspace 高层业务信息，
   可以直接依据长期记忆回答，
   不要求再次查询知识库。

3. 如果 Knowledge Agent 返回了
   [1]、[2] 等来源编号，应保留。

4. 如果操作节点告诉你操作成功，
   明确说明操作已经完成。

5. 如果操作被用户取消，
   明确告诉用户操作已经取消。

6. 如果操作无法执行，
   根据专业节点给出的原因进行说明。

7. 不要告诉用户内部存在 Router、
   StateGraph、节点、ToolRuntime 等实现细节。

8. direct 路由没有 specialist_answer 时，
   直接根据聊天历史回答用户。
""".strip()


@lru_cache(maxsize=1)
def get_router_model():
    return (
        get_chat_model()
        .with_structured_output(
            RouteDecision,
            method="function_calling",
        )
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


def get_latest_user_message(
    messages: List[AnyMessage],
) -> str:
    for message in reversed(
        messages
    ):
        if isinstance(
            message,
            HumanMessage,
        ):
            return get_message_text(
                message
            )

    return ""


def get_latest_ai_answer(
    messages: List[Any],
) -> str:
    for message in reversed(
        messages
    ):
        if not isinstance(
            message,
            AIMessage,
        ):
            continue

        text = get_message_text(
            message
        )

        if text:
            return text

    return ""


def extract_sources(
    messages: List[Any],
) -> List[
    Dict[str, Any]
]:
    sources_by_chunk: Dict[
        int,
        Dict[str, Any],
    ] = {}

    for message in messages:
        if not isinstance(
            message,
            ToolMessage,
        ):
            continue

        artifact = (
            message.artifact
        )

        if not isinstance(
            artifact,
            dict,
        ):
            continue

        raw_sources = artifact.get(
            "sources",
            [],
        )

        if not isinstance(
            raw_sources,
            list,
        ):
            continue

        for raw_source in raw_sources:
            if not isinstance(
                raw_source,
                dict,
            ):
                continue

            chunk_id = raw_source.get(
                "chunk_id"
            )

            if chunk_id is None:
                continue

            sources_by_chunk[
                int(chunk_id)
            ] = raw_source

    return list(
        sources_by_chunk.values()
    )


def extract_documents(
    messages: List[Any],
) -> List[
    Dict[str, Any]
]:
    documents_by_id: Dict[
        int,
        Dict[str, Any],
    ] = {}

    for message in messages:
        if not isinstance(
            message,
            ToolMessage,
        ):
            continue

        artifact = (
            message.artifact
        )

        if not isinstance(
            artifact,
            dict,
        ):
            continue

        raw_documents = artifact.get(
            "documents",
            [],
        )

        if not isinstance(
            raw_documents,
            list,
        ):
            continue

        for raw_document in (
            raw_documents
        ):
            if not isinstance(
                raw_document,
                dict,
            ):
                continue

            document_id = (
                raw_document.get(
                    "id"
                )
            )

            if document_id is None:
                continue

            documents_by_id[
                int(document_id)
            ] = raw_document

    return list(
        documents_by_id.values()
    )


async def router_node(
    state: MainGraphState,
    runtime: Runtime[
        AgentContext
    ],
):
    messages = state.get(
        "messages",
        [],
    )

    router_system_prompt = (
        ROUTER_SYSTEM_PROMPT
    )

    memory_prompt = (
        runtime.context.get(
            "memory_prompt",
            "",
        )
    )

    if memory_prompt:
        router_system_prompt += (
            "\n\n"
            "==================================================\n"
            "当前可用长期记忆\n"
            "==================================================\n"
            f"{memory_prompt}\n\n"
            "这些长期记忆已经可以作为 Router "
            "判断路由时的上下文。"
            "如果长期记忆本身足以回答用户问题，"
            "请选择 direct。"
            "只有确实需要读取 Workspace 文档正文时，"
            "才选择 knowledge。"
        )

    decision = await (
        get_router_model().ainvoke(
            [
                SystemMessage(
                    content=(
                        router_system_prompt
                    )
                ),
                *messages,
            ]
        )
    )

    return {
        "route": decision.route,

        "specialist_task": (
            decision.task
        ),

        "action_document_id": (
            decision.document_id
        ),

        "specialist_answer": "",

        "specialist_agent": "",

        "sources": [],

        "documents": [],

        "action_ready": False,

        "action_approved": False,

        "pending_action": {},
    }


def route_after_router(
    state: MainGraphState,
) -> RouteName:
    return state["route"]


async def knowledge_node(
    state: MainGraphState,
    runtime: Runtime[
        AgentContext
    ],
):
    task = state.get(
        "specialist_task",
        "",
    )

    if not task:
        task = (
            get_latest_user_message(
                state.get(
                    "messages",
                    [],
                )
            )
        )

    result = await (
        get_knowledge_agent()
        .ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=task
                    )
                ]
            },
            context=runtime.context,
        )
    )

    messages = result.get(
        "messages",
        [],
    )

    answer = (
        get_latest_ai_answer(
            messages
        )
    )

    sources = extract_sources(
        messages
    )

    if not answer:
        answer = (
            "知识库专业 Agent "
            "没有生成有效回答。"
        )

    return {
        "specialist_agent": (
            "knowledge"
        ),
        "specialist_answer": answer,
        "sources": sources,
        "documents": [],
    }


async def document_node(
    state: MainGraphState,
    runtime: Runtime[
        AgentContext
    ],
):
    task = state.get(
        "specialist_task",
        "",
    )

    if not task:
        task = (
            get_latest_user_message(
                state.get(
                    "messages",
                    [],
                )
            )
        )

    result = await (
        get_document_agent()
        .ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=task
                    )
                ]
            },
            context=runtime.context,
        )
    )

    messages = result.get(
        "messages",
        [],
    )

    answer = (
        get_latest_ai_answer(
            messages
        )
    )

    documents = extract_documents(
        messages
    )

    if not answer:
        answer = (
            "文档专业 Agent "
            "没有生成有效回答。"
        )

    return {
        "specialist_agent": (
            "document"
        ),
        "specialist_answer": answer,
        "sources": [],
        "documents": documents,
    }


async def prepare_delete_node(
    state: MainGraphState,
    runtime: Runtime[
        AgentContext
    ],
):
    document_id = state.get(
        "action_document_id"
    )

    if document_id is None:
        return {
            "action_ready": False,
            "specialist_agent": (
                "document_action"
            ),
            "specialist_answer": (
                "无法确定需要删除的 "
                "document_id。"
                "请先明确要删除的文档。"
            ),
        }

    workspace_id = (
        runtime.context[
            "workspace_id"
        ]
    )

    try:
        document = await asyncio.to_thread(
            (
                document_action_service
                .get_document_snapshot
            ),
            workspace_id,
            document_id,
        )

    except DocumentActionError as exc:
        return {
            "action_ready": False,
            "specialist_agent": (
                "document_action"
            ),
            "specialist_answer": (
                f"无法准备删除文档："
                f"{exc}"
            ),
        }

    processing_status = (
        document[
            "processing_status"
        ]
    )

    if processing_status in {
        "queued",
        "processing",
    }:
        return {
            "action_ready": False,
            "specialist_agent": (
                "document_action"
            ),
            "specialist_answer": (
                f"文档 "
                f"{document_id} "
                f"当前状态为 "
                f"{processing_status}，"
                "正在处理的文档暂时"
                "不能删除。"
            ),
        }

    return {
        "action_ready": True,

        "specialist_agent": (
            "document_action"
        ),

        "pending_action": {
            "action": (
                "delete_document"
            ),
            "document_id": (
                document["id"]
            ),
            "filename": (
                document["filename"]
            ),
            "processing_status": (
                processing_status
            ),
        },
    }


def route_after_prepare_delete(
    state: MainGraphState,
) -> Literal[
    "confirm_delete",
    "chat",
]:
    if state.get(
        "action_ready",
        False,
    ):
        return "confirm_delete"

    return "chat"


def confirm_delete_node(
    state: MainGraphState,
):
    pending_action = state.get(
        "pending_action",
        {},
    )

    document_id = (
        pending_action.get(
            "document_id"
        )
    )

    filename = (
        pending_action.get(
            "filename"
        )
    )

    approval = interrupt(
        {
            "type": (
                "approval_required"
            ),

            "action": (
                "delete_document"
            ),

            "document_id": (
                document_id
            ),

            "filename": (
                filename
            ),

            "message": (
                f"确认删除文档 "
                f"{document_id} "
                f"({filename})？"
                "删除后文档记录、"
                "向量 Chunk 和本地文件"
                "都会被移除。"
            ),
        },
        response_schema=(
            DeleteApproval
        ),
    )

    if isinstance(
        approval,
        DeleteApproval,
    ):
        decision = (
            approval.decision
        )

    elif isinstance(
        approval,
        dict,
    ):
        decision = (
            approval.get(
                "decision"
            )
        )

    else:
        decision = None

    approved = (
        decision == "approve"
    )

    if approved:
        return {
            "action_approved": True,
            "specialist_answer": "",
        }

    return {
        "action_approved": False,

        "specialist_agent": (
            "document_action"
        ),

        "specialist_answer": (
            f"已取消删除文档 "
            f"{document_id} "
            f"({filename})。"
        ),
    }


def route_after_delete_confirmation(
    state: MainGraphState,
) -> Literal[
    "execute_delete",
    "chat",
]:
    if state.get(
        "action_approved",
        False,
    ):
        return "execute_delete"

    return "chat"


async def execute_delete_node(
    state: MainGraphState,
    runtime: Runtime[
        AgentContext
    ],
):
    pending_action = state.get(
        "pending_action",
        {},
    )

    document_id = (
        pending_action.get(
            "document_id"
        )
    )

    if document_id is None:
        return {
            "specialist_agent": (
                "document_action"
            ),
            "specialist_answer": (
                "删除失败：缺少 "
                "document_id。"
            ),
        }

    workspace_id = (
        runtime.context[
            "workspace_id"
        ]
    )

    try:
        result = await asyncio.to_thread(
            (
                document_action_service
                .delete_document
            ),
            workspace_id,
            int(document_id),
        )

    except DocumentActionError as exc:
        return {
            "specialist_agent": (
                "document_action"
            ),

            "specialist_answer": (
                f"删除文档失败："
                f"{exc}"
            ),

            "pending_action": {},
        }

    return {
        "specialist_agent": (
            "document_action"
        ),

        "specialist_answer": (
            f"文档 "
            f"{result['document_id']} "
            f"({result['filename']}) "
            "已删除。"
        ),

        "pending_action": {},

        "action_ready": False,

        "action_approved": False,

        "documents": [],

        "sources": [],
    }


async def chat_node(
    state: MainGraphState,
    runtime: Runtime[
        AgentContext
    ],
):
    specialist_answer = (
        state.get(
            "specialist_answer",
            "",
        )
    )

    specialist_agent = (
        state.get(
            "specialist_agent",
            "",
        )
    )

    system_prompt = (
        CHAT_SYSTEM_PROMPT
    )

    memory_prompt = (
        runtime.context.get(
            "memory_prompt",
            "",
        )
    )

    if memory_prompt:
        system_prompt += (
            "\n\n"
            "以下是当前用户和 Workspace "
            "的长期记忆。"
            "这些信息可以作为回答时的长期上下文。"
            "如果长期记忆与用户当前消息冲突，"
            "以用户当前消息为准。"
            "不要向用户暴露记忆系统的内部实现。"
            "\n\n"
            f"{memory_prompt}"
        )

    if specialist_answer:
        system_prompt += (
            "\n\n"
            "本轮已有专业处理结果："
            "\n"
            f"专业类型："
            f"{specialist_agent}"
            "\n\n"
            "结果："
            "\n"
            "--------------------"
            "\n"
            f"{specialist_answer}"
            "\n"
            "--------------------"
            "\n"
            "请基于这个结果生成"
            "最终用户回答。"
        )

    response = await (
        get_chat_model().ainvoke(
            [
                SystemMessage(
                    content=(
                        system_prompt
                    )
                ),
                *state.get(
                    "messages",
                    [],
                ),
            ]
        )
    )

    return {
        "messages": [
            response
        ]
    }


def build_main_graph(
    checkpointer,
):
    graph = StateGraph(
        MainGraphState,
        context_schema=AgentContext,
    )

    graph.add_node(
        "router",
        router_node,
    )

    graph.add_node(
        "knowledge",
        knowledge_node,
    )

    graph.add_node(
        "document",
        document_node,
    )

    graph.add_node(
        "prepare_delete",
        prepare_delete_node,
    )

    graph.add_node(
        "confirm_delete",
        confirm_delete_node,
    )

    graph.add_node(
        "execute_delete",
        execute_delete_node,
    )

    graph.add_node(
        "chat",
        chat_node,
    )

    graph.add_edge(
        START,
        "router",
    )

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "direct": "chat",

            "knowledge": (
                "knowledge"
            ),

            "document": (
                "document"
            ),

            "delete_document": (
                "prepare_delete"
            ),
        },
    )

    graph.add_edge(
        "knowledge",
        "chat",
    )

    graph.add_edge(
        "document",
        "chat",
    )

    graph.add_conditional_edges(
        "prepare_delete",
        route_after_prepare_delete,
        {
            "confirm_delete": (
                "confirm_delete"
            ),

            "chat": "chat",
        },
    )

    graph.add_conditional_edges(
        "confirm_delete",
        route_after_delete_confirmation,
        {
            "execute_delete": (
                "execute_delete"
            ),

            "chat": "chat",
        },
    )

    graph.add_edge(
        "execute_delete",
        "chat",
    )

    graph.add_edge(
        "chat",
        END,
    )

    return graph.compile(
        checkpointer=checkpointer,
        name="nodagent_main_graph",
    )