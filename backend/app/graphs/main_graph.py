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
from backend.app.agents.memory_agent import (
    get_memory_agent,
)
from backend.app.agents.external_agent import (
    run_external_agent,
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
    "memory",
    "external",
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

    action_document_filename: Optional[
        str
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

    document_filename: Optional[str] = Field(
        default=None,
        description=(
            "Exact filename when a delete "
            "request identifies a document "
            "by name rather than ID. "
            "Do not invent a filename."
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
memory
external
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
memory
==================================================

用户明确要求管理长期记忆时选择 memory。

包括：

记住以后代码修改给完整文件。
以后回答尽量一步一步讲。
把这个偏好保存下来。
我有哪些长期记忆？
你现在记得我什么？
把代码修改偏好改成只给关键修改。
忘掉我的代码修改偏好。
删除这个长期记忆。

只是在询问已经存在的某条长期信息，
并且当前长期记忆已经足够回答时，
可以选择 direct。

只要用户是在保存、更新、列出或删除长期记忆，
必须选择 memory。

==================================================
external
==================================================

需要访问外部系统或实时外部信息时选择 external。

例如：

东京现在天气怎么样？
上海今天气温多少？
纽约现在几点？
东京当前时间是多少？

查看 GitHub 某个 Repository。
读取 GitHub Repository 的 README。
查看 GitHub Issue。
查看 GitHub Pull Request。
读取 GitHub Repository 当前文件内容。

这类问题依赖实时数据或外部系统，
不能只依赖模型训练知识回答，
应交给 External Agent 调用 MCP Tool。

如果问题只是通用知识，
不需要实时外部数据，
选择 direct。

==================================================
delete_document
==================================================

用户明确要求删除某个文档。

例如：

删除 7 号文档。
把刚才那个文件删掉。
删除 document_id=12。
删除 CET6_202209_322110221202516_1.pdf。

如果用户在删除对话中补充文件名、
回答候选编号，或说“确认删除”，
仍应根据聊天历史进入 delete_document。
聊天里的“确认”不能代替人工确认步骤。

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

如果只知道文档的完整文件名，
请把不带引号和路径的名称写入 document_filename，
document_id 保持 null。
文件名是否存在由后续节点查询，
不要因为不知道 ID 就拒绝删除。

如果无法可靠确定 document_id：

document_id 必须为 null。

绝对不要猜 document_id。
也不要猜 document_filename。

task 必须是一个完整、独立可理解的任务。

例如：

用户：
“把它删掉”

历史中明确知道它是 document_id=7。

则：

route = delete_document
document_id = 7
task = 删除 document_id=7 的文档。

用户：
“删除 report.pdf”

则：

route = delete_document
document_id = null
document_filename = report.pdf
task = 删除 report.pdf。
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
   只根据专业节点给出的原因进行说明。
   不要猜测文件名、路径或文档是否存在。
   不要声称已经检索过没有实际查询的记录。

7. 系统支持删除文档，但执行前需要用户确认。
   不要因为 Document Agent 只有查询工具，
   就告诉用户整个系统无法删除文档。

8. 不要告诉用户内部存在 Router、
   StateGraph、节点、ToolRuntime 等实现细节。

9. direct 路由没有 specialist_answer 时，
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
            "如果用户要保存、更新、列出或删除长期记忆，"
            "请选择 memory。"
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

        "action_document_filename": (
            decision.document_filename
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


async def memory_node(
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
        get_memory_agent()
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

    if not answer:
        answer = (
            "长期记忆 Agent "
            "没有生成有效回答。"
        )

    return {
        "specialist_agent": (
            "memory"
        ),
        "specialist_answer": answer,
        "sources": [],
        "documents": [],
    }


async def external_node(
    state: MainGraphState,
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

    answer = await (
        run_external_agent(
            task
        )
    )

    return {
        "specialist_agent": (
            "external"
        ),
        "specialist_answer": answer,
        "sources": [],
        "documents": [],
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

    document_filename = state.get(
        "action_document_filename"
    )

    workspace_id = (
        runtime.context[
            "workspace_id"
        ]
    )

    if document_id is not None:
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

    elif document_filename:
        matches = await asyncio.to_thread(
            (
                document_action_service
                .find_document_snapshots_by_filename
            ),
            workspace_id,
            document_filename,
        )

        if not matches:
            return {
                "action_ready": False,
                "specialist_agent": (
                    "document_action"
                ),
                "specialist_answer": (
                    "当前 Workspace 没有找到"
                    f"文件名为 {document_filename} "
                    "的文档。"
                ),
            }

        if len(matches) > 1:
            document_ids = ", ".join(
                str(document["id"])
                for document in matches
            )

            return {
                "action_ready": False,
                "specialist_agent": (
                    "document_action"
                ),
                "specialist_answer": (
                    f"文件名 {document_filename} "
                    "对应多个文档，ID 分别为 "
                    f"{document_ids}。"
                    "请指定要删除的文档 ID。"
                ),
            }

        document = matches[0]
        document_id = document["id"]

    else:
        return {
            "action_ready": False,
            "specialist_agent": (
                "document_action"
            ),
            "specialist_answer": (
                "无法确定需要删除的文档。"
                "请提供文档 ID 或完整文件名。"
            ),
        }

    if (
        document_filename
        and document["filename"]
        != (
            document_action_service
            .normalize_document_filename(
                document_filename
            )
        )
    ):
        return {
            "action_ready": False,
            "specialist_agent": (
                "document_action"
            ),
            "specialist_answer": (
                "文档 ID 与文件名不一致。"
                "请核对后重新发起删除。"
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
        "memory",
        memory_node,
    )

    graph.add_node(
        "external",
        external_node,
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

            "memory": (
                "memory"
            ),

            "external": (
                "external"
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

    graph.add_edge(
        "memory",
        "chat",
    )

    graph.add_edge(
        "external",
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
