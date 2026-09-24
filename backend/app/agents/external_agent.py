from typing import List

from fastmcp import Client
from langchain.agents import (
    create_agent,
)
from langchain.mcp import (
    MCPAdapter,
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from backend.app.services.llm_service import (
    get_chat_model,
)
from backend.app.services.mcp_service import (
    EXTERNAL_INFO_SERVER_PATH,
)


EXTERNAL_AGENT_SYSTEM_PROMPT = """
你是 NodAgent 的 External Tool Agent。

你的职责是处理需要实时外部信息的问题。

当前 MCP Server 提供的工具可能包括：

- 当前时间查询
- 天气查询
- 其他外部实时信息工具


==================================================
工具使用规则
==================================================

当用户询问当前天气、当前时间等实时信息时，
必须调用对应 MCP Tool。

不要根据模型训练知识猜测实时数据。

必须先获取 Tool 返回结果，
再根据真实结果回答。


==================================================
职责边界
==================================================

你不负责：

- Workspace 文档检索
- RAG
- 文档管理
- 长期记忆管理
- 文档删除

这些任务由其他 Specialist Agent 负责。


==================================================
回答规则
==================================================

调用 Tool 成功后，
基于 Tool 返回的真实数据生成简洁、自然的回答。

不要向用户暴露：

- MCP JSON-RPC
- stdio
- MCPAdapter
- ToolMessage 内部格式
- 子进程实现

如果 Tool 调用失败，
明确说明无法获得实时数据，
不要编造结果。
""".strip()


def get_latest_ai_content(
    messages: List,
) -> str:
    for message in reversed(
        messages
    ):
        if isinstance(
            message,
            AIMessage,
        ):
            content = message.content

            if isinstance(
                content,
                str,
            ):
                return content.strip()

    return ""


async def run_external_agent(
    task: str,
) -> str:
    """
    Execute one External Agent task.

    MCP connection stays open during the
    complete Agent + Tool execution.
    """

    client = Client(
        EXTERNAL_INFO_SERVER_PATH
    )

    async with MCPAdapter(
        client
    ) as adapter:
        tools = await (
            adapter.list_tools(
                cache_mode="refresh"
            )
        )

        if not tools:
            return (
                "当前没有可用的外部 MCP 工具。"
            )

        agent = create_agent(
            model=get_chat_model(),
            tools=tools,
            system_prompt=(
                EXTERNAL_AGENT_SYSTEM_PROMPT
            ),
            name="external_agent",
        )

        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=task
                    )
                ]
            }
        )

        answer = (
            get_latest_ai_content(
                result.get(
                    "messages",
                    [],
                )
            )
        )

        if not answer:
            return (
                "External Agent "
                "没有生成有效回答。"
            )

        return answer
