from typing import List

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
    create_external_info_client,
    create_github_mcp_client,
)


EXTERNAL_AGENT_SYSTEM_PROMPT = """
你是 NodAgent 的 External Tool Agent。

你的职责是处理需要访问外部系统
或实时外部数据的问题。

当前连接两个 MCP Server。

1. External Info MCP

负责：

- 当前天气
- 当前时间


2. GitHub MCP

负责：

- Repository 信息
- Repository 搜索
- Repository 文件
- README
- Issue
- Pull Request


==================================================
工具调用规则
==================================================

涉及实时信息或外部系统数据时，
必须优先调用 MCP Tool。

不要根据模型训练知识猜测：

- 当前天气
- 当前时间
- GitHub Repository 当前状态
- 当前 Issue
- 当前 Pull Request
- Repository 当前文件内容


==================================================
GitHub 权限
==================================================

当前 GitHub MCP 使用只读模式。

不得执行或声称已经执行：

- 创建 Issue
- 修改 Issue
- 删除 Issue
- 创建 Pull Request
- Merge Pull Request
- Push 代码
- 修改 Repository


==================================================
职责边界
==================================================

你不负责：

- Workspace 知识库 RAG
- Workspace 文档管理
- Long-term Memory 管理
- Workspace 文档删除

这些任务由其他 Specialist Agent 负责。


==================================================
回答规则
==================================================

必须根据 MCP Tool 返回的真实数据回答。

调用 Tool 后，
将结果整理成自然语言。

不要向用户暴露：

- MCP JSON-RPC
- MCPAdapter
- stdio
- HTTP Transport
- Docker 内部实现
- ToolMessage 内部格式

如果 Tool 调用失败，
明确说明无法获取对应外部数据，
不要编造。
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
    Execute one external task.

    Two MCP connections are kept alive
    during the complete Agent execution:

    1. Local stdio MCP
    2. GitHub HTTP MCP
    """

    external_info_client = (
        create_external_info_client()
    )

    github_client = (
        create_github_mcp_client()
    )

    async with MCPAdapter(
        external_info_client
    ) as external_info_adapter:

        async with MCPAdapter(
            github_client
        ) as github_adapter:

            external_info_tools = await (
                external_info_adapter.list_tools(
                    cache_mode="refresh"
                )
            )

            github_tools = await (
                github_adapter.list_tools(
                    cache_mode="refresh"
                )
            )

            tools = (
                external_info_tools
                + github_tools
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

            result = await (
                agent.ainvoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=task
                            )
                        ]
                    }
                )
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
