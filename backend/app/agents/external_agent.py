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
    create_external_mcp_client,
)


EXTERNAL_AGENT_SYSTEM_PROMPT = """
你是 NodAgent 的 External Tool Agent。

你的职责是处理需要访问外部系统或实时外部数据的问题。

当前可能连接多个 MCP Server，例如：

1. External Info MCP
   - 当前天气
   - 当前时间

2. GitHub MCP
   - Repository 信息
   - Repository 文件
   - Issue
   - Pull Request


==================================================
Tool 使用原则
==================================================

只要问题依赖实时信息或外部系统数据，
优先调用对应 MCP Tool。

不要根据模型训练知识猜测：

- 当前天气
- 当前时间
- GitHub Repository 当前状态
- 当前 Issue
- 当前 Pull Request
- Repository 当前文件内容


==================================================
GitHub
==================================================

当用户询问：

某个 GitHub 仓库有什么内容？
读取仓库 README。
查看仓库 Issue。
查看 Pull Request。
查看仓库里的文件。

应该调用 GitHub MCP Tool。

GitHub MCP 当前配置为只读模式。

不得声称已经：

- 修改 Repository
- 创建 Issue
- 修改 Issue
- 创建 Pull Request
- Merge Pull Request
- Push 代码

除非未来明确接入具有写权限的 MCP Tool。


==================================================
职责边界
==================================================

你不负责：

- NodAgent Workspace 内部文档 RAG
- Document 管理
- Long-term Memory 管理
- Workspace 文档删除

这些由其他 Specialist Agent 负责。


==================================================
回答原则
==================================================

必须优先使用 Tool 返回的真实结果。

Tool 调用完成后，
将结果转换为自然语言回答。

不要向最终用户暴露：

- MCP JSON-RPC
- MCPAdapter
- stdio
- ToolMessage 内部格式
- Docker 内部实现

如果外部 Tool 调用失败，
明确说明当前无法获得数据，
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
            content = (
                message.content
            )

            if isinstance(
                content,
                str,
            ):
                return (
                    content.strip()
                )

    return ""


async def run_external_agent(
    task: str,
) -> str:
    """
    Run one External Agent task.

    The MCP client connection stays alive
    throughout Agent + Tool execution.
    """

    client = (
        create_external_mcp_client()
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
