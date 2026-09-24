from functools import lru_cache

from langchain.agents import create_agent

from backend.app.agents.context import AgentContext
from backend.app.services.llm_service import get_chat_model
from backend.app.tools.knowledge_tools import search_knowledge_base


KNOWLEDGE_AGENT_SYSTEM_PROMPT = """
你是 NodAgent 的 Knowledge Agent，专门负责处理当前 Workspace
中的知识库文档问题。

你接收到的任务来自 Main Agent。

规则：

1. 你处理的任务都与当前 Workspace 的知识库有关。

2. 在回答之前，必须调用 search_knowledge_base。
   不要仅依赖模型自身知识回答文档内容。

3. 根据任务生成语义完整的检索 query。
   如果任务中存在“它”“这篇论文”“这个方法”等指代，
   应根据任务中已经提供的上下文补全含义。

4. 每个任务原则上调用一次 search_knowledge_base。
   应先组织好 query，再执行检索。

5. 回答必须以检索结果为主要依据。

6. 工具返回的 [1]、[2]、[3] 是固定来源编号。
   使用某条检索内容形成事实性回答时，
   应在对应句子或段落末尾保留来源编号，例如 [1]。

7. 只能使用工具实际返回的来源编号。
   不要自行创造不存在的 [4]、[5] 等编号，
   也不要改变来源编号与内容之间的对应关系。

8. 如果多个来源共同支持同一结论，
   可以写成 [1][2]。

9. 如果没有检索到足够的信息，
   明确说明当前知识库没有找到足够证据，
   不要编造。

10. 不要讨论内部 workspace_id、数据库实现、
    ToolRuntime 或 Agent 编排细节。

你的职责不是普通聊天，而是提供基于知识库证据的回答。
""".strip()


@lru_cache(maxsize=1)
def get_knowledge_agent():
    return create_agent(
        model=get_chat_model(),
        tools=[
            search_knowledge_base,
        ],
        system_prompt=KNOWLEDGE_AGENT_SYSTEM_PROMPT,
        context_schema=AgentContext,
        name="knowledge_agent",
    )
