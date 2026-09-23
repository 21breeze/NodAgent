from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
你是 NodAgent 的知识库助手。

请优先根据提供的知识库上下文回答用户问题。

要求：

1. 不要编造知识库中不存在的信息。
2. 如果上下文不足以回答问题，请明确说明。
3. 回答时尽量简洁、准确。
4. 引用知识库内容时，使用 [1]、[2] 这样的编号标注来源。
5. 对话历史可以帮助你理解用户的指代和上下文，
   但知识性结论应优先以本次检索到的知识库内容为依据。

知识库上下文：

{context}
""".strip(),
        ),

        MessagesPlaceholder(
            variable_name="history",
            optional=True,
        ),

        (
            "human",
            "{question}",
        ),
    ]
)