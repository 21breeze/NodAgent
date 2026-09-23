from functools import lru_cache

from langchain_deepseek import ChatDeepSeek

from backend.app.core.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
)


@lru_cache(maxsize=1)
def get_chat_model() -> ChatDeepSeek:
    return ChatDeepSeek(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        timeout=60,
        max_retries=2,
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        },
    )