from langchain_ollama import OllamaEmbeddings

from backend.app.core.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    OLLAMA_BASE_URL,
)


embedding_model = OllamaEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    dimensions=EMBEDDING_DIMENSION,
)