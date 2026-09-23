import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[3]

load_dotenv(BASE_DIR / ".env")


# =========================
# PostgreSQL
# =========================

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv(
    "POSTGRES_PASSWORD"
)
POSTGRES_DB = os.getenv("POSTGRES_DB")

POSTGRES_HOST = os.getenv(
    "POSTGRES_HOST",
    "localhost",
)

POSTGRES_PORT = os.getenv(
    "POSTGRES_PORT",
    "15432",
)


if not POSTGRES_USER:
    raise RuntimeError(
        "POSTGRES_USER is not configured"
    )

if not POSTGRES_PASSWORD:
    raise RuntimeError(
        "POSTGRES_PASSWORD is not configured"
    )

if not POSTGRES_DB:
    raise RuntimeError(
        "POSTGRES_DB is not configured"
    )


DATABASE_URL = (
    "postgresql+psycopg://"
    f"{POSTGRES_USER}:"
    f"{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:"
    f"{POSTGRES_PORT}/"
    f"{POSTGRES_DB}"
)


CHECKPOINT_DATABASE_URL = (
    "postgresql://"
    f"{POSTGRES_USER}:"
    f"{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:"
    f"{POSTGRES_PORT}/"
    f"{POSTGRES_DB}"
)


# =========================
# Ollama Embedding
# =========================

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "qwen3-embedding:4b",
)

EMBEDDING_DIMENSION = int(
    os.getenv(
        "EMBEDDING_DIMENSION",
        "1024",
    )
)


# =========================
# DeepSeek LLM
# =========================

DEEPSEEK_API_KEY = os.getenv(
    "DEEPSEEK_API_KEY"
)

DEEPSEEK_MODEL = os.getenv(
    "DEEPSEEK_MODEL",
    "deepseek-flash",
)


if not DEEPSEEK_API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not configured"
    )


# =========================
# LangGraph
# =========================

LANGGRAPH_STRICT_MSGPACK = os.getenv(
    "LANGGRAPH_STRICT_MSGPACK",
    "true",
)

os.environ[
    "LANGGRAPH_STRICT_MSGPACK"
] = LANGGRAPH_STRICT_MSGPACK