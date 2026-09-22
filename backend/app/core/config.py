import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[3]

load_dotenv(BASE_DIR / ".env")


POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "15432")


if not POSTGRES_USER:
    raise RuntimeError("POSTGRES_USER is not configured")

if not POSTGRES_PASSWORD:
    raise RuntimeError("POSTGRES_PASSWORD is not configured")

if not POSTGRES_DB:
    raise RuntimeError("POSTGRES_DB is not configured")


DATABASE_URL = (
    "postgresql+psycopg2://"
    f"{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)