from fastapi import APIRouter

from backend.app.api.routes.chat import (
    router as chat_router,
)
from backend.app.api.routes.chat_history import (
    router as chat_history_router,
)
from backend.app.api.routes.documents import (
    router as documents_router,
)
from backend.app.api.routes.health import (
    router as health_router,
)
from backend.app.api.routes.memories import (
    router as memories_router,
)
from backend.app.api.routes.search import (
    router as search_router,
)
from backend.app.api.routes.workspaces import (
    router as workspaces_router,
)


api_router = APIRouter(
    prefix="/api",
)

api_router.include_router(
    health_router
)

api_router.include_router(
    workspaces_router
)

api_router.include_router(
    documents_router
)

api_router.include_router(
    search_router
)

api_router.include_router(
    memories_router
)

api_router.include_router(
    chat_router
)

api_router.include_router(
    chat_history_router
)
