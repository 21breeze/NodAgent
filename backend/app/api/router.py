from fastapi import APIRouter

from backend.app.api.routes.chat import (
    router as chat_router,
)
from backend.app.api.routes.documents import (
    router as document_router,
)
from backend.app.api.routes.health import (
    router as health_router,
)
from backend.app.api.routes.search import (
    router as search_router,
)
from backend.app.api.routes.workspaces import (
    router as workspace_router,
)


api_router = APIRouter(
    prefix="/api"
)


api_router.include_router(
    health_router
)

api_router.include_router(
    workspace_router
)

api_router.include_router(
    document_router
)

api_router.include_router(
    search_router
)

api_router.include_router(
    chat_router
)