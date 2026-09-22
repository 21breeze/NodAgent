from fastapi import APIRouter

from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.workspaces import router as workspace_router


api_router = APIRouter(prefix="/api")

api_router.include_router(health_router)
api_router.include_router(workspace_router)