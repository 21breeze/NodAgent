from fastapi import FastAPI

from backend.app.api.router import api_router
from backend.app.db.database import Base, engine
from backend.app.models import Workspace


Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="NodAgent",
    description="AI Agent Knowledge Base System",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "message": "NodAgent is running"
    }


app.include_router(api_router)