from contextlib import (
    asynccontextmanager,
)

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import (
    AsyncPostgresSaver,
)

from backend.app.api.router import (
    api_router,
)
from backend.app.core.config import (
    CHECKPOINT_DATABASE_URL,
)
from backend.app.db.database import (
    Base,
    engine,
)
from backend.app.graphs.main_graph import (
    build_main_graph,
)
from backend.app.models import (
    ChatMessage,
    ChatThread,
    Document,
    DocumentChunk,
    Workspace,
)


Base.metadata.create_all(
    bind=engine,
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    async with (
        AsyncPostgresSaver
        .from_conn_string(
            CHECKPOINT_DATABASE_URL
        )
    ) as checkpointer:
        await checkpointer.setup()

        app.state.checkpointer = (
            checkpointer
        )

        app.state.agent_graph = (
            build_main_graph(
                checkpointer=(
                    checkpointer
                )
            )
        )

        yield


app = FastAPI(
    title="NodAgent",
    description=(
        "AI Agent Knowledge "
        "Base System"
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message": (
            "NodAgent is running"
        )
    }


app.include_router(
    api_router
)