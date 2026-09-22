from fastapi import (
    APIRouter,
    Depends,
)
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.schemas.search import (
    SearchRequest,
    SearchResponse,
)
from backend.app.services import (
    retrieval_service,
)


router = APIRouter(
    prefix="/workspaces/{workspace_id}/search",
    tags=["search"],
)


@router.post(
    "",
    response_model=SearchResponse,
)
def search_workspace(
    workspace_id: int,
    request: SearchRequest,
    db: Session = Depends(get_db),
):
    results = (
        retrieval_service.search_workspace(
            db=db,
            workspace_id=workspace_id,
            query=request.query,
            top_k=request.top_k,
        )
    )

    return SearchResponse(
        query=request.query,
        top_k=request.top_k,
        results=results,
    )