from typing import List

from sqlalchemy.orm import Session

from backend.app.models.workspace import (
    Workspace,
)
from backend.app.schemas.search import (
    SearchResult,
)
from backend.app.services.vector_store_service import (
    vector_store,
)


def search_workspace(
    db: Session,
    workspace_id: int,
    query: str,
    top_k: int = 5,
) -> List[SearchResult]:
    workspace = (
        db.query(Workspace)
        .filter(
            Workspace.id == workspace_id
        )
        .first()
    )

    if workspace is None:
        return []

    documents_with_scores = (
        vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter={
                "workspace_id": workspace_id,
            },
        )
    )

    results = []

    for document, distance in (
        documents_with_scores
    ):
        metadata = dict(
            document.metadata
        )

        document_id = metadata.pop(
            "document_id"
        )

        chunk_index = metadata.pop(
            "chunk_index"
        )

        metadata.pop(
            "workspace_id",
            None,
        )

        similarity = (
            1.0 - float(distance)
        )

        result = SearchResult(
            chunk_id=int(document.id),
            document_id=int(document_id),
            chunk_index=int(chunk_index),
            content=document.page_content,
            metadata_json=metadata,
            distance=float(distance),
            similarity=similarity,
        )

        results.append(
            result
        )

    return results