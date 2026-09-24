from typing import (
    List,
    Sequence,
    Tuple,
)

from langchain_core.documents import (
    Document as LangChainDocument,
)
from sqlalchemy.orm import Session

from backend.app.core.document_status import (
    DocumentProcessingStatus,
)
from backend.app.db.database import (
    SessionLocal,
)
from backend.app.models.document import (
    Document,
)
from backend.app.models.workspace import (
    Workspace,
)
from backend.app.schemas.search import (
    SearchResult,
)
from backend.app.services.vector_store_service import (
    vector_store,
)


def get_completed_document_ids(
    db: Session,
    workspace_id: int,
) -> List[int]:
    rows = (
        db.query(Document.id)
        .filter(
            Document.workspace_id
            == workspace_id,
            Document.processing_status
            == (
                DocumentProcessingStatus
                .COMPLETED
                .value
            ),
        )
        .all()
    )

    return [
        row[0]
        for row in rows
    ]


def build_vector_filter(
    workspace_id: int,
    document_ids: List[int],
):
    return {
        "$and": [
            {
                "workspace_id": (
                    workspace_id
                )
            },
            {
                "document_id": {
                    "$in": document_ids
                }
            },
        ]
    }


def convert_vector_results(
    documents_with_scores: Sequence[
        Tuple[
            LangChainDocument,
            float,
        ]
    ],
) -> List[SearchResult]:
    results: List[
        SearchResult
    ] = []

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
            1.0
            - float(distance)
        )

        result = SearchResult(
            chunk_id=int(
                document.id
            ),
            document_id=int(
                document_id
            ),
            chunk_index=int(
                chunk_index
            ),
            content=(
                document.page_content
            ),
            metadata_json=metadata,
            distance=float(
                distance
            ),
            similarity=similarity,
        )

        results.append(
            result
        )

    return results


def search_workspace(
    db: Session,
    workspace_id: int,
    query: str,
    top_k: int = 5,
) -> List[SearchResult]:
    workspace = (
        db.query(Workspace)
        .filter(
            Workspace.id
            == workspace_id
        )
        .first()
    )

    if workspace is None:
        return []

    completed_document_ids = (
        get_completed_document_ids(
            db=db,
            workspace_id=workspace_id,
        )
    )

    if not completed_document_ids:
        return []

    vector_filter = (
        build_vector_filter(
            workspace_id=workspace_id,
            document_ids=(
                completed_document_ids
            ),
        )
    )

    documents_with_scores = (
        vector_store
        .similarity_search_with_score(
            query=query,
            k=top_k,
            filter=vector_filter,
        )
    )

    return convert_vector_results(
        documents_with_scores
    )


async def search_workspace_async(
    workspace_id: int,
    query: str,
    top_k: int = 5,
) -> List[SearchResult]:
    with SessionLocal() as db:
        workspace = (
            db.query(Workspace)
            .filter(
                Workspace.id
                == workspace_id
            )
            .first()
        )

        if workspace is None:
            return []

        completed_document_ids = (
            get_completed_document_ids(
                db=db,
                workspace_id=(
                    workspace_id
                ),
            )
        )

    if not completed_document_ids:
        return []

    vector_filter = (
        build_vector_filter(
            workspace_id=workspace_id,
            document_ids=(
                completed_document_ids
            ),
        )
    )

    documents_with_scores = (
        await vector_store
        .asimilarity_search_with_score(
            query=query,
            k=top_k,
            filter=vector_filter,
        )
    )

    return convert_vector_results(
        documents_with_scores
    )