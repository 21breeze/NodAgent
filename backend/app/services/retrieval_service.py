from typing import (
    Dict,
    List,
    Sequence,
    Tuple,
)

from langchain_core.documents import (
    Document as LangChainDocument,
)
from sqlalchemy import func
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
from backend.app.models.document_chunk import (
    DocumentChunk,
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


RRF_K = 60
CANDIDATE_MULTIPLIER = 5


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


def get_candidate_k(
    top_k: int,
) -> int:
    return max(
        top_k,
        top_k
        * CANDIDATE_MULTIPLIER,
    )


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
            keyword_score=None,
            vector_rank=None,
            keyword_rank=None,
            rrf_score=0.0,
        )

        results.append(
            result
        )

    return results


def search_keyword_candidates(
    db: Session,
    workspace_id: int,
    document_ids: List[int],
    query: str,
    candidate_k: int,
) -> List[
    Tuple[
        DocumentChunk,
        float,
    ]
]:
    normalized_query = query.strip()

    if not normalized_query:
        return []

    search_vector = func.to_tsvector(
        "simple",
        DocumentChunk.content,
    )

    search_query = func.plainto_tsquery(
        "simple",
        normalized_query,
    )

    keyword_score = func.ts_rank_cd(
        search_vector,
        search_query,
    ).label(
        "keyword_score"
    )

    rows = (
        db.query(
            DocumentChunk,
            keyword_score,
        )
        .filter(
            DocumentChunk.workspace_id
            == workspace_id,
            DocumentChunk.document_id.in_(
                document_ids
            ),
            search_vector.op("@@")(
                search_query
            ),
        )
        .order_by(
            keyword_score.desc(),
            DocumentChunk.id.asc(),
        )
        .limit(candidate_k)
        .all()
    )

    return [
        (
            chunk,
            float(score),
        )
        for chunk, score in rows
    ]


def fuse_with_rrf(
    vector_results: List[SearchResult],
    keyword_results: List[
        Tuple[
            DocumentChunk,
            float,
        ]
    ],
    top_k: int,
) -> List[SearchResult]:
    fused: Dict[
        int,
        Dict[str, object],
    ] = {}

    for rank, result in enumerate(
        vector_results,
        start=1,
    ):
        fused[result.chunk_id] = {
            "chunk_id": result.chunk_id,
            "document_id": (
                result.document_id
            ),
            "chunk_index": (
                result.chunk_index
            ),
            "content": result.content,
            "metadata_json": dict(
                result.metadata_json
            ),
            "distance": (
                result.distance
            ),
            "similarity": (
                result.similarity
            ),
            "keyword_score": None,
            "vector_rank": rank,
            "keyword_rank": None,
            "rrf_score": (
                1.0
                / (RRF_K + rank)
            ),
        }

    for rank, (
        chunk,
        keyword_score,
    ) in enumerate(
        keyword_results,
        start=1,
    ):
        chunk_id = int(
            chunk.id
        )

        if chunk_id not in fused:
            fused[chunk_id] = {
                "chunk_id": chunk_id,
                "document_id": int(
                    chunk.document_id
                ),
                "chunk_index": int(
                    chunk.chunk_index
                ),
                "content": chunk.content,
                "metadata_json": dict(
                    chunk.metadata_json
                    or {}
                ),
                "distance": None,
                "similarity": None,
                "keyword_score": (
                    keyword_score
                ),
                "vector_rank": None,
                "keyword_rank": rank,
                "rrf_score": (
                    1.0
                    / (RRF_K + rank)
                ),
            }

            continue

        item = fused[chunk_id]

        item["keyword_score"] = (
            keyword_score
        )
        item["keyword_rank"] = rank
        item["rrf_score"] = (
            float(item["rrf_score"])
            + (
                1.0
                / (RRF_K + rank)
            )
        )

    ordered_items = sorted(
        fused.values(),
        key=lambda item: (
            -float(item["rrf_score"]),
            (
                int(item["vector_rank"])
                if item["vector_rank"]
                is not None
                else 10**9
            ),
            (
                int(item["keyword_rank"])
                if item["keyword_rank"]
                is not None
                else 10**9
            ),
            int(item["chunk_id"]),
        ),
    )

    return [
        SearchResult(**item)
        for item in ordered_items[
            :top_k
        ]
    ]


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

    candidate_k = get_candidate_k(
        top_k
    )

    vector_filter = (
        build_vector_filter(
            workspace_id=workspace_id,
            document_ids=(
                completed_document_ids
            ),
        )
    )

    vector_documents_with_scores = (
        vector_store
        .similarity_search_with_score(
            query=query,
            k=candidate_k,
            filter=vector_filter,
        )
    )

    vector_results = (
        convert_vector_results(
            vector_documents_with_scores
        )
    )

    keyword_results = (
        search_keyword_candidates(
            db=db,
            workspace_id=workspace_id,
            document_ids=(
                completed_document_ids
            ),
            query=query,
            candidate_k=candidate_k,
        )
    )

    return fuse_with_rrf(
        vector_results=vector_results,
        keyword_results=keyword_results,
        top_k=top_k,
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

        candidate_k = get_candidate_k(
            top_k
        )

        keyword_results = (
            search_keyword_candidates(
                db=db,
                workspace_id=(
                    workspace_id
                ),
                document_ids=(
                    completed_document_ids
                ),
                query=query,
                candidate_k=(
                    candidate_k
                ),
            )
        )

    vector_filter = (
        build_vector_filter(
            workspace_id=workspace_id,
            document_ids=(
                completed_document_ids
            ),
        )
    )

    vector_documents_with_scores = (
        await vector_store
        .asimilarity_search_with_score(
            query=query,
            k=candidate_k,
            filter=vector_filter,
        )
    )

    vector_results = (
        convert_vector_results(
            vector_documents_with_scores
        )
    )

    return fuse_with_rrf(
        vector_results=vector_results,
        keyword_results=keyword_results,
        top_k=top_k,
    )