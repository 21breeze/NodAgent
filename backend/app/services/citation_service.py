from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from backend.app.db.database import (
    SessionLocal,
)
from backend.app.models.document import (
    Document,
)
from backend.app.schemas.search import (
    SearchResult,
)


def get_page_number(
    metadata: Dict[str, Any],
) -> Optional[int]:
    page_label = metadata.get(
        "page_label"
    )

    if page_label is not None:
        try:
            return int(page_label)
        except (
            TypeError,
            ValueError,
        ):
            pass

    page = metadata.get(
        "page"
    )

    if isinstance(page, int):
        return page + 1

    return None


def get_document_filenames(
    document_ids: List[int],
) -> Dict[int, str]:
    if not document_ids:
        return {}

    unique_ids = list(
        dict.fromkeys(
            document_ids
        )
    )

    with SessionLocal() as db:
        rows = (
            db.query(
                Document.id,
                Document.filename,
            )
            .filter(
                Document.id.in_(
                    unique_ids
                )
            )
            .all()
        )

    return {
        int(document_id): filename
        for document_id, filename in rows
    }


def build_citation_sources(
    results: List[SearchResult],
) -> List[Dict[str, Any]]:
    filenames = (
        get_document_filenames(
            [
                result.document_id
                for result in results
            ]
        )
    )

    sources: List[
        Dict[str, Any]
    ] = []

    for source_number, result in enumerate(
        results,
        start=1,
    ):
        metadata = dict(
            result.metadata_json
            or {}
        )

        filename = filenames.get(
            result.document_id,
            f"document_{result.document_id}",
        )

        page_number = get_page_number(
            metadata
        )

        sources.append(
            {
                "source_number": (
                    source_number
                ),
                "citation": (
                    f"[{source_number}]"
                ),
                "chunk_id": (
                    result.chunk_id
                ),
                "document_id": (
                    result.document_id
                ),
                "filename": filename,
                "chunk_index": (
                    result.chunk_index
                ),
                "page_number": (
                    page_number
                ),
                "content": (
                    result.content
                ),
                "metadata_json": (
                    metadata
                ),
                "distance": (
                    result.distance
                ),
                "similarity": (
                    result.similarity
                ),
                "keyword_score": (
                    result.keyword_score
                ),
                "vector_rank": (
                    result.vector_rank
                ),
                "keyword_rank": (
                    result.keyword_rank
                ),
                "rrf_score": (
                    result.rrf_score
                ),
            }
        )

    return sources


def build_citation_context(
    sources: List[Dict[str, Any]],
) -> str:
    context_parts: List[str] = []

    for source in sources:
        citation = source[
            "citation"
        ]

        filename = source[
            "filename"
        ]

        page_number = source.get(
            "page_number"
        )

        if page_number is None:
            source_title = (
                f"{citation} "
                f"{filename}"
            )
        else:
            source_title = (
                f"{citation} "
                f"{filename} "
                f"(page {page_number})"
            )

        context_parts.append(
            (
                f"{source_title}\n"
                f"{source['content']}"
            )
        )

    return "\n\n".join(
        context_parts
    )
