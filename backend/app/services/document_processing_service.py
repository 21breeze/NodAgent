from typing import (
    Any,
    Dict,
    List,
    TypedDict,
)

from backend.app.services.document_parser import (
    parse_and_split_document,
)
from backend.app.services.embedding_service import (
    embedding_model,
)


class PreparedChunk(TypedDict):
    chunk_index: int

    content: str

    metadata_json: Dict[
        str,
        Any,
    ]

    embedding: List[float]


def prepare_document_chunks(
    file_path: str,
) -> List[PreparedChunk]:
    chunks = (
        parse_and_split_document(
            file_path=file_path
        )
    )

    if not chunks:
        raise ValueError(
            "Document produced no chunks"
        )

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = (
        embedding_model.embed_documents(
            texts
        )
    )

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "Embedding count does not "
            "match chunk count"
        )

    prepared_chunks: List[
        PreparedChunk
    ] = []

    for index, (
        chunk,
        embedding,
    ) in enumerate(
        zip(
            chunks,
            embeddings,
        )
    ):
        prepared_chunks.append(
            PreparedChunk(
                chunk_index=index,
                content=(
                    chunk.page_content
                ),
                metadata_json=dict(
                    chunk.metadata
                ),
                embedding=embedding,
            )
        )

    return prepared_chunks