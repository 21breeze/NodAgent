from langchain_postgres.v2.indexes import (
    DistanceStrategy,
    HNSWIndex,
)

from backend.app.services.vector_store_service import (
    vector_store,
)


INDEX_NAME = (
    "ix_document_chunks_embedding_hnsw"
)


def main():
    exists = vector_store.is_valid_index(
        INDEX_NAME
    )

    if exists:
        print(
            f"HNSW index already exists: "
            f"{INDEX_NAME}"
        )

        return

    index = HNSWIndex(
        name=INDEX_NAME,
        m=16,
        ef_construction=64,
        distance_strategy=(
            DistanceStrategy.COSINE_DISTANCE
        ),
    )

    vector_store.apply_vector_index(
        index=index
    )

    print(
        f"HNSW index created: "
        f"{INDEX_NAME}"
    )


if __name__ == "__main__":
    main()