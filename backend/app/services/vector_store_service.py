from langchain_postgres import (
    PGEngine,
    PGVectorStore,
)
from langchain_postgres.v2.indexes import (
    DistanceStrategy,
    HNSWQueryOptions,
)

from backend.app.core.config import DATABASE_URL
from backend.app.services.embedding_service import (
    embedding_model,
)


VECTOR_TABLE_NAME = "document_chunks"


pg_engine = PGEngine.from_connection_string(
    url=DATABASE_URL
)


vector_store = PGVectorStore.create_sync(
    engine=pg_engine,
    table_name=VECTOR_TABLE_NAME,
    embedding_service=embedding_model,

    id_column="id",

    content_column="content",

    embedding_column="embedding",

    metadata_columns=[
        "workspace_id",
        "document_id",
        "chunk_index",
    ],

    metadata_json_column="metadata_json",

    distance_strategy=(
        DistanceStrategy.COSINE_DISTANCE
    ),

    index_query_options=HNSWQueryOptions(
        ef_search=40
    ),
)