import csv
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[2]

load_dotenv(ROOT_DIR / ".env")


DATASET_DIR = BASE_DIR / "datasets" / "scifact"

CORPUS_PATH = DATASET_DIR / "corpus.jsonl"
QUERIES_PATH = DATASET_DIR / "queries.jsonl"
QRELS_PATH = DATASET_DIR / "qrels" / "test.tsv"

CACHE_DIR = BASE_DIR / "cache"

DOCUMENT_EMBEDDINGS_PATH = (
    CACHE_DIR
    / "scifact_document_embeddings.jsonl"
)

QUERY_EMBEDDINGS_PATH = (
    CACHE_DIR
    / "scifact_query_embeddings.jsonl"
)

REPORT_DIR = BASE_DIR / "reports"


# ============================================================
# Evaluation config
# ============================================================

TOP_K = 5

CANDIDATE_MULTIPLIER = 5

CANDIDATE_K = (
    TOP_K
    * CANDIDATE_MULTIPLIER
)

RRF_K = 60


# ============================================================
# Import project's actual embedding model
# ============================================================

from backend.app.services.embedding_service import (
    embedding_model,
)

from backend.app.core.config import (
    EMBEDDING_MODEL_NAME,
)


# ============================================================
# Load SciFact
# ============================================================

def load_corpus():
    corpus = {}

    with CORPUS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            document_id = str(
                item["_id"]
            )

            title = item.get(
                "title",
                "",
            )

            text = item.get(
                "text",
                "",
            )

            content = (
                title
                + "\n"
                + text
            ).strip()

            corpus[document_id] = {
                "title": title,
                "text": text,
                "content": content,
            }

    return corpus


def load_queries():
    queries = {}

    with QUERIES_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            queries[
                str(item["_id"])
            ] = item["text"]

    return queries


def load_qrels():
    qrels = {}

    with QRELS_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:
            query_id = str(
                row["query-id"]
            )

            document_id = str(
                row["corpus-id"]
            )

            score = int(
                row["score"]
            )

            if score <= 0:
                continue

            qrels.setdefault(
                query_id,
                set(),
            ).add(
                document_id
            )

    return qrels


# ============================================================
# Document Embeddings
# ============================================================

def load_document_embeddings(
    corpus,
):
    embedding_map = {}

    with DOCUMENT_EMBEDDINGS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            embedding_map[
                str(item["document_id"])
            ] = item["embedding"]

    print(
        f"已读取 Document Embedding："
        f"{len(embedding_map)}"
    )

    missing = [
        document_id
        for document_id in corpus
        if document_id
        not in embedding_map
    ]

    if missing:
        raise RuntimeError(
            "存在未生成 Embedding 的文档，"
            f"数量：{len(missing)}"
        )

    document_ids = list(
        corpus.keys()
    )

    matrix = np.asarray(
        [
            embedding_map[
                document_id
            ]
            for document_id
            in document_ids
        ],
        dtype=np.float32,
    )

    # cosine similarity 需要归一化
    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    norms[norms == 0] = 1.0

    matrix = matrix / norms

    return (
        document_ids,
        matrix,
    )


# ============================================================
# Query Embeddings
# ============================================================

def load_query_embedding_cache():
    result = {}

    if not QUERY_EMBEDDINGS_PATH.exists():
        return result

    with QUERY_EMBEDDINGS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            result[
                str(item["query_id"])
            ] = item["embedding"]

    return result


def prepare_query_embeddings(
    queries,
    qrels,
):
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache = (
        load_query_embedding_cache()
    )

    print(
        f"已有 Query Embedding："
        f"{len(cache)}"
    )

    test_query_ids = list(
        qrels.keys()
    )

    missing_query_ids = [
        query_id
        for query_id
        in test_query_ids
        if query_id not in cache
    ]

    if missing_query_ids:
        print(
            f"需要生成 Query Embedding："
            f"{len(missing_query_ids)}"
        )

    with QUERY_EMBEDDINGS_PATH.open(
        "a",
        encoding="utf-8",
    ) as output_file:

        for index, query_id in enumerate(
            missing_query_ids,
            start=1,
        ):
            query_text = (
                queries[
                    query_id
                ]
            )

            embedding = (
                embedding_model
                .embed_query(
                    query_text
                )
            )

            cache[
                query_id
            ] = embedding

            output_file.write(
                json.dumps(
                    {
                        "query_id":
                            query_id,
                        "embedding":
                            embedding,
                    }
                )
                + "\n"
            )

            output_file.flush()

            if (
                index % 25 == 0
                or index
                == len(
                    missing_query_ids
                )
            ):
                print(
                    "Query Embedding："
                    f"{index}"
                    " / "
                    f"{len(missing_query_ids)}"
                )

    return cache


# ============================================================
# Vector Retrieval
# ============================================================

def vector_search(
    query_embedding,
    document_ids,
    document_matrix,
    top_k,
):
    query_vector = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    norm = np.linalg.norm(
        query_vector
    )

    if norm != 0:
        query_vector = (
            query_vector
            / norm
        )

    scores = (
        document_matrix
        @ query_vector
    )

    ranking = np.argsort(
        -scores
    )[:top_k]

    results = []

    for index in ranking:
        results.append(
            {
                "document_id":
                    document_ids[
                        int(index)
                    ],
                "score":
                    float(
                        scores[index]
                    ),
            }
        )

    return results


# ============================================================
# PostgreSQL Keyword Retrieval
# ============================================================

def create_keyword_connection(
    corpus,
):
    host = os.getenv(
        "POSTGRES_HOST",
        "localhost",
    )

    port = int(
        os.getenv(
            "POSTGRES_PORT",
            "15432",
        )
    )

    # 本脚本运行在 Mac Host。
    # Docker 内部 postgres:5432
    # Host 访问 localhost:15432
    if host == "postgres":
        host = "localhost"
        port = 15432

    connection = psycopg.connect(
        host=host,
        port=port,
        dbname=os.environ[
            "POSTGRES_DB"
        ],
        user=os.environ[
            "POSTGRES_USER"
        ],
        password=os.environ[
            "POSTGRES_PASSWORD"
        ],
    )

    connection.autocommit = True

    with connection.cursor() as cursor:

        cursor.execute(
            """
            DROP TABLE IF EXISTS
            scifact_eval_docs
            """
        )

        cursor.execute(
            """
            CREATE TEMP TABLE
            scifact_eval_docs (
                document_id TEXT PRIMARY KEY,
                content TEXT NOT NULL
            )
            ON COMMIT PRESERVE ROWS
            """
        )

        rows = [
            (
                document_id,
                document["content"],
            )
            for document_id, document
            in corpus.items()
        ]

        cursor.executemany(
            """
            INSERT INTO
            scifact_eval_docs (
                document_id,
                content
            )
            VALUES (%s, %s)
            """,
            rows,
        )

        cursor.execute(
            """
            CREATE INDEX
            scifact_eval_fts_idx
            ON scifact_eval_docs
            USING GIN (
                to_tsvector(
                    'simple',
                    content
                )
            )
            """
        )

        cursor.execute(
            """
            ANALYZE
            scifact_eval_docs
            """
        )

    print(
        "PostgreSQL Keyword "
        "临时索引创建完成"
    )

    return connection


def keyword_search(
    connection,
    query,
    top_k,
):
    with connection.cursor() as cursor:

        cursor.execute(
            """
            SELECT
                document_id,
                ts_rank_cd(
                    to_tsvector(
                        'simple',
                        content
                    ),
                    plainto_tsquery(
                        'simple',
                        %s
                    )
                ) AS keyword_score
            FROM scifact_eval_docs
            WHERE
                to_tsvector(
                    'simple',
                    content
                )
                @@
                plainto_tsquery(
                    'simple',
                    %s
                )
            ORDER BY
                keyword_score DESC,
                document_id ASC
            LIMIT %s
            """,
            (
                query,
                query,
                top_k,
            ),
        )

        rows = cursor.fetchall()

    return [
        {
            "document_id":
                str(row[0]),
            "score":
                float(row[1]),
        }
        for row in rows
    ]


# ============================================================
# RRF
# ============================================================

def reciprocal_rank_fusion(
    vector_results,
    keyword_results,
    top_k,
):
    fused = {}

    for rank, result in enumerate(
        vector_results,
        start=1,
    ):
        document_id = (
            result["document_id"]
        )

        fused[
            document_id
        ] = {
            "document_id":
                document_id,
            "vector_rank":
                rank,
            "keyword_rank":
                None,
            "rrf_score":
                (
                    1.0
                    / (
                        RRF_K
                        + rank
                    )
                ),
        }

    for rank, result in enumerate(
        keyword_results,
        start=1,
    ):
        document_id = (
            result["document_id"]
        )

        if document_id not in fused:
            fused[
                document_id
            ] = {
                "document_id":
                    document_id,
                "vector_rank":
                    None,
                "keyword_rank":
                    rank,
                "rrf_score":
                    (
                        1.0
                        / (
                            RRF_K
                            + rank
                        )
                    ),
            }

        else:
            fused[
                document_id
            ][
                "keyword_rank"
            ] = rank

            fused[
                document_id
            ][
                "rrf_score"
            ] += (
                1.0
                / (
                    RRF_K
                    + rank
                )
            )

    ordered = sorted(
        fused.values(),
        key=lambda item: (
            -item[
                "rrf_score"
            ],
            (
                item[
                    "vector_rank"
                ]
                if item[
                    "vector_rank"
                ]
                is not None
                else 10**9
            ),
            (
                item[
                    "keyword_rank"
                ]
                if item[
                    "keyword_rank"
                ]
                is not None
                else 10**9
            ),
            item[
                "document_id"
            ],
        ),
    )

    return ordered[:top_k]


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    retrieved_ids,
    relevant_ids,
):
    relevant_ids = set(
        relevant_ids
    )

    retrieved_ids = list(
        retrieved_ids
    )

    matched = (
        set(retrieved_ids)
        & relevant_ids
    )

    hit = (
        1.0
        if matched
        else 0.0
    )

    recall = (
        len(matched)
        / len(relevant_ids)
        if relevant_ids
        else 0.0
    )

    reciprocal_rank = 0.0

    for rank, document_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if document_id in relevant_ids:
            reciprocal_rank = (
                1.0
                / rank
            )
            break

    return {
        "hit": hit,
        "recall": recall,
        "rr": reciprocal_rank,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("NodAgent SciFact RAG Retrieval Evaluation")
    print("=" * 80)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels()

    print(
        f"\nCorpus：{len(corpus)}"
    )

    print(
        f"Test Queries：{len(qrels)}"
    )

    print(
        f"TOP_K：{TOP_K}"
    )

    print(
        f"Hybrid Candidate K："
        f"{CANDIDATE_K}"
    )

    print(
        f"RRF_K：{RRF_K}"
    )

    print(
        f"Embedding："
        f"{EMBEDDING_MODEL_NAME}"
    )

    print("\n加载 Document Embedding...")

    (
        document_ids,
        document_matrix,
    ) = load_document_embeddings(
        corpus
    )

    print(
        "Document Matrix："
        f"{document_matrix.shape}"
    )

    print("\n准备 Query Embedding...")

    query_embeddings = (
        prepare_query_embeddings(
            queries,
            qrels,
        )
    )

    print(
        "\n准备 PostgreSQL "
        "Keyword Index..."
    )

    keyword_connection = (
        create_keyword_connection(
            corpus
        )
    )

    metric_totals = {
        "vector": {
            "hit": 0.0,
            "recall": 0.0,
            "rr": 0.0,
        },
        "keyword": {
            "hit": 0.0,
            "recall": 0.0,
            "rr": 0.0,
        },
        "hybrid": {
            "hit": 0.0,
            "recall": 0.0,
            "rr": 0.0,
        },
    }

    details = []

    test_query_ids = list(
        qrels.keys()
    )

    total_queries = len(
        test_query_ids
    )

    try:
        for index, query_id in enumerate(
            test_query_ids,
            start=1,
        ):
            query = (
                queries[
                    query_id
                ]
            )

            relevant_ids = (
                qrels[
                    query_id
                ]
            )

            # --------------------------
            # Vector baseline
            # --------------------------

            vector_top5 = (
                vector_search(
                    query_embeddings[
                        query_id
                    ],
                    document_ids,
                    document_matrix,
                    TOP_K,
                )
            )

            # Hybrid 需要 Top-25
            vector_candidates = (
                vector_search(
                    query_embeddings[
                        query_id
                    ],
                    document_ids,
                    document_matrix,
                    CANDIDATE_K,
                )
            )

            # --------------------------
            # Keyword baseline
            # --------------------------

            keyword_top5 = (
                keyword_search(
                    keyword_connection,
                    query,
                    TOP_K,
                )
            )

            keyword_candidates = (
                keyword_search(
                    keyword_connection,
                    query,
                    CANDIDATE_K,
                )
            )

            # --------------------------
            # Hybrid + RRF
            # --------------------------

            hybrid_top5 = (
                reciprocal_rank_fusion(
                    vector_candidates,
                    keyword_candidates,
                    TOP_K,
                )
            )

            vector_ids = [
                item["document_id"]
                for item
                in vector_top5
            ]

            keyword_ids = [
                item["document_id"]
                for item
                in keyword_top5
            ]

            hybrid_ids = [
                item["document_id"]
                for item
                in hybrid_top5
            ]

            vector_metrics = (
                calculate_metrics(
                    vector_ids,
                    relevant_ids,
                )
            )

            keyword_metrics = (
                calculate_metrics(
                    keyword_ids,
                    relevant_ids,
                )
            )

            hybrid_metrics = (
                calculate_metrics(
                    hybrid_ids,
                    relevant_ids,
                )
            )

            for method, metrics in [
                (
                    "vector",
                    vector_metrics,
                ),
                (
                    "keyword",
                    keyword_metrics,
                ),
                (
                    "hybrid",
                    hybrid_metrics,
                ),
            ]:
                for metric_name in [
                    "hit",
                    "recall",
                    "rr",
                ]:
                    metric_totals[
                        method
                    ][
                        metric_name
                    ] += metrics[
                        metric_name
                    ]

            details.append(
                {
                    "query_id":
                        query_id,
                    "query":
                        query,
                    "relevant_document_ids":
                        sorted(
                            relevant_ids
                        ),
                    "vector_top5":
                        vector_ids,
                    "keyword_top5":
                        keyword_ids,
                    "hybrid_top5":
                        hybrid_ids,
                    "vector_metrics":
                        vector_metrics,
                    "keyword_metrics":
                        keyword_metrics,
                    "hybrid_metrics":
                        hybrid_metrics,
                }
            )

            if (
                index % 25 == 0
                or index
                == total_queries
            ):
                print(
                    f"评测进度："
                    f"{index}"
                    f" / "
                    f"{total_queries}"
                )

    finally:
        keyword_connection.close()

    # ========================================================
    # Summary
    # ========================================================

    summary = {}

    for method in [
        "vector",
        "keyword",
        "hybrid",
    ]:
        summary[
            method
        ] = {
            "Hit@5":
                metric_totals[
                    method
                ][
                    "hit"
                ]
                / total_queries,

            "Recall@5":
                metric_totals[
                    method
                ][
                    "recall"
                ]
                / total_queries,

            "MRR@5":
                metric_totals[
                    method
                ][
                    "rr"
                ]
                / total_queries,
        }

    print("\n")
    print("=" * 80)
    print("Final Results")
    print("=" * 80)

    print(
        f"\n{'Method':<20}"
        f"{'Hit@5':>12}"
        f"{'Recall@5':>12}"
        f"{'MRR@5':>12}"
    )

    print("-" * 56)

    display_names = {
        "vector":
            "Vector Only",
        "keyword":
            "Keyword Only",
        "hybrid":
            "Hybrid + RRF",
    }

    for method in [
        "vector",
        "keyword",
        "hybrid",
    ]:
        values = summary[
            method
        ]

        print(
            f"{display_names[method]:<20}"
            f"{values['Hit@5']:>12.4f}"
            f"{values['Recall@5']:>12.4f}"
            f"{values['MRR@5']:>12.4f}"
        )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    details_path = (
        REPORT_DIR
        / (
            "scifact_retrieval_"
            f"results_{timestamp}.jsonl"
        )
    )

    summary_path = (
        REPORT_DIR
        / (
            "scifact_retrieval_"
            f"summary_{timestamp}.json"
        )
    )

    with details_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for item in details:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "dataset":
                    "BEIR SciFact",
                "corpus_size":
                    len(corpus),
                "test_queries":
                    total_queries,
                "top_k":
                    TOP_K,
                "candidate_k":
                    CANDIDATE_K,
                "rrf_k":
                    RRF_K,
                "embedding_model":
                    EMBEDDING_MODEL_NAME,
                "results":
                    summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\n详细结果："
        f"{details_path}"
    )

    print(
        f"汇总结果："
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()
