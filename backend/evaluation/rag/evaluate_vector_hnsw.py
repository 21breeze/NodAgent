import csv
import json
import os
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[2]

load_dotenv(ROOT_DIR / ".env")

DATASET_DIR = BASE_DIR / "datasets" / "scifact"

QRELS_PATH = (
    DATASET_DIR
    / "qrels"
    / "test.tsv"
)

DOCUMENT_EMBEDDINGS_PATH = (
    BASE_DIR
    / "cache"
    / "scifact_document_embeddings.jsonl"
)

QUERY_EMBEDDINGS_PATH = (
    BASE_DIR
    / "cache"
    / "scifact_query_embeddings.jsonl"
)


# ============================================================
# HNSW Config
# 与 NodAgent 正式项目保持一致
# ============================================================

TOP_K = 5

EMBEDDING_DIMENSION = 1024

HNSW_M = 16

HNSW_EF_CONSTRUCTION = 64

HNSW_EF_SEARCH = 100


# ============================================================
# Load qrels
# ============================================================

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
# Load document embeddings
# ============================================================

def load_document_embeddings():

    documents = []

    with DOCUMENT_EMBEDDINGS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            item = json.loads(line)

            documents.append(
                (
                    str(
                        item[
                            "document_id"
                        ]
                    ),
                    np.asarray(
                        item[
                            "embedding"
                        ],
                        dtype=np.float32,
                    ),
                )
            )

    return documents


# ============================================================
# Load query embeddings
# ============================================================

def load_query_embeddings():

    embeddings = {}

    with QUERY_EMBEDDINGS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            item = json.loads(line)

            embeddings[
                str(
                    item["query_id"]
                )
            ] = np.asarray(
                item["embedding"],
                dtype=np.float32,
            )

    return embeddings


# ============================================================
# PostgreSQL connection
# ============================================================

def create_connection():

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

    # 脚本从 Mac Host 运行
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

    register_vector(
        connection
    )

    return connection


# ============================================================
# 创建 SciFact 临时向量表
# ============================================================

def prepare_hnsw_table(
    connection,
    documents,
):

    print(
        "\n创建 SciFact "
        "临时 Vector Table..."
    )

    with connection.cursor() as cursor:

        cursor.execute(
            """
            DROP TABLE IF EXISTS
            scifact_eval_vectors
            """
        )

        cursor.execute(
            f"""
            CREATE TEMP TABLE
            scifact_eval_vectors (
                document_id TEXT PRIMARY KEY,
                embedding vector(
                    {EMBEDDING_DIMENSION}
                )
            )
            """
        )

        print(
            f"正在写入 "
            f"{len(documents)} "
            f"个 Document Embedding..."
        )

        cursor.executemany(
            """
            INSERT INTO
            scifact_eval_vectors (
                document_id,
                embedding
            )
            VALUES (%s, %s)
            """,
            documents,
        )

        print(
            "Document Embedding "
            "写入完成"
        )

        print(
            "\n正在创建 HNSW Index..."
        )

        cursor.execute(
            f"""
            CREATE INDEX
            scifact_eval_embedding_hnsw
            ON scifact_eval_vectors
            USING hnsw (
                embedding
                vector_cosine_ops
            )
            WITH (
                m = {HNSW_M},
                ef_construction =
                {HNSW_EF_CONSTRUCTION}
            )
            """
        )

        cursor.execute(
            """
            ANALYZE
            scifact_eval_vectors
            """
        )

        # 与 NodAgent 当前配置保持一致
        cursor.execute(
            f"""
            SET hnsw.ef_search =
            {HNSW_EF_SEARCH}
            """
        )

        # SciFact 只有约 5K 文档。
        # PostgreSQL 有时会认为 Seq Scan 更便宜。
        # 为了明确测试 HNSW ANN，
        # 这里强制关闭 Seq Scan。
        cursor.execute(
            """
            SET enable_seqscan = off
            """
        )

    connection.commit()

    print(
        "HNSW Index 创建完成"
    )


# ============================================================
# HNSW Search
# ============================================================

def hnsw_search(
    connection,
    query_embedding,
    top_k,
):

    with connection.cursor() as cursor:

        cursor.execute(
            """
            SELECT
                document_id,
                embedding <=> %s
                    AS distance

            FROM
                scifact_eval_vectors

            ORDER BY
                embedding <=> %s

            LIMIT %s
            """,
            (
                query_embedding,
                query_embedding,
                top_k,
            ),
        )

        rows = cursor.fetchall()

    return [
        str(row[0])
        for row in rows
    ]


# ============================================================
# 验证 PostgreSQL 是否真的走 HNSW
# ============================================================

def show_query_plan(
    connection,
    query_embedding,
):

    with connection.cursor() as cursor:

        cursor.execute(
            """
            EXPLAIN (
                COSTS OFF
            )

            SELECT
                document_id

            FROM
                scifact_eval_vectors

            ORDER BY
                embedding <=> %s

            LIMIT 5
            """,
            (
                query_embedding,
            ),
        )

        rows = cursor.fetchall()

    print("\n")
    print("=" * 80)
    print("PostgreSQL Query Plan")
    print("=" * 80)

    for row in rows:
        print(
            row[0]
        )


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

    return (
        hit,
        recall,
        reciprocal_rank,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print(
        "SciFact Vector-Only "
        "HNSW Evaluation"
    )
    print("=" * 80)

    qrels = load_qrels()

    documents = (
        load_document_embeddings()
    )

    query_embeddings = (
        load_query_embeddings()
    )

    print(
        f"\nDocuments："
        f"{len(documents)}"
    )

    print(
        f"Test Queries："
        f"{len(qrels)}"
    )

    print(
        f"Top-K："
        f"{TOP_K}"
    )

    print(
        "\nHNSW Config:"
    )

    print(
        f"m = {HNSW_M}"
    )

    print(
        "ef_construction = "
        f"{HNSW_EF_CONSTRUCTION}"
    )

    print(
        "ef_search = "
        f"{HNSW_EF_SEARCH}"
    )

    connection = (
        create_connection()
    )

    try:

        prepare_hnsw_table(
            connection,
            documents,
        )

        # ----------------------------------------------------
        # 用第一个 Query 验证执行计划
        # ----------------------------------------------------

        first_query_id = next(
            iter(qrels)
        )

        show_query_plan(
            connection,
            query_embeddings[
                first_query_id
            ],
        )

        print("\n")
        print("=" * 80)
        print("开始 HNSW Retrieval")
        print("=" * 80)

        hit_total = 0.0
        recall_total = 0.0
        rr_total = 0.0

        total_queries = len(
            qrels
        )

        for index, (
            query_id,
            relevant_ids,
        ) in enumerate(
            qrels.items(),
            start=1,
        ):

            retrieved_ids = (
                hnsw_search(
                    connection,
                    query_embeddings[
                        query_id
                    ],
                    TOP_K,
                )
            )

            (
                hit,
                recall,
                rr,
            ) = calculate_metrics(
                retrieved_ids,
                relevant_ids,
            )

            hit_total += hit
            recall_total += recall
            rr_total += rr

            if (
                index % 25 == 0
                or index
                == total_queries
            ):

                print(
                    "评测进度："
                    f"{index}"
                    " / "
                    f"{total_queries}"
                )

        hit_at_5 = (
            hit_total
            / total_queries
        )

        recall_at_5 = (
            recall_total
            / total_queries
        )

        mrr_at_5 = (
            rr_total
            / total_queries
        )

        print("\n")
        print("=" * 80)
        print("Final HNSW Results")
        print("=" * 80)

        print(
            f"Hit@5    : "
            f"{hit_at_5:.4f}"
        )

        print(
            f"Recall@5 : "
            f"{recall_at_5:.4f}"
        )

        print(
            f"MRR@5    : "
            f"{mrr_at_5:.4f}"
        )

    finally:

        connection.close()


if __name__ == "__main__":
    main()
