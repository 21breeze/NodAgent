import csv
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector


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


TOP_K = 5

EMBEDDING_DIMENSION = 1024

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64
HNSW_EF_SEARCH = 100


def load_test_query_ids():
    query_ids = []

    seen = set()

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

            if query_id not in seen:
                seen.add(query_id)
                query_ids.append(query_id)

    return query_ids


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
                        item["document_id"]
                    ),
                    np.asarray(
                        item["embedding"],
                        dtype=np.float32,
                    ),
                )
            )

    return documents


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
                str(item["query_id"])
            ] = np.asarray(
                item["embedding"],
                dtype=np.float32,
            )

    return embeddings


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

    if host == "postgres":
        host = "localhost"
        port = 15432

    connection = psycopg.connect(
        host=host,
        port=port,
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )

    register_vector(
        connection
    )

    return connection


def prepare_table(
    connection,
    documents,
):
    print(
        f"正在写入 {len(documents)} "
        "个 SciFact Vector..."
    )

    with connection.cursor() as cursor:

        cursor.execute(
            """
            DROP TABLE IF EXISTS
            scifact_latency_vectors
            """
        )

        cursor.execute(
            f"""
            CREATE TEMP TABLE
            scifact_latency_vectors (
                document_id TEXT PRIMARY KEY,
                embedding vector(
                    {EMBEDDING_DIMENSION}
                )
            )
            """
        )

        cursor.executemany(
            """
            INSERT INTO
            scifact_latency_vectors (
                document_id,
                embedding
            )
            VALUES (%s, %s)
            """,
            documents,
        )

        print(
            "正在创建 HNSW Index..."
        )

        cursor.execute(
            f"""
            CREATE INDEX
            scifact_latency_hnsw
            ON scifact_latency_vectors
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
            scifact_latency_vectors
            """
        )

    connection.commit()

    print(
        "Vector Table + HNSW "
        "准备完成"
    )


def exact_search(
    connection,
    query_embedding,
):
    with connection.cursor() as cursor:

        # 强制不用索引
        cursor.execute(
            """
            SET LOCAL enable_indexscan = off;
            SET LOCAL enable_indexonlyscan = off;
            SET LOCAL enable_bitmapscan = off;
            SET LOCAL enable_seqscan = on;
            """
        )

        start = time.perf_counter()

        cursor.execute(
            """
            SELECT
                document_id
            FROM
                scifact_latency_vectors
            ORDER BY
                embedding <=> %s
            LIMIT %s
            """,
            (
                query_embedding,
                TOP_K,
            ),
        )

        rows = cursor.fetchall()

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000

    return (
        [
            str(row[0])
            for row in rows
        ],
        elapsed_ms,
    )


def hnsw_search(
    connection,
    query_embedding,
):
    with connection.cursor() as cursor:

        cursor.execute(
            f"""
            SET LOCAL hnsw.ef_search =
            {HNSW_EF_SEARCH};

            SET LOCAL enable_seqscan = off;
            SET LOCAL enable_indexscan = on;
            """
        )

        start = time.perf_counter()

        cursor.execute(
            """
            SELECT
                document_id
            FROM
                scifact_latency_vectors
            ORDER BY
                embedding <=> %s
            LIMIT %s
            """,
            (
                query_embedding,
                TOP_K,
            ),
        )

        rows = cursor.fetchall()

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000

    return (
        [
            str(row[0])
            for row in rows
        ],
        elapsed_ms,
    )


def percentile(
    values,
    p,
):
    ordered = sorted(
        values
    )

    index = int(
        round(
            (len(ordered) - 1)
            * p
        )
    )

    return ordered[index]


def print_stats(
    name,
    values,
):
    print(
        f"\n{name}"
    )

    print(
        f"Average : "
        f"{statistics.mean(values):.3f} ms"
    )

    print(
        f"P50     : "
        f"{percentile(values, 0.50):.3f} ms"
    )

    print(
        f"P95     : "
        f"{percentile(values, 0.95):.3f} ms"
    )

    print(
        f"Min     : "
        f"{min(values):.3f} ms"
    )

    print(
        f"Max     : "
        f"{max(values):.3f} ms"
    )


def main():
    print("=" * 80)
    print(
        "SciFact Vector Search "
        "Latency Benchmark"
    )
    print("=" * 80)

    query_ids = (
        load_test_query_ids()
    )

    documents = (
        load_document_embeddings()
    )

    query_embeddings = (
        load_query_embeddings()
    )

    print(
        f"\nDocuments: "
        f"{len(documents)}"
    )

    print(
        f"Queries: "
        f"{len(query_ids)}"
    )

    connection = (
        create_connection()
    )

    try:
        prepare_table(
            connection,
            documents,
        )

        # ====================================================
        # Warm-up
        # ====================================================

        print(
            "\nWarm-up..."
        )

        for query_id in query_ids[:20]:

            query_embedding = (
                query_embeddings[
                    query_id
                ]
            )

            exact_search(
                connection,
                query_embedding,
            )

            hnsw_search(
                connection,
                query_embedding,
            )

            connection.commit()

        print(
            "Warm-up 完成"
        )

        # ====================================================
        # Benchmark
        # ====================================================

        exact_times = []
        hnsw_times = []

        mismatch_count = 0

        for index, query_id in enumerate(
            query_ids,
            start=1,
        ):
            query_embedding = (
                query_embeddings[
                    query_id
                ]
            )

            (
                exact_ids,
                exact_ms,
            ) = exact_search(
                connection,
                query_embedding,
            )

            connection.commit()

            (
                hnsw_ids,
                hnsw_ms,
            ) = hnsw_search(
                connection,
                query_embedding,
            )

            connection.commit()

            exact_times.append(
                exact_ms
            )

            hnsw_times.append(
                hnsw_ms
            )

            if exact_ids != hnsw_ids:
                mismatch_count += 1

            if (
                index % 50 == 0
                or index
                == len(query_ids)
            ):
                print(
                    f"进度："
                    f"{index}"
                    " / "
                    f"{len(query_ids)}"
                )

        print("\n")
        print("=" * 80)
        print("Latency Results")
        print("=" * 80)

        print_stats(
            "Exact PostgreSQL Scan",
            exact_times,
        )

        print_stats(
            "HNSW Index Search",
            hnsw_times,
        )

        exact_avg = (
            statistics.mean(
                exact_times
            )
        )

        hnsw_avg = (
            statistics.mean(
                hnsw_times
            )
        )

        speedup = (
            exact_avg
            / hnsw_avg
            if hnsw_avg > 0
            else 0
        )

        print(
            "\nSpeedup:"
        )

        print(
            f"{speedup:.2f}x"
        )

        print(
            "\nTop-5 完全不同的 Query 数："
            f"{mismatch_count}"
            " / "
            f"{len(query_ids)}"
        )

        print(
            "\nHNSW Config:"
        )

        print(
            f"m={HNSW_M}, "
            f"ef_construction="
            f"{HNSW_EF_CONSTRUCTION}, "
            f"ef_search="
            f"{HNSW_EF_SEARCH}"
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
