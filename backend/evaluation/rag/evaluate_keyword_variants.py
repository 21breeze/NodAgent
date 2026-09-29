import os
import re

import psycopg

from backend.evaluation.rag.evaluate_retrieval import (
    load_corpus,
    load_queries,
    load_qrels,
    calculate_metrics,
)


TOP_K = 5
CANDIDATE_K = 25


# ============================================================
# PostgreSQL Connection
# ============================================================

def create_connection(corpus):
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

    # 从 Mac Host 执行脚本时，
    # Docker postgres 服务通过 localhost:15432 访问
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

    connection.autocommit = True

    with connection.cursor() as cursor:

        cursor.execute(
            """
            DROP TABLE IF EXISTS
            scifact_keyword_eval
            """
        )

        cursor.execute(
            """
            CREATE TEMP TABLE
            scifact_keyword_eval (
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
            scifact_keyword_eval (
                document_id,
                content
            )
            VALUES (%s, %s)
            """,
            rows,
        )

        # 当前 NodAgent 使用的 simple index
        cursor.execute(
            """
            CREATE INDEX
            scifact_keyword_simple_idx
            ON scifact_keyword_eval
            USING GIN (
                to_tsvector(
                    'simple',
                    content
                )
            )
            """
        )

        # 用来测试 English FTS
        cursor.execute(
            """
            CREATE INDEX
            scifact_keyword_english_idx
            ON scifact_keyword_eval
            USING GIN (
                to_tsvector(
                    'english',
                    content
                )
            )
            """
        )

        cursor.execute(
            """
            ANALYZE scifact_keyword_eval
            """
        )

    return connection


# ============================================================
# Query 转 OR 查询
# ============================================================

def build_or_query(query):
    """
    例如：

    A deficiency of vitamin B12

    转成：

    A OR deficiency OR of OR vitamin OR B12

    然后交给 PostgreSQL websearch_to_tsquery。
    """

    tokens = re.findall(
        r"[A-Za-z0-9]+",
        query,
    )

    if not tokens:
        return ""

    return " OR ".join(tokens)


# ============================================================
# Keyword Search
# ============================================================

def keyword_search(
    connection,
    query,
    config,
    mode,
    top_k,
):
    if mode == "and":

        query_function = (
            "plainto_tsquery"
        )

        query_value = query

    elif mode == "or":

        query_function = (
            "websearch_to_tsquery"
        )

        query_value = (
            build_or_query(query)
        )

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    if not query_value:
        return []

    sql = f"""
        SELECT
            document_id,
            ts_rank_cd(
                to_tsvector(
                    '{config}',
                    content
                ),
                {query_function}(
                    '{config}',
                    %s
                )
            ) AS score

        FROM scifact_keyword_eval

        WHERE
            to_tsvector(
                '{config}',
                content
            )
            @@
            {query_function}(
                '{config}',
                %s
            )

        ORDER BY
            score DESC,
            document_id ASC

        LIMIT %s
    """

    with connection.cursor() as cursor:

        cursor.execute(
            sql,
            (
                query_value,
                query_value,
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
# Evaluate
# ============================================================

def evaluate_variant(
    connection,
    queries,
    qrels,
    name,
    config,
    mode,
):
    total_queries = len(qrels)

    zero_count = 0

    one_to_four_count = 0

    five_or_more_count = 0

    total_candidates = 0

    hit_total = 0.0
    recall_total = 0.0
    rr_total = 0.0

    for index, query_id in enumerate(
        qrels.keys(),
        start=1,
    ):

        query = queries[
            query_id
        ]

        relevant_ids = qrels[
            query_id
        ]

        candidates = keyword_search(
            connection=connection,
            query=query,
            config=config,
            mode=mode,
            top_k=CANDIDATE_K,
        )

        candidate_count = len(
            candidates
        )

        total_candidates += (
            candidate_count
        )

        if candidate_count == 0:
            zero_count += 1

        elif candidate_count < 5:
            one_to_four_count += 1

        else:
            five_or_more_count += 1

        retrieved_ids = [
            item["document_id"]
            for item
            in candidates[:TOP_K]
        ]

        metrics = calculate_metrics(
            retrieved_ids,
            relevant_ids,
        )

        hit_total += metrics["hit"]
        recall_total += metrics["recall"]
        rr_total += metrics["rr"]

        if (
            index % 100 == 0
            or index == total_queries
        ):
            print(
                f"{name}: "
                f"{index}"
                f" / "
                f"{total_queries}"
            )

    return {
        "name":
            name,

        "zero":
            zero_count,

        "zero_rate":
            zero_count
            / total_queries,

        "one_to_four":
            one_to_four_count,

        "five_or_more":
            five_or_more_count,

        "avg_candidates":
            total_candidates
            / total_queries,

        "hit_at_5":
            hit_total
            / total_queries,

        "recall_at_5":
            recall_total
            / total_queries,

        "mrr_at_5":
            rr_total
            / total_queries,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("SciFact Keyword Retrieval Ablation")
    print("=" * 80)

    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels()

    print(
        f"\nCorpus: "
        f"{len(corpus)}"
    )

    print(
        f"Test Queries: "
        f"{len(qrels)}"
    )

    print(
        "\n正在创建 PostgreSQL "
        "临时 FTS 索引..."
    )

    connection = (
        create_connection(
            corpus
        )
    )

    print(
        "临时索引创建完成\n"
    )

    variants = [
        (
            "Simple + AND (Current)",
            "simple",
            "and",
        ),

        (
            "English + AND",
            "english",
            "and",
        ),

        (
            "Simple + OR",
            "simple",
            "or",
        ),

        (
            "English + OR",
            "english",
            "or",
        ),
    ]

    results = []

    try:

        for (
            name,
            config,
            mode,
        ) in variants:

            print(
                "\n"
                + "=" * 80
            )

            print(
                f"Testing: {name}"
            )

            print(
                "=" * 80
            )

            result = (
                evaluate_variant(
                    connection=connection,
                    queries=queries,
                    qrels=qrels,
                    name=name,
                    config=config,
                    mode=mode,
                )
            )

            results.append(
                result
            )

    finally:
        connection.close()

    print("\n")
    print("=" * 100)
    print("Final Keyword Ablation Results")
    print("=" * 100)

    print(
        f"{'Method':<28}"
        f"{'Zero%':>10}"
        f"{'AvgCand':>10}"
        f"{'Hit@5':>12}"
        f"{'Recall@5':>12}"
        f"{'MRR@5':>12}"
    )

    print("-" * 84)

    for result in results:

        print(
            f"{result['name']:<28}"
            f"{result['zero_rate']:>10.2%}"
            f"{result['avg_candidates']:>10.2f}"
            f"{result['hit_at_5']:>12.4f}"
            f"{result['recall_at_5']:>12.4f}"
            f"{result['mrr_at_5']:>12.4f}"
        )


if __name__ == "__main__":
    main()
