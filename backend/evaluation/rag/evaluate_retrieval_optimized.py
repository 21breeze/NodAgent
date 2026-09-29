import re

import backend.evaluation.rag.evaluate_retrieval as base


def build_or_query(query: str) -> str:
    tokens = re.findall(
        r"[A-Za-z0-9]+",
        query,
    )

    if not tokens:
        return ""

    return " OR ".join(tokens)


def optimized_keyword_search(
    connection,
    query,
    top_k,
):
    query_value = build_or_query(
        query
    )

    if not query_value:
        return []

    with connection.cursor() as cursor:

        cursor.execute(
            """
            SELECT
                document_id,
                ts_rank_cd(
                    to_tsvector(
                        'english',
                        content
                    ),
                    websearch_to_tsquery(
                        'english',
                        %s
                    )
                ) AS keyword_score

            FROM scifact_eval_docs

            WHERE
                to_tsvector(
                    'english',
                    content
                )
                @@
                websearch_to_tsquery(
                    'english',
                    %s
                )

            ORDER BY
                keyword_score DESC,
                document_id ASC

            LIMIT %s
            """,
            (
                query_value,
                query_value,
                top_k,
            ),
        )

        rows = cursor.fetchall()

    return [
        {
            "document_id": str(
                row[0]
            ),
            "score": float(
                row[1]
            ),
        }
        for row in rows
    ]


# 把原评测脚本中的 Keyword 检索
# 临时替换成优化版本
base.keyword_search = (
    optimized_keyword_search
)


if __name__ == "__main__":
    print(
        "Using optimized keyword retrieval:"
    )
    print(
        "english + OR + "
        "websearch_to_tsquery"
    )
    print()

    base.main()
