import csv
import json
import math
from pathlib import Path

from backend.app.services.embedding_service import embedding_model


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets" / "scifact"

CORPUS_PATH = DATASET_DIR / "corpus.jsonl"
QUERIES_PATH = DATASET_DIR / "queries.jsonl"
QRELS_PATH = DATASET_DIR / "qrels" / "test.tsv"


def load_corpus():
    corpus = {}

    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            corpus[str(item["_id"])] = {
                "title": item.get("title", ""),
                "text": item.get("text", ""),
            }

    return corpus


def load_queries():
    queries = {}

    with QUERIES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            queries[str(item["_id"])] = item["text"]

    return queries


def load_first_qrel():
    with QRELS_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        row = next(reader)

        return (
            str(row["query-id"]),
            str(row["corpus-id"]),
        )


def cosine_similarity(a, b):
    dot_product = sum(
        x * y
        for x, y in zip(a, b)
    )

    norm_a = math.sqrt(
        sum(x * x for x in a)
    )

    norm_b = math.sqrt(
        sum(y * y for y in b)
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (
        norm_a * norm_b
    )


def main():
    print("=" * 80)
    print("SciFact Vector Smoke Test")
    print("=" * 80)

    corpus = load_corpus()
    queries = load_queries()

    query_id, document_id = (
        load_first_qrel()
    )

    query_text = queries[query_id]

    document = corpus[document_id]

    document_text = (
        document["title"]
        + "\n"
        + document["text"]
    ).strip()

    print(f"\nQuery ID: {query_id}")
    print(f"Query:\n{query_text}")

    print(
        f"\nGround Truth Document ID: "
        f"{document_id}"
    )

    print(
        f"Title:\n"
        f"{document['title']}"
    )

    print("\n正在生成 Query Embedding...")

    query_embedding = (
        embedding_model.embed_query(
            query_text
        )
    )

    print("Query Embedding 完成")

    print("\n正在生成 Document Embedding...")

    document_embedding = (
        embedding_model.embed_query(
            document_text
        )
    )

    print("Document Embedding 完成")

    print(
        f"\nQuery Embedding 维度："
        f"{len(query_embedding)}"
    )

    print(
        f"Document Embedding 维度："
        f"{len(document_embedding)}"
    )

    similarity = cosine_similarity(
        query_embedding,
        document_embedding,
    )

    print(
        f"\nQuery 与 Ground Truth Document "
        f"余弦相似度：{similarity:.6f}"
    )

    print("\n" + "=" * 80)
    print("Vector Smoke Test 成功")
    print("=" * 80)


if __name__ == "__main__":
    main()
