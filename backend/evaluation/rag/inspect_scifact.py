import csv
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets" / "scifact"

CORPUS_PATH = DATASET_DIR / "corpus.jsonl"
QUERIES_PATH = DATASET_DIR / "queries.jsonl"
QRELS_PATH = DATASET_DIR / "qrels" / "test.tsv"


def load_corpus():
    corpus = {}

    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            document_id = str(item["_id"])

            corpus[document_id] = {
                "title": item.get("title", ""),
                "text": item.get("text", ""),
            }

    return corpus


def load_queries():
    queries = {}

    with QUERIES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            query_id = str(item["_id"])
            queries[query_id] = item["text"]

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
            query_id = str(row["query-id"])
            corpus_id = str(row["corpus-id"])
            score = int(row["score"])

            if query_id not in qrels:
                qrels[query_id] = {}

            qrels[query_id][corpus_id] = score

    return qrels


def main():
    print("=" * 80)
    print("SciFact Dataset Inspection")
    print("=" * 80)

    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels()

    print(f"\nCorpus 文档数量：{len(corpus)}")
    print(f"Queries 总数量：{len(queries)}")
    print(f"Test Qrels Query 数量：{len(qrels)}")

    query_id = next(iter(qrels))
    query_text = queries[query_id]
    relevant_documents = qrels[query_id]

    print("\n")
    print("=" * 80)
    print("示例 Query")
    print("=" * 80)

    print(f"Query ID:\n{query_id}")
    print(f"\nQuery Text:\n{query_text}")

    print("\nGround Truth Documents:")

    for document_id, score in relevant_documents.items():
        document = corpus[document_id]

        print("\n" + "-" * 80)

        print(f"Document ID: {document_id}")
        print(f"Relevance Score: {score}")

        print(f"\nTitle:\n{document['title']}")

        text = document["text"]
        preview = text[:800]

        print(f"\nText Preview:\n{preview}")

    print("\n")
    print("=" * 80)
    print("SciFact 数据读取成功")
    print("=" * 80)


if __name__ == "__main__":
    main()
