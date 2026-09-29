import json
from pathlib import Path

from backend.app.services.embedding_service import embedding_model


BASE_DIR = Path(__file__).resolve().parent

CORPUS_PATH = (
    BASE_DIR
    / "datasets"
    / "scifact"
    / "corpus.jsonl"
)

CACHE_DIR = (
    BASE_DIR
    / "cache"
)

CACHE_PATH = (
    CACHE_DIR
    / "scifact_document_embeddings.jsonl"
)


# 每批处理多少篇文档
BATCH_SIZE = 16


def load_corpus():
    documents = []

    with CORPUS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
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

            # Vector 检索统一使用：
            # title + 正文
            embedding_text = (
                title
                + "\n"
                + text
            ).strip()

            documents.append(
                {
                    "document_id": document_id,
                    "title": title,
                    "text": text,
                    "embedding_text": embedding_text,
                }
            )

    return documents


def load_completed_ids():
    """
    读取已经生成好的 document_id。

    如果程序中途退出，
    下次运行时会自动跳过这些文档。
    """

    completed_ids = set()

    if not CACHE_PATH.exists():
        return completed_ids

    with CACHE_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)

                completed_ids.add(
                    str(
                        item["document_id"]
                    )
                )

            except Exception:
                # 如果最后一行因为异常中断写坏，
                # 暂时跳过
                continue

    return completed_ids


def main():
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("SciFact Document Embedding Builder")
    print("=" * 80)

    documents = load_corpus()

    print(
        f"\nCorpus 文档总数："
        f"{len(documents)}"
    )

    completed_ids = (
        load_completed_ids()
    )

    print(
        f"已经完成："
        f"{len(completed_ids)}"
    )

    remaining_documents = [
        document
        for document in documents
        if document["document_id"]
        not in completed_ids
    ]

    print(
        f"剩余文档："
        f"{len(remaining_documents)}"
    )

    if not remaining_documents:
        print(
            "\n所有文档 Embedding "
            "已经生成完成。"
        )
        return

    total = len(documents)

    with CACHE_PATH.open(
        "a",
        encoding="utf-8",
    ) as output_file:

        for start in range(
            0,
            len(remaining_documents),
            BATCH_SIZE,
        ):

            batch = (
                remaining_documents[
                    start:
                    start + BATCH_SIZE
                ]
            )

            texts = [
                document[
                    "embedding_text"
                ]
                for document in batch
            ]

            print(
                f"\n正在处理 Batch："
                f"{start + 1}"
                f" - "
                f"{start + len(batch)}"
                f" / "
                f"{len(remaining_documents)}"
            )

            embeddings = (
                embedding_model
                .embed_documents(
                    texts
                )
            )

            if (
                len(embeddings)
                != len(batch)
            ):
                raise RuntimeError(
                    "Embedding 返回数量"
                    "与输入文档数量不一致"
                )

            for document, embedding in zip(
                batch,
                embeddings,
            ):

                result = {
                    "document_id":
                        document[
                            "document_id"
                        ],
                    "embedding":
                        embedding,
                }

                output_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            # 每个 Batch 都立即写入硬盘
            output_file.flush()

            finished = (
                len(completed_ids)
                + start
                + len(batch)
            )

            print(
                f"总进度："
                f"{finished}"
                f" / "
                f"{total}"
            )

    print("\n" + "=" * 80)
    print("SciFact Embedding 生成完成")
    print("=" * 80)

    print(
        f"\n缓存文件："
        f"{CACHE_PATH}"
    )


if __name__ == "__main__":
    main()
