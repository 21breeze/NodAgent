from backend.app.services.embedding_service import (
    EMBEDDING_DIMENSION,
    embedding_model,
)


def main():
    text = "你好，这是一个测试。"

    vector = embedding_model.embed_query(
        text
    )

    print(
        "实际向量维度:",
        len(vector),
    )

    print(
        "配置向量维度:",
        EMBEDDING_DIMENSION,
    )

    print(
        "前10个向量值:",
        vector[:10],
    )

    assert len(vector) == EMBEDDING_DIMENSION

    print("Embedding 测试通过")


if __name__ == "__main__":
    main()