from backend.evaluation.rag.evaluate_retrieval import (
    load_corpus,
    load_queries,
    load_qrels,
    create_keyword_connection,
    keyword_search,
)


TOP_K = 5
CANDIDATE_K = 25


def main():
    print("=" * 80)
    print("SciFact Keyword Retrieval Diagnosis")
    print("=" * 80)

    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels()

    connection = create_keyword_connection(
        corpus
    )

    total_queries = len(qrels)

    zero_count = 0
    one_to_four_count = 0
    five_or_more_count = 0

    total_candidate_count = 0

    hit_at_5_count = 0

    zero_examples = []
    miss_examples = []

    try:
        for index, query_id in enumerate(
            qrels.keys(),
            start=1,
        ):
            query = queries[query_id]

            relevant_ids = qrels[
                query_id
            ]

            # 看 Top-25 候选数量
            candidates = keyword_search(
                connection,
                query,
                CANDIDATE_K,
            )

            candidate_count = len(
                candidates
            )

            total_candidate_count += (
                candidate_count
            )

            if candidate_count == 0:
                zero_count += 1

                if len(zero_examples) < 10:
                    zero_examples.append(
                        {
                            "query_id":
                                query_id,
                            "query":
                                query,
                        }
                    )

            elif candidate_count < 5:
                one_to_four_count += 1

            else:
                five_or_more_count += 1

            # 再正式看 Top-5 是否命中
            top5_ids = [
                item["document_id"]
                for item
                in candidates[:TOP_K]
            ]

            if (
                set(top5_ids)
                & set(relevant_ids)
            ):
                hit_at_5_count += 1

            elif len(miss_examples) < 10:
                miss_examples.append(
                    {
                        "query_id":
                            query_id,
                        "query":
                            query,
                        "candidate_count":
                            candidate_count,
                        "top5":
                            top5_ids,
                        "relevant":
                            sorted(
                                relevant_ids
                            ),
                    }
                )

            if (
                index % 50 == 0
                or index == total_queries
            ):
                print(
                    f"进度："
                    f"{index}"
                    f" / "
                    f"{total_queries}"
                )

    finally:
        connection.close()

    average_candidates = (
        total_candidate_count
        / total_queries
    )

    hit_at_5 = (
        hit_at_5_count
        / total_queries
    )

    print("\n")
    print("=" * 80)
    print("Keyword Diagnosis Results")
    print("=" * 80)

    print(
        f"\n总 Query 数："
        f"{total_queries}"
    )

    print(
        f"\n返回 0 条："
        f"{zero_count}"
        f" "
        f"({zero_count / total_queries:.2%})"
    )

    print(
        f"返回 1~4 条："
        f"{one_to_four_count}"
        f" "
        f"({one_to_four_count / total_queries:.2%})"
    )

    print(
        f"返回 >=5 条："
        f"{five_or_more_count}"
        f" "
        f"({five_or_more_count / total_queries:.2%})"
    )

    print(
        f"\n平均 Keyword Candidate 数："
        f"{average_candidates:.2f}"
    )

    print(
        f"Keyword Hit@5："
        f"{hit_at_5:.4f}"
    )

    print("\n")
    print("=" * 80)
    print("部分 0 Candidate Query")
    print("=" * 80)

    for item in zero_examples:
        print(
            f"\n[{item['query_id']}] "
            f"{item['query']}"
        )

    print("\n")
    print("=" * 80)
    print("部分未命中 Query")
    print("=" * 80)

    for item in miss_examples:
        print(
            f"\nQuery ID: "
            f"{item['query_id']}"
        )

        print(
            f"Query: "
            f"{item['query']}"
        )

        print(
            f"Candidate Count: "
            f"{item['candidate_count']}"
        )

        print(
            f"Top-5: "
            f"{item['top5']}"
        )

        print(
            f"Relevant: "
            f"{item['relevant']}"
        )


if __name__ == "__main__":
    main()
