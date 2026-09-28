import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)


# ============================================================
# 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# 使用项目真实 Router
# ============================================================

from backend.app.graphs.main_graph import (
    ROUTER_SYSTEM_PROMPT,
    get_router_model,
)


# ============================================================
# 路径
# ============================================================

EVALUATION_DIR = Path(__file__).parent

DATASET_PATH = (
    EVALUATION_DIR
    / "datasets"
    / "router_benchmark_180.jsonl"
)

REPORT_DIR = (
    EVALUATION_DIR
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Router 类别
# ============================================================

ROUTES = [
    "direct",
    "knowledge",
    "document",
    "memory",
    "external",
    "delete_document",
]


# ============================================================
# 命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="NodAgent Router Evaluation"
    )

    parser.add_argument(
        "--split",
        choices=[
            "easy",
            "validation",
            "test",
        ],
        default="validation",
        help=(
            "easy: 基础 60 条；"
            "validation: 用于分析和调 Prompt 的 60 条；"
            "test: 最终未见测试集 60 条。"
        ),
    )

    return parser.parse_args()


# ============================================================
# 数据读取
# ============================================================

def load_dataset():
    cases = []

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                case = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"第 {line_number} 行 JSON 格式错误: {exc}"
                ) from exc

            cases.append(case)

    return cases


# ============================================================
# 从 ID 提取编号
#
# direct_01 -> 1
# knowledge_15 -> 15
# delete_document_30 -> 30
# ============================================================

def get_case_number(case):
    case_id = case["id"]

    try:
        number = int(
            case_id.rsplit(
                "_",
                1,
            )[1]
        )

    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"无法从 case id 中解析编号: {case_id}"
        ) from exc

    return number


# ============================================================
# 数据集划分
#
# easy:
#   01-10
#
# validation:
#   medium: 11-15
#   hard:   21-25
#
# test:
#   medium: 16-20
#   hard:   26-30
# ============================================================

def filter_cases(
    cases,
    evaluation_split,
):
    selected = []

    for case in cases:

        number = get_case_number(
            case
        )

        if evaluation_split == "easy":

            if 1 <= number <= 10:
                selected.append(
                    case
                )

        elif evaluation_split == "validation":

            if (
                11 <= number <= 15
                or
                21 <= number <= 25
            ):
                selected.append(
                    case
                )

        elif evaluation_split == "test":

            if (
                16 <= number <= 20
                or
                26 <= number <= 30
            ):
                selected.append(
                    case
                )

    return selected


# ============================================================
# 检查数据划分
# ============================================================

def validate_split(
    cases,
    evaluation_split,
):
    """
    确保：
    - 总数 = 60
    - 每个 Route = 10
    """

    route_counter = Counter(
        case["expected_route"]
        for case in cases
    )

    print("\n数据划分检查：")

    print(
        f"Split: {evaluation_split}"
    )

    print(
        f"Total: {len(cases)}"
    )

    for route in ROUTES:

        count = route_counter[
            route
        ]

        print(
            f"  {route:<20}: {count}"
        )

    if len(cases) != 60:
        raise ValueError(
            f"{evaluation_split} 应该有 60 条，"
            f"实际得到 {len(cases)} 条。"
        )

    for route in ROUTES:

        if route_counter[route] != 10:
            raise ValueError(
                f"{route} 应该有 10 条，"
                f"实际得到 {route_counter[route]} 条。"
            )


# ============================================================
# Benchmark History -> LangChain Message
# ============================================================

def convert_history_to_messages(
    history,
):
    messages = []

    for item in history:

        role = item["role"]
        content = item["content"]

        if role == "user":

            messages.append(
                HumanMessage(
                    content=content
                )
            )

        elif role == "assistant":

            messages.append(
                AIMessage(
                    content=content
                )
            )

        else:

            raise ValueError(
                f"暂不支持 history role: {role}"
            )

    return messages


# ============================================================
# Router System Prompt
# ============================================================

def build_router_system_prompt(
    memory_prompt,
):
    router_system_prompt = (
        ROUTER_SYSTEM_PROMPT
    )

    if memory_prompt:

        router_system_prompt += (
            "\n\n"
            "==================================================\n"
            "当前可用长期记忆\n"
            "==================================================\n"
            f"{memory_prompt}\n\n"
            "这些长期记忆已经可以作为 Router "
            "判断路由时的上下文。"
            "如果长期记忆本身足以回答用户问题，"
            "请选择 direct。"
            "如果用户要保存、更新、列出或删除长期记忆，"
            "请选择 memory。"
            "只有确实需要读取 Workspace 文档正文时，"
            "才选择 knowledge。"
        )

    return router_system_prompt


# ============================================================
# 调用真实 Router
# ============================================================

async def evaluate_one_case(
    case,
):
    messages = (
        convert_history_to_messages(
            case.get(
                "history",
                [],
            )
        )
    )

    messages.append(
        HumanMessage(
            content=case[
                "current_query"
            ]
        )
    )

    router_system_prompt = (
        build_router_system_prompt(
            case.get(
                "memory_prompt",
                "",
            )
        )
    )

    decision = await (
        get_router_model().ainvoke(
            [
                SystemMessage(
                    content=router_system_prompt
                ),
                *messages,
            ]
        )
    )

    return decision


# ============================================================
# 计算指标
# ============================================================

def calculate_metrics(
    results,
):
    total = len(
        results
    )

    correct = sum(
        1
        for result in results
        if result["correct"]
    )

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    # ========================================================
    # Confusion Matrix
    # ========================================================

    confusion_matrix = {
        expected: {
            predicted: 0
            for predicted in ROUTES
        }
        for expected in ROUTES
    }

    for result in results:

        expected = (
            result[
                "expected_route"
            ]
        )

        predicted = (
            result[
                "predicted_route"
            ]
        )

        if (
            predicted is not None
            and
            expected in confusion_matrix
            and
            predicted
            in confusion_matrix[
                expected
            ]
        ):

            confusion_matrix[
                expected
            ][
                predicted
            ] += 1

    # ========================================================
    # 每类 Precision / Recall / F1
    # ========================================================

    per_class = {}

    for route in ROUTES:

        tp = sum(
            1
            for result in results
            if (
                result[
                    "expected_route"
                ] == route
                and
                result[
                    "predicted_route"
                ] == route
            )
        )

        fp = sum(
            1
            for result in results
            if (
                result[
                    "expected_route"
                ] != route
                and
                result[
                    "predicted_route"
                ] == route
            )
        )

        fn = sum(
            1
            for result in results
            if (
                result[
                    "expected_route"
                ] == route
                and
                result[
                    "predicted_route"
                ] != route
            )
        )

        support = sum(
            1
            for result in results
            if result[
                "expected_route"
            ] == route
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
            if (
                precision
                + recall
            ) > 0
            else 0.0
        )

        per_class[
            route
        ] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    macro_f1 = (
        sum(
            value["f1"]
            for value
            in per_class.values()
        )
        / len(ROUTES)
    )

    return {
        "total": total,
        "correct": correct,
        "wrong": (
            total
            - correct
        ),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": (
            confusion_matrix
        ),
    }


# ============================================================
# 打印结果
# ============================================================

def print_metrics(
    metrics,
    evaluation_split,
):
    print("\n")
    print("=" * 90)
    print(
        "NodAgent Router Evaluation Summary"
    )
    print("=" * 90)

    print(
        f"Split     : "
        f"{evaluation_split}"
    )

    print(
        f"Total     : "
        f"{metrics['total']}"
    )

    print(
        f"Correct   : "
        f"{metrics['correct']}"
    )

    print(
        f"Wrong     : "
        f"{metrics['wrong']}"
    )

    print(
        f"Accuracy  : "
        f"{metrics['accuracy']:.2%}"
    )

    print(
        f"Macro-F1  : "
        f"{metrics['macro_f1']:.4f}"
    )

    print("\n")

    print(
        f"{'Route':<20}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>12}"
    )

    print(
        "-" * 68
    )

    for route in ROUTES:

        data = (
            metrics[
                "per_class"
            ][route]
        )

        print(
            f"{route:<20}"
            f"{data['precision']:>12.4f}"
            f"{data['recall']:>12.4f}"
            f"{data['f1']:>12.4f}"
            f"{data['support']:>12}"
        )

    # ========================================================
    # Confusion Matrix
    # ========================================================

    print("\n")
    print(
        "Confusion Matrix"
    )

    print(
        "-" * 90
    )

    header = (
        f"{'Expected':<18}"
        + "".join(
            f"{route[:8]:>10}"
            for route in ROUTES
        )
    )

    print(
        header
    )

    for expected in ROUTES:

        row = (
            f"{expected:<18}"
            + "".join(
                f"{metrics['confusion_matrix'][expected][predicted]:>10}"
                for predicted in ROUTES
            )
        )

        print(
            row
        )

    print(
        "=" * 90
    )


# ============================================================
# 保存报告
# ============================================================

def save_reports(
    results,
    metrics,
    evaluation_split,
):
    timestamp = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    result_path = (
        REPORT_DIR
        / (
            f"router_"
            f"{evaluation_split}_"
            f"results_"
            f"{timestamp}.jsonl"
        )
    )

    summary_path = (
        REPORT_DIR
        / (
            f"router_"
            f"{evaluation_split}_"
            f"summary_"
            f"{timestamp}.json"
        )
    )

    # ========================================================
    # 详细结果
    # ========================================================

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for result in results:

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ========================================================
    # 汇总结果
    # ========================================================

    summary = {
        "evaluation_split": (
            evaluation_split
        ),
        **metrics,
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return (
        result_path,
        summary_path,
    )


# ============================================================
# Main
# ============================================================

async def main():

    args = parse_args()

    evaluation_split = (
        args.split
    )

    print("=" * 90)
    print(
        "NodAgent Router Evaluation"
    )
    print("=" * 90)

    print(
        f"\nEvaluation Split: "
        f"{evaluation_split}"
    )

    # ========================================================
    # 加载数据
    # ========================================================

    all_cases = (
        load_dataset()
    )

    cases = filter_cases(
        all_cases,
        evaluation_split,
    )

    # ========================================================
    # 检查数据划分
    # ========================================================

    validate_split(
        cases,
        evaluation_split,
    )

    print(
        "\n开始逐条调用真实 Router..."
    )

    results = []

    # ========================================================
    # 顺序执行
    # ========================================================

    for index, case in enumerate(
        cases,
        start=1,
    ):

        print(
            f"\n"
            f"[{index:02d}/"
            f"{len(cases)}] "
            f"{case['id']}"
        )

        print(
            f"Difficulty: "
            f"{case['difficulty']}"
        )

        print(
            f"Query     : "
            f"{case['current_query']}"
        )

        print(
            f"Expected  : "
            f"{case['expected_route']}"
        )

        try:

            decision = (
                await evaluate_one_case(
                    case
                )
            )

            predicted_route = (
                decision.route
            )

            correct = (
                predicted_route
                == case[
                    "expected_route"
                ]
            )

            result = {
                "id": case[
                    "id"
                ],
                "evaluation_split": (
                    evaluation_split
                ),
                "difficulty": case[
                    "difficulty"
                ],
                "current_query": case[
                    "current_query"
                ],
                "history": case.get(
                    "history",
                    [],
                ),
                "memory_prompt": case.get(
                    "memory_prompt",
                    "",
                ),
                "expected_route": case[
                    "expected_route"
                ],
                "predicted_route": (
                    predicted_route
                ),
                "correct": (
                    correct
                ),
                "specialist_task": (
                    decision.task
                ),
                "document_id": (
                    decision.document_id
                ),
                "document_filename": (
                    decision.document_filename
                ),
                "error": None,
            }

            print(
                f"Predicted : "
                f"{predicted_route}"
            )

            if correct:

                print(
                    "Result    : ✅"
                )

            else:

                print(
                    "Result    : ❌"
                )

        except Exception as exc:

            result = {
                "id": case[
                    "id"
                ],
                "evaluation_split": (
                    evaluation_split
                ),
                "difficulty": case[
                    "difficulty"
                ],
                "current_query": case[
                    "current_query"
                ],
                "history": case.get(
                    "history",
                    [],
                ),
                "memory_prompt": case.get(
                    "memory_prompt",
                    "",
                ),
                "expected_route": case[
                    "expected_route"
                ],
                "predicted_route": None,
                "correct": False,
                "specialist_task": None,
                "document_id": None,
                "document_filename": None,
                "error": (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            }

            print(
                "Predicted : ERROR"
            )

            print(
                f"Error     : "
                f"{exc}"
            )

            print(
                "Result    : ❌"
            )

        results.append(
            result
        )

    # ========================================================
    # 计算结果
    # ========================================================

    metrics = calculate_metrics(
        results
    )

    print_metrics(
        metrics,
        evaluation_split,
    )

    # ========================================================
    # 打印错题
    # ========================================================

    wrong_cases = [
        result
        for result in results
        if not result[
            "correct"
        ]
    ]

    if wrong_cases:

        print("\n")
        print("=" * 90)
        print(
            "错误案例"
        )
        print("=" * 90)

        for result in wrong_cases:

            print(
                f"\nID        : "
                f"{result['id']}"
            )

            print(
                f"Difficulty: "
                f"{result['difficulty']}"
            )

            print(
                f"Query     : "
                f"{result['current_query']}"
            )

            print(
                f"Expected  : "
                f"{result['expected_route']}"
            )

            print(
                f"Predicted : "
                f"{result['predicted_route']}"
            )

            print(
                f"History   : "
                f"{result['history']}"
            )

            print(
                f"Memory    : "
                f"{result['memory_prompt']}"
            )

            if result["error"]:

                print(
                    f"Error     : "
                    f"{result['error']}"
                )

            print(
                "-" * 90
            )

    else:

        print("\n")
        print(
            "✅ 本次没有分类错误。"
        )

    # ========================================================
    # 保存报告
    # ========================================================

    result_path, summary_path = (
        save_reports(
            results,
            metrics,
            evaluation_split,
        )
    )

    print("\n")
    print("=" * 90)
    print(
        "评测完成"
    )
    print("=" * 90)

    print(
        f"详细结果："
        f"{result_path}"
    )

    print(
        f"汇总结果："
        f"{summary_path}"
    )


if __name__ == "__main__":

    asyncio.run(
        main()
    )
