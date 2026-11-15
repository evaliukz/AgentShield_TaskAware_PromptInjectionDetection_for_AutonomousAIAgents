
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def evaluate(y_true, y_pred):
    # 标签 1 表示注入，标签 0 表示良性；混淆矩阵按 [0, 1] 排列。
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        # 良性样本中被误判为注入的比例。
        "fpr": fp / (fp + tn),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", required=True)
    parser.add_argument("--v2", required=True)
    args = parser.parse_args()

    # 读取两个版本的预测结果；文件都需包含 id、actual、predicted 列。
    v1 = pd.read_csv(args.v1)
    v2 = pd.read_csv(args.v2)

    # 确认样本一一对应：ID 不重复，且两个文件包含完全相同的样本。
    assert v1["id"].is_unique
    assert v2["id"].is_unique
    assert set(v1["id"]) == set(v2["id"])

    # 按样本 ID 配对两版预测，并用后缀区分各版本的真实标签和预测值。
    df = v1[["id", "actual", "predicted"]].merge(
        v2[["id", "actual", "predicted"]],
        on="id",
        suffixes=("_v1", "_v2"),
        validate="one_to_one",
    )

    # 配对比较前，确保两个文件对同一样本使用相同的真实标签。
    assert (
        df["actual_v1"] == df["actual_v2"]
    ).all(), "Ground-truth labels differ!"

    # 缺失预测无法用于指标或版本间比较，及时报错而不是静默忽略。
    assert df["predicted_v1"].notna().all()
    assert df["predicted_v2"].notna().all()

    # y 是真实标签，p1/p2 分别是 v1/v2 的预测标签。
    y = df["actual_v1"]
    p1 = df["predicted_v1"]
    p2 = df["predicted_v2"]

    # 分别标记两个版本是否判断正确，以统计改进、退化和一致结果。
    correct1 = p1 == y
    correct2 = p2 == y

    # fixed：v1 错、v2 对；regressed：v1 对、v2 错。
    fixed = (~correct1) & correct2
    regressed = correct1 & (~correct2)
    both_wrong = (~correct1) & (~correct2)

    print("========== PAIRED COMPARISON ==========")
    print("v1 errors fixed by v2:", fixed.sum())
    print("v2 new errors:", regressed.sum())
    print("Both wrong:", both_wrong.sum())
    print("Both correct:", (correct1 & correct2).sum())

    print("\n========== METRICS ==========")
    print("v1:", evaluate(y, p1))
    print("v2:", evaluate(y, p2))

    print("\n========== CHANGED SAMPLES ==========")

    # 输出预测标签发生变化的样本，并标记变化前后的判断是否正确。
    changed = df[p1 != p2].copy()
    changed["v1_correct"] = correct1[p1 != p2]
    changed["v2_correct"] = correct2[p1 != p2]

    print(changed.to_string(index=False))

    # McNemar 精确检验只比较两版结果不一致的样本：
    # b 是 v1 错而 v2 对的数量，c 是 v1 对而 v2 错的数量。
    b = int(fixed.sum())
    c = int(regressed.sum())

    if b + c > 0:
        # 在两版本准确率无差异的假设下，改进与退化的概率各为 0.5。
        p_value = binomtest(
            min(b, c),
            n=b + c,
            p=0.5,
            alternative="two-sided"
        ).pvalue

        print("\nMcNemar exact p-value:", p_value)

    # 将预测不同的样本另存为 CSV，便于逐条检查错误变化。
    changed.to_csv(
        Path(__file__).resolve().parents[1]
        / "data"
        / "reports"
        / "paired_changes.csv",
        index=False
    )


if __name__ == "__main__":
    main()

# python -m milestones.milestone1.scripts.compare_models \
#   --v1 milestones/milestone1/data/reports/zero_shot_v1_validation.csv \
#   --v2 milestones/milestone1/data/reports/zero_shot_v2_validation.csv