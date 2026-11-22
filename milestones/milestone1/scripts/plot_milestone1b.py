import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MILESTONE_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = MILESTONE_ROOT / "assets"
REPORTS_DIR = MILESTONE_ROOT / "data" / "reports"

# Palette aligned with Milestone 1 figures.
NAVY = "#1B2A4A"
BLUE = "#3B77A6"
TEAL = "#238368"
SLATE = "#7A8899"
CRIMSON = "#BA5152"
AMBER = "#CE8032"

# Reference metrics from Milestone1Report.md (gpt-4o-mini v2, validation n=467).
GPT4O_MINI_V2_REF = {
    "accuracy": 0.9507,
    "precision": 0.9553,
    "recall": 0.9708,
    "f1": 0.9630,
    "fpr": 0.0881,
}


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_metrics_comparison(v1_meta: dict, v2_meta: dict, out_path: Path):
    labels = ["Accuracy", "Precision", "Recall", "F1", "FPR"]
    keys = ["accuracy", "precision", "recall", "f1", "fpr"]

    v1_vals = [v1_meta["metrics"][k] * 100 for k in keys]
    v2_vals = [v2_meta["metrics"][k] * 100 for k in keys]
    ref_vals = [GPT4O_MINI_V2_REF[k] * 100 for k in keys]

    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
    b1 = ax.bar(x - width, v1_vals, width, label="Qwen2.5-3B v1", color=BLUE)
    b2 = ax.bar(x, v2_vals, width, label="Qwen2.5-3B v2", color=TEAL)
    b3 = ax.bar(
        x + width,
        ref_vals,
        width,
        label="gpt-4o-mini v2 (M1 ref)",
        color=SLATE,
        alpha=0.85,
    )

    for bars in (b1, b2, b3):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax.set_ylabel("Percent (%)", fontsize=11)
    ax.set_title(
        "Milestone 1b validation metrics: Qwen2.5-3B-Instruct vs. M1 reference (n=467)",
        fontsize=12.5,
        fontweight="bold",
        color=NAVY,
        pad=28,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 112)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.09),
        ncol=3,
        fontsize=10,
    )
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(v1_meta: dict, v2_meta: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), dpi=200)
    fig.suptitle(
        "Qwen2.5-3B-Instruct confusion matrices on validation (n=467)",
        fontsize=13,
        fontweight="bold",
        color=NAVY,
        y=1.02,
    )

    panels = [
        ("Zero-shot v1", v1_meta["metrics"]),
        ("Zero-shot v2", v2_meta["metrics"]),
    ]
    vmax = max(
        m["tp"] for _, m in panels
    )

    for ax, (title, m) in zip(axes, panels):
        matrix = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Benign", "Injection"], fontsize=10.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Benign", "Injection"], fontsize=10.5)
        ax.set_xlabel("Predicted", fontsize=10.5)
        ax.set_ylabel("Actual", fontsize=10.5)
        style_axes(ax)

        threshold = vmax * 0.45
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                color = "white" if val > threshold else NAVY
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                    color=color,
                )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_paired_outcomes(v1_csv: Path, v2_csv: Path, out_path: Path):
    v1 = pd.read_csv(v1_csv)
    v2 = pd.read_csv(v2_csv)
    df = v1[["id", "actual", "predicted"]].merge(
        v2[["id", "actual", "predicted"]],
        on="id",
        suffixes=("_v1", "_v2"),
        validate="one_to_one",
    )
    c1 = df["predicted_v1"] == df["actual_v1"]
    c2 = df["predicted_v2"] == df["actual_v2"]

    both_correct = int((c1 & c2).sum())
    both_wrong = int((~c1 & ~c2).sum())
    fixed = int((~c1 & c2).sum())
    regressed = int((c1 & ~c2).sum())

    segments = [
        (both_correct, TEAL, f"Both correct: {both_correct}"),
        (both_wrong, CRIMSON, f"Both wrong: {both_wrong}"),
        (fixed, BLUE, f"v2 fixed: {fixed}"),
        (regressed, AMBER, f"v2 regression: {regressed}"),
    ]

    fig, ax = plt.subplots(figsize=(10, 2.8), dpi=200)
    left = 0
    for count, color, label in segments:
        ax.barh([0], [count], left=left, height=0.55, color=color, label=label)
        if count >= 12:
            ax.text(
                left + count / 2,
                0,
                str(count),
                ha="center",
                va="center",
                color="white",
                fontsize=12,
                fontweight="bold",
            )
        left += count

    ax.set_xlim(0, len(df))
    ax.set_yticks([])
    ax.set_xlabel("Paired validation cases (n=467)", fontsize=11, labelpad=8)
    ax.set_title(
        "Qwen2.5-3B-Instruct paired prediction outcomes (v1 → v2, McNemar p = 4.30e-07)",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
        pad=10,
    )
    style_axes(ax)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=4,
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_family_breakdown(v1_meta: dict, v2_meta: dict, out_path: Path):
    attack_families = [
        "approval_authority_escalation",
        "memory_trust_boundary",
        "rag_document_embedded",
        "tool_function_abuse",
    ]
    pretty_names = [
        "Approval / Authority\nEscalation (n=98)",
        "Memory Trust\nBoundary (n=56)",
        "RAG Document\nEmbedded (n=42)",
        "Tool / Function\nAbuse (n=112)",
        "Benign Specificity\n(1 − FPR, n=159)",
    ]

    v1_scores = [
        v1_meta["attack_family_metrics"][f]["recall"] * 100
        for f in attack_families
    ] + [(1.0 - v1_meta["attack_family_metrics"]["benign"]["fpr"]) * 100]

    v2_scores = [
        v2_meta["attack_family_metrics"][f]["recall"] * 100
        for f in attack_families
    ] + [(1.0 - v2_meta["attack_family_metrics"]["benign"]["fpr"]) * 100]

    x = np.arange(len(pretty_names))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=200)
    b1 = ax.bar(x - width / 2, v1_scores, width, label="Qwen2.5-3B v1", color=BLUE)
    b2 = ax.bar(x + width / 2, v2_scores, width, label="Qwen2.5-3B v2", color=TEAL)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax.set_ylabel("Subgroup accuracy / recall (%)", fontsize=10.5)
    ax.set_title(
        "Qwen2.5-3B-Instruct subgroup performance by attack family and benign specificity",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(pretty_names, fontsize=9.5)
    ax.set_ylim(0, 114)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Milestone 1b figures from Qwen2.5-3B-Instruct reports."
    )
    parser.add_argument(
        "--v1-prefix",
        default=str(REPORTS_DIR / "zero_shot_qwen2.5-3b-instruct_v1_validation"),
    )
    parser.add_argument(
        "--v2-prefix",
        default=str(REPORTS_DIR / "zero_shot_qwen2.5-3b-instruct_v2_validation"),
    )
    parser.add_argument("--assets-dir", default=str(ASSETS_DIR))
    args = parser.parse_args()

    v1_meta = json.loads(Path(f"{args.v1_prefix}.json").read_text())
    v2_meta = json.loads(Path(f"{args.v2_prefix}.json").read_text())
    v1_csv = Path(f"{args.v1_prefix}.csv")
    v2_csv = Path(f"{args.v2_prefix}.csv")

    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    plot_metrics_comparison(v1_meta, v2_meta, assets_dir / "06_qwen_model_metrics.png")
    plot_confusion_matrices(v1_meta, v2_meta, assets_dir / "07_qwen_confusion_matrices.png")
    plot_paired_outcomes(v1_csv, v2_csv, assets_dir / "08_qwen_paired_outcomes.png")
    plot_family_breakdown(v1_meta, v2_meta, assets_dir / "09_qwen_family_breakdown.png")
    print(f"Generated 4 Milestone 1b figures in {assets_dir}")


if __name__ == "__main__":
    main()
