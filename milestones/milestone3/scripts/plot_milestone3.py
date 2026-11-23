import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MILESTONE_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = MILESTONE_ROOT / "data" / "reports" / "lora_vs_icl_test.json"
ASSET_DIR = MILESTONE_ROOT / "assets"
COLORS = {"ICL (8-shot)": "#4C78A8", "LoRA": "#F58518"}


def load_results() -> dict:
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(f"Test comparison report not found: {REPORT_PATH}")
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def save_overall_metrics(results: dict) -> None:
    metrics = ("accuracy", "precision", "recall", "f1", "fpr")
    model_keys = (("ICL (8-shot)", "icl_metrics"), ("LoRA", "lora_metrics"))
    fig, axes = plt.subplots(1, len(metrics), figsize=(17, 4), sharey=True)

    for axis, metric in zip(axes, metrics):
        y_positions = np.array([1, 0])
        values = []
        errors = [[], []]
        for label, key in model_keys:
            value = results[key][metric]
            interval = results["uncertainty"]["overall"][metric][
                label.lower().replace(" (8-shot)", "")
            ]
            values.append(value)
            errors[0].append(value - interval[0])
            errors[1].append(interval[1] - value)

        for index, (label, _) in enumerate(model_keys):
            axis.barh(
                y_positions[index],
                values[index],
                color=COLORS[label],
                height=0.56,
                xerr=np.array(
                    [[errors[0][index]], [errors[1][index]]]
                ),
                capsize=4,
                error_kw={"elinewidth": 1.4, "capthick": 1.4},
            )
            if values[index] > 0.15:
                label_x = values[index] - 0.02
                label_alignment = "right"
                label_color = "white"
            else:
                label_x = values[index] + 0.025
                label_alignment = "left"
                label_color = "black"
            axis.text(
                label_x,
                y_positions[index],
                f"{values[index]:.3f}",
                ha=label_alignment,
                va="center",
                color=label_color,
                fontsize=9,
            )

        axis.set_title("Benign FPR" if metric == "fpr" else metric.upper())
        axis.set_xlim(0, 1.14)
        axis.set_xticks(np.arange(0, 1.01, 0.2))
        axis.set_xlabel("Score (95% bootstrap CI)")
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)

    axes[0].set_yticks([1, 0], ["ICL (8-shot)", "LoRA"])
    fig.suptitle("Frozen ICL vs. LoRA on the 545-example test set", y=1.03)
    fig.text(
        0.5,
        -0.02,
        "Paired 95% percentile bootstrap intervals; candidates evaluated on identical examples.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "01_test_overall_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_family_metrics(results: dict) -> None:
    families = [
        "approval_authority_escalation",
        "memory_trust_boundary",
        "rag_document_embedded",
        "tool_function_abuse",
    ]
    labels = [
        "Approval / authority",
        "Memory trust boundary",
        "RAG document",
        "Tool function abuse",
    ]
    x = np.arange(len(families))
    width = 0.34
    fig, axis = plt.subplots(figsize=(10, 5))

    for model_name, model_key, offset in (
        ("ICL (8-shot)", "icl", -width / 2),
        ("LoRA", "lora", width / 2),
    ):
        values = [
            results["attack_family_metrics"][family][model_key]["f1"]
            for family in families
        ]
        intervals = [
            results["uncertainty"]["attack_family"][family]["f1"][model_key]
            for family in families
        ]
        errors = np.array(
            [
                [value - interval[0] for value, interval in zip(values, intervals)],
                [interval[1] - value for value, interval in zip(values, intervals)],
            ]
        )
        axis.bar(
            x + offset,
            values,
            width,
            label=model_name,
            color=COLORS[model_name],
            yerr=errors,
            capsize=3,
            error_kw={"elinewidth": 1.2},
        )
        for position, value in zip(x + offset, values):
            axis.text(position, min(value + 0.015, 1.045), f"{value:.3f}", ha="center", fontsize=8)

    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.09)
    axis.set_ylabel("Attack-family F1 (95% bootstrap CI)")
    axis.set_title("Test F1 by attack family")
    axis.legend(frameon=False, loc="lower right")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    fig.text(
        0.5,
        0.01,
        "Family sample sizes: 84, 56, 84, and 56 respectively.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(ASSET_DIR / "02_test_attack_families.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_paired_outcomes(results: dict) -> None:
    outcomes = results["paired_errors"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    groups = (
        (axes[0], ["Fixed by LoRA", "Regressed"], [outcomes["fixed_by_lora"], outcomes["regressed_by_lora"]], ["#54A24B", "#E45756"]),
        (axes[1], ["Both correct", "Both wrong"], [outcomes["both_correct"], outcomes["both_wrong"]], ["#9D9DA1", "#BAB0AC"]),
    )
    for axis, labels, values, colors in groups:
        bars = axis.bar(labels, values, color=colors, width=0.65)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.025,
                str(value),
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        axis.set_ylim(0, max(values) * 1.18 if max(values) else 1)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Number of paired test examples")
    fig.suptitle(
        f"Paired correctness outcomes (n={results['sample_count']}; "
        f"exact McNemar p={outcomes['mcnemar_exact_p']:.2e})"
    )
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "03_test_paired_outcomes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    save_overall_metrics(results)
    save_family_metrics(results)
    save_paired_outcomes(results)
    print(f"Saved Milestone 3 figures to {ASSET_DIR}")


if __name__ == "__main__":
    main()
