import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MILESTONE_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = MILESTONE_ROOT / "data" / "reports"
ASSETS_DIR = MILESTONE_ROOT / "assets"

NAVY = "#1B2A4A"
SLATE = "#6C7A89"
BLUE = "#3B77A6"
TEAL = "#238368"
AMBER = "#CE8032"
CRIMSON = "#BA5152"

RUN_ORDER = [
    "0-shot (v2)",
    "4-shot (balanced)",
    "8-shot (balanced)",
    "8-shot (random s42)",
]
RUN_COLORS = {
    "0-shot (v2)": SLATE,
    "4-shot (balanced)": BLUE,
    "8-shot (balanced)": TEAL,
    "8-shot (random s42)": AMBER,
}


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def available_runs(summary: dict) -> list[str]:
    return [r for r in RUN_ORDER if r in summary["runs"]]


def plot_shot_scaling(summary: dict, out_path: Path):
    runs = available_runs(summary)
    labels = ["Accuracy", "Precision", "Recall", "F1", "FPR"]
    keys = ["accuracy", "precision", "recall", "f1", "fpr"]

    x = np.arange(len(labels))
    n = len(runs)
    width = 0.78 / n

    fig, ax = plt.subplots(figsize=(11, 5.0), dpi=200)
    for idx, run in enumerate(runs):
        vals = [summary["runs"][run]["metrics"][k] * 100 for k in keys]
        offset = (idx - (n - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=run,
            color=RUN_COLORS[run],
        )
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.8,
            )

    ax.set_ylabel("Percent (%)", fontsize=11)
    ax.set_title(
        "Milestone 2 ICL validation metrics on Qwen2.5-3B-Instruct (n=467)",
        fontsize=12.5,
        fontweight="bold",
        color=NAVY,
        pad=30,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 114)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.09),
        ncol=len(runs),
        fontsize=9.5,
    )
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(summary: dict, out_path: Path):
    runs = available_runs(summary)
    fig, axes = plt.subplots(1, len(runs), figsize=(3.8 * len(runs), 3.9), dpi=200)
    if len(runs) == 1:
        axes = [axes]
    fig.suptitle(
        "Qwen2.5-3B-Instruct confusion matrices across ICL configurations (n=467)",
        fontsize=13,
        fontweight="bold",
        color=NAVY,
        y=1.03,
    )

    vmax = max(summary["runs"][r]["metrics"]["tp"] for r in runs)
    for ax, run in zip(axes, runs):
        m = summary["runs"][run]["metrics"]
        mat = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(run, fontsize=11, fontweight="bold", pad=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Benign", "Injection"], fontsize=9.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Benign", "Injection"], fontsize=9.5)
        ax.set_xlabel("Predicted", fontsize=9.5)
        if ax is axes[0]:
            ax.set_ylabel("Actual", fontsize=9.5)
        style_axes(ax)

        thresh = vmax * 0.45
        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=15,
                    fontweight="bold",
                    color="white" if val > thresh else NAVY,
                )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_family_breakdown(summary: dict, out_path: Path):
    runs = available_runs(summary)
    attack_fams = [
        "approval_authority_escalation",
        "memory_trust_boundary",
        "rag_document_embedded",
        "tool_function_abuse",
    ]
    pretty = [
        "Approval / Authority\nEscalation (n=98)",
        "Memory Trust\nBoundary (n=56)",
        "RAG Document\nEmbedded (n=42)",
        "Tool / Function\nAbuse (n=112)",
        "Benign Specificity\n(1 − FPR, n=159)",
    ]

    x = np.arange(len(pretty))
    n = len(runs)
    width = 0.78 / n

    fig, ax = plt.subplots(figsize=(11.5, 4.9), dpi=200)
    for idx, run in enumerate(runs):
        scores = [
            summary["subgroups"][fam][run]["recall"] * 100 for fam in attack_fams
        ] + [(1.0 - summary["subgroups"]["benign"][run]["fpr"]) * 100]
        offset = (idx - (n - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            scores,
            width,
            label=run,
            color=RUN_COLORS[run],
        )
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.3,
            )

    ax.set_ylabel("Subgroup recall / specificity (%)", fontsize=10.5)
    ax.set_title(
        "Subgroup performance across attack families and benign specificity (Qwen2.5-3B-Instruct)",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
        pad=30,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(pretty, fontsize=9.5)
    ax.set_ylim(0, 115)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.09),
        ncol=len(runs),
        fontsize=9.5,
    )
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cost_latency_tradeoff(summary: dict, out_path: Path):
    runs = available_runs(summary)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=200)
    fig.suptitle(
        "Accuracy–token–latency trade-off across ICL shot counts (Apple M3 Pro MPS, batch=1)",
        fontsize=12.5,
        fontweight="bold",
        color=NAVY,
        y=1.03,
    )

    # Panel 1: F1 and FPR vs Mean Input Tokens
    toks = [summary["runs"][r]["operational"]["mean_input_tokens"] for r in runs]
    f1s = [summary["runs"][r]["metrics"]["f1"] * 100 for r in runs]
    fprs = [summary["runs"][r]["metrics"]["fpr"] * 100 for r in runs]

    f1_offsets = {
        "0-shot (v2)": (0, 9),
        "4-shot (balanced)": (0, 9),
        "8-shot (balanced)": (4, -24),
        "8-shot (random s42)": (-18, 9),
    }
    fpr_offsets = {
        "0-shot (v2)": (0, -16),
        "4-shot (balanced)": (0, -16),
        "8-shot (balanced)": (14, -15),
        "8-shot (random s42)": (-22, 9),
    }

    for r, t, f1, fpr in zip(runs, toks, f1s, fprs):
        c = RUN_COLORS[r]
        ax1.scatter([t], [f1], s=95, color=c, marker="o", zorder=4)
        ax1.annotate(
            f"{r}\nF1 {f1:.1f}%",
            xy=(t, f1),
            xytext=f1_offsets.get(r, (0, 8)),
            textcoords="offset points",
            ha="center",
            fontsize=8.0,
            color=NAVY,
        )
        ax1.scatter([t], [fpr], s=85, color=c, marker="s", zorder=4)
        ax1.annotate(
            f"FPR {fpr:.1f}%",
            xy=(t, fpr),
            xytext=fpr_offsets.get(r, (0, -14)),
            textcoords="offset points",
            ha="center",
            fontsize=7.8,
            color=CRIMSON,
        )

    # Connect balanced shot-scaling path (0 -> 4 -> 8)
    bal_runs = [r for r in ["0-shot (v2)", "4-shot (balanced)", "8-shot (balanced)"] if r in runs]
    if len(bal_runs) >= 2:
        bt = [summary["runs"][r]["operational"]["mean_input_tokens"] for r in bal_runs]
        bf1 = [summary["runs"][r]["metrics"]["f1"] * 100 for r in bal_runs]
        bfpr = [summary["runs"][r]["metrics"]["fpr"] * 100 for r in bal_runs]
        ax1.plot(bt, bf1, "--", color=TEAL, alpha=0.6, linewidth=1.4, label="F1 (balanced path)")
        ax1.plot(bt, bfpr, ":", color=CRIMSON, alpha=0.6, linewidth=1.4, label="FPR (balanced path)")

    ax1.set_xlabel("Mean input tokens per request", fontsize=10)
    ax1.set_ylabel("Percent (%)", fontsize=10)
    ax1.set_title("Quality vs. prompt token budget", fontsize=11, fontweight="bold", color=NAVY)
    ax1.set_ylim(-6, 116)
    ax1.set_xlim(min(toks) * 0.72, max(toks) * 1.18)
    ax1.legend(frameon=False, loc="center left", fontsize=8.5)
    style_axes(ax1)

    # Panel 2: p50 & p95 latency bars with ref $/1k annotation
    x = np.arange(len(runs))
    w = 0.34
    p50s = [summary["runs"][r]["operational"]["p50_latency_s"] for r in runs]
    p95s = [summary["runs"][r]["operational"]["p95_latency_s"] for r in runs]
    costs = [summary["runs"][r]["operational"]["ref_api_equiv_cost_per_1k_usd"] for r in runs]

    b1 = ax2.bar(x - w / 2, p50s, w, label="p50 latency (s)", color=BLUE)
    b2 = ax2.bar(x + w / 2, p95s, w, label="p95 latency (s)", color=TEAL)

    for idx, (bar1, bar2, cost) in enumerate(zip(b1, b2, costs)):
        ax2.annotate(
            f"{bar1.get_height():.2f}s",
            xy=(bar1.get_x() + bar1.get_width() / 2, bar1.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8.0,
        )
        ax2.annotate(
            f"{bar2.get_height():.2f}s",
            xy=(bar2.get_x() + bar2.get_width() / 2, bar2.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8.0,
        )
        ax2.annotate(
            f"Ref ${cost:.3f}/1k",
            xy=(x[idx], max(bar1.get_height(), bar2.get_height())),
            xytext=(0, 16),
            textcoords="offset points",
            ha="center",
            fontsize=7.8,
            color=SLATE,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels([r.replace(" (", "\n(") for r in runs], fontsize=8.8)
    ax2.set_ylabel("Seconds per request", fontsize=10)
    ax2.set_title("Per-request MPS latency & token-equiv cost", fontsize=11, fontweight="bold", color=NAVY)
    ax2.set_ylim(0, max(p95s) * 1.32)
    ax2.legend(frameon=False, loc="upper left", fontsize=9)
    style_axes(ax2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Milestone 2 PNG figures.")
    parser.add_argument(
        "--summary-json",
        default=str(REPORTS_DIR / "icl_comparison_summary.json"),
    )
    parser.add_argument("--assets-dir", default=str(ASSETS_DIR))
    args = parser.parse_args()

    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    out_dir = Path(args.assets_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_shot_scaling(summary, out_dir / "01_icl_shot_scaling.png")
    plot_confusion_matrices(summary, out_dir / "02_icl_confusion_matrices.png")
    plot_family_breakdown(summary, out_dir / "03_icl_family_breakdown.png")
    plot_cost_latency_tradeoff(summary, out_dir / "04_icl_latency_tokens_tradeoff.png")
    print(f"Generated 4 Milestone 2 figures in {out_dir}")


if __name__ == "__main__":
    main()
