import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parents[3]
M1_REPORTS = REPO_ROOT / "milestones" / "milestone1" / "data" / "reports"
M2_REPORTS = REPO_ROOT / "milestones" / "milestone2" / "data" / "reports"

DEFAULT_RUNS = [
    ("0-shot (v2)", M1_REPORTS / "zero_shot_qwen2.5-3b-instruct_v2_validation"),
    (
        "4-shot (balanced)",
        M2_REPORTS / "few_shot_qwen2.5-3b-instruct_v2_balanced_4shot_validation",
    ),
    (
        "8-shot (balanced)",
        M2_REPORTS / "few_shot_qwen2.5-3b-instruct_v2_balanced_8shot_validation",
    ),
    (
        "8-shot (random s42)",
        M2_REPORTS / "few_shot_qwen2.5-3b-instruct_v2_random_seed42_8shot_validation",
    ),
]


def load_run(prefix: Path) -> tuple[pd.DataFrame, dict]:
    csv_path = Path(f"{prefix}.csv")
    json_path = Path(f"{prefix}.json")
    df = pd.read_csv(csv_path)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    assert df["id"].is_unique, f"Duplicate IDs in {csv_path}"
    assert (df["status"] == "success").all(), f"Incomplete rows in {csv_path}"
    return df, meta


def compute_metrics(df: pd.DataFrame) -> dict:
    y = df["actual"].astype(int)
    p = df["predicted"].astype(int)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else None
    return {
        "samples": len(df),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / len(df) if len(df) else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
    }


def compute_operational(
    df: pd.DataFrame,
    ref_input_price_per_1m: float = 0.15,
    ref_output_price_per_1m: float = 0.60,
) -> dict:
    lat = df["latency_seconds"].astype(float).to_numpy()
    tok_in = df["input_tokens"].astype(int).to_numpy()
    tok_out = df["output_tokens"].astype(int).to_numpy()
    mean_lat = float(lat.mean())
    mean_in = float(tok_in.mean())
    mean_out = float(tok_out.mean())
    ref_cost_1k = (
        (mean_in * 1000.0 / 1_000_000.0) * ref_input_price_per_1m
        + (mean_out * 1000.0 / 1_000_000.0) * ref_output_price_per_1m
    )
    return {
        "mean_input_tokens": round(mean_in, 2),
        "total_input_tokens": int(tok_in.sum()),
        "mean_output_tokens": round(mean_out, 2),
        "total_output_tokens": int(tok_out.sum()),
        "mean_latency_s": round(mean_lat, 4),
        "p50_latency_s": round(float(np.percentile(lat, 50)), 4),
        "p95_latency_s": round(float(np.percentile(lat, 95)), 4),
        "total_latency_s": round(float(lat.sum()), 2),
        "compute_min_per_1k": round(mean_lat * 1000.0 / 60.0, 2),
        "direct_api_cost_per_1k_usd": 0.0,
        "ref_api_equiv_cost_per_1k_usd": round(ref_cost_1k, 4),
    }


def paired_compare(base_df: pd.DataFrame, cand_df: pd.DataFrame) -> dict:
    merged = base_df[
        ["id", "actual", "predicted", "attack_family", "technique"]
    ].merge(
        cand_df[["id", "actual", "predicted"]],
        on="id",
        suffixes=("_base", "_cand"),
        validate="one_to_one",
    )
    assert (merged["actual_base"] == merged["actual_cand"]).all()
    c_base = merged["predicted_base"] == merged["actual_base"]
    c_cand = merged["predicted_cand"] == merged["actual_cand"]
    fixed = (~c_base) & c_cand
    regressed = c_base & (~c_cand)
    both_correct = c_base & c_cand
    both_wrong = (~c_base) & (~c_cand)

    b = int(fixed.sum())
    c = int(regressed.sum())
    p_value = (
        float(binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue)
        if (b + c) > 0
        else 1.0
    )

    merged["fixed"] = fixed
    merged["regressed"] = regressed
    by_family = (
        merged.groupby("attack_family")[["fixed", "regressed"]]
        .sum()
        .astype(int)
        .to_dict(orient="index")
    )

    return {
        "both_correct": int(both_correct.sum()),
        "both_wrong": int(both_wrong.sum()),
        "fixed": b,
        "regressed": c,
        "net_gain": b - c,
        "mcnemar_exact_p": p_value,
        "by_family": by_family,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare Milestone 2 few-shot ICL runs against M1b 0-shot."
    )
    parser.add_argument(
        "--output-json",
        default=str(M2_REPORTS / "icl_comparison_summary.json"),
    )
    args = parser.parse_args()

    loaded = []
    for label, prefix in DEFAULT_RUNS:
        if Path(f"{prefix}.csv").exists():
            df, meta = load_run(prefix)
            loaded.append((label, df, meta))
        else:
            print(f"[SKIP] Missing run artifact: {prefix}.csv")

    if len(loaded) < 2:
        raise RuntimeError("Need at least 2 completed runs to compare.")

    base_ids = set(loaded[0][1]["id"])
    for label, df, _ in loaded[1:]:
        assert set(df["id"]) == base_ids, f"Sample ID mismatch in {label}"

    summary = {"runs": {}, "paired_comparisons": {}, "subgroups": {}, "case_studies": {}}
    base_mean_in = None

    print("\n================ AGGREGATE METRICS ================")
    for label, df, meta in loaded:
        m = compute_metrics(df)
        ops = compute_operational(df)
        if base_mean_in is None:
            base_mean_in = ops["mean_input_tokens"]
        ops["input_token_multiplier"] = round(
            ops["mean_input_tokens"] / base_mean_in, 2
        )
        summary["runs"][label] = {
            "num_shots": int(meta.get("num_shots", 0)),
            "strategy": meta.get("manifest_strategy", "zero_shot"),
            "metrics": m,
            "operational": ops,
        }
        print(
            f"{label:20s} | Acc={m['accuracy']:.4f} Prec={m['precision']:.4f} "
            f"Rec={m['recall']:.4f} F1={m['f1']:.4f} FPR={m['fpr']:.4f} "
            f"(TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}) | "
            f"in_tok={ops['mean_input_tokens']:.1f} ({ops['input_token_multiplier']:.2f}x) "
            f"p50={ops['p50_latency_s']:.3f}s p95={ops['p95_latency_s']:.3f}s "
            f"ref$/1k=${ops['ref_api_equiv_cost_per_1k_usd']:.4f}"
        )

    print("\n================ PAIRED COMPARISONS ================")
    base_label, base_df, _ = loaded[0]
    for label, df, _ in loaded[1:]:
        key = f"{base_label} -> {label}"
        comp = paired_compare(base_df, df)
        summary["paired_comparisons"][key] = comp
        print(
            f"{key:42s} | fixed={comp['fixed']:2d} regressed={comp['regressed']:2d} "
            f"net={comp['net_gain']:+3d} both_ok={comp['both_correct']} "
            f"both_err={comp['both_wrong']} p={comp['mcnemar_exact_p']:.4g}"
        )

    # Also compare consecutive/strategy pairs if present
    by_label = {label: df for label, df, _ in loaded}
    extra_pairs = [
        ("4-shot (balanced)", "8-shot (balanced)"),
        ("8-shot (random s42)", "8-shot (balanced)"),
    ]
    for a, b in extra_pairs:
        if a in by_label and b in by_label:
            key = f"{a} -> {b}"
            comp = paired_compare(by_label[a], by_label[b])
            summary["paired_comparisons"][key] = comp
            print(
                f"{key:42s} | fixed={comp['fixed']:2d} regressed={comp['regressed']:2d} "
                f"net={comp['net_gain']:+3d} both_ok={comp['both_correct']} "
                f"both_err={comp['both_wrong']} p={comp['mcnemar_exact_p']:.4g}"
            )

    print("\n================ SUBGROUP BREAKDOWN ================")
    families = sorted(base_df["attack_family"].unique())
    for fam in families:
        summary["subgroups"][fam] = {}
        line = [f"{fam:30s}"]
        for label, df, _ in loaded:
            sub = df[df["attack_family"] == fam]
            m = compute_metrics(sub)
            summary["subgroups"][fam][label] = m
            if fam == "benign":
                line.append(f"{label}: FPR={m['fpr']:.4f} ({m['fp']} FP)")
            else:
                line.append(f"{label}: Rec={m['recall']:.4f} ({m['fn']} FN)")
        print(" | ".join(line))

    print("\n================ KEY TECHNIQUES ================")
    key_techs = [
        ("benign", "quoting_attack_for_analysis"),
        ("benign", "hard_negative"),
        ("approval_authority_escalation", "document_metadata_forged_stamp"),
        ("approval_authority_escalation", "ticket_comment_status_spoof"),
        ("memory_trust_boundary", "browser_agent_persist"),
        ("memory_trust_boundary", "tool_output_poison"),
        ("tool_function_abuse", "support_ticket_indirect_exfil"),
    ]
    summary["key_techniques"] = {}
    for fam, tech in key_techs:
        tkey = f"{fam}::{tech}"
        summary["key_techniques"][tkey] = {}
        line = [f"{tech:32s}"]
        for label, df, _ in loaded:
            sub = df[(df["attack_family"] == fam) & (df["technique"] == tech)]
            m = compute_metrics(sub)
            summary["key_techniques"][tkey][label] = m
            errs = m["fp"] if fam == "benign" else m["fn"]
            line.append(f"{label}: err={errs}/{m['samples']}")
        print(" | ".join(line))

    print("\n================ CASE STUDY IDS (11, 137, 200) ================")
    for cid in [11, 137, 200]:
        row_info = {}
        for label, df, _ in loaded:
            rec = df[df["id"] == cid].iloc[0]
            row_info[label] = {
                "actual": int(rec["actual"]),
                "predicted": int(rec["predicted"]),
                "correct": bool(int(rec["actual"]) == int(rec["predicted"])),
            }
        summary["case_studies"][str(cid)] = row_info
        preds = " | ".join(
            f"{lbl}: pred={info['predicted']} ({'OK' if info['correct'] else 'ERR'})"
            for lbl, info in row_info.items()
        )
        print(f"ID {cid:3d} (actual={row_info[base_label]['actual']}): {preds}")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nSaved summary JSON: {out_path}")


if __name__ == "__main__":
    main()
