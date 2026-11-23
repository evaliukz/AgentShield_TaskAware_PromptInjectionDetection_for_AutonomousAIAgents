import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from milestones.milestone3.scripts.evaluate_lora import calculate_metrics

MILESTONE_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {"id", "actual", "predicted", "status", "attack_family"}


def load_predictions(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    csv_path = Path(csv_path)
    metadata_path = csv_path.with_suffix(".json")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {csv_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Prediction metadata not found: {metadata_path}")

    frame = pd.read_csv(csv_path, dtype={"id": "string"})
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError(f"{csv_path} has a missing or duplicate sample ID.")
    if frame["attack_family"].isna().any():
        raise ValueError(f"{csv_path} has rows without an attack family.")
    if not (frame["status"] == "success").all():
        raise ValueError(f"{csv_path} contains unsuccessful predictions.")
    for column in ("actual", "predicted"):
        values = pd.to_numeric(frame[column], errors="raise")
        if values.isna().any() or not values.isin((0, 1)).all():
            raise ValueError(f"{csv_path} has invalid {column} labels.")
        frame[column] = values.astype(int)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return frame, metadata


def validate_matching_runs(icl_metadata: dict, lora_metadata: dict) -> None:
    split = icl_metadata.get("split")
    if split not in ("validation", "test"):
        raise ValueError("The ICL run must use validation or the final test split.")
    if lora_metadata.get("split") != split:
        raise ValueError("The runs use different data splits.")
    if split == "test" and (
        icl_metadata.get("test_split_accessed") is not True
        or lora_metadata.get("test_split_accessed") is not True
    ):
        raise ValueError("Test runs must explicitly record test-split access.")
    if icl_metadata.get("backend") != "huggingface":
        raise ValueError("The ICL comparator must use the Hugging Face backend.")
    if (
        icl_metadata.get("manifest_name") != "random_seed42_8shot"
        or icl_metadata.get("num_shots") != 8
        or icl_metadata.get("prompt_version") != "v2"
    ):
        raise ValueError("The ICL run is not the frozen M2 8-shot v2 candidate.")
    if lora_metadata.get("prompt_version") != "ZERO_SHOT_V2":
        raise ValueError("The LoRA run does not use ZERO_SHOT_V2.")
    if icl_metadata.get("dataset") != lora_metadata.get("dataset"):
        raise ValueError("The runs use different datasets.")

    if icl_metadata.get("dataset_revision") != lora_metadata.get(
        "dataset_revision"
    ):
        raise ValueError("The runs use different dataset revisions.")
    if icl_metadata.get("sample_count") != lora_metadata.get("sample_count"):
        raise ValueError("The runs have different sample counts.")
    expected_count = 545 if split == "test" else 467
    if icl_metadata.get("sample_count") != expected_count:
        raise ValueError(
            f"The paired comparison must cover all {expected_count} {split} rows."
        )
    if split == "test" and (
        not icl_metadata.get("sample_ids_sha256")
        or icl_metadata.get("sample_ids_sha256")
        != lora_metadata.get("sample_ids_sha256")
    ):
        raise ValueError("The test runs do not attest to the same sample IDs.")
    if icl_metadata.get("model") != lora_metadata.get("base_model"):
        raise ValueError("The runs use different base models.")

    icl_runtime = icl_metadata.get("runtime", {})
    lora_runtime = lora_metadata.get("runtime", {})
    if icl_runtime.get("model_revision") != lora_metadata.get(
        "base_model_revision"
    ):
        raise ValueError("The runs use different base-model revisions.")

    icl_prompt_hash = icl_metadata.get(
        "effective_prompt_sha256",
        icl_runtime.get("effective_prompt_sha256"),
    )
    if icl_prompt_hash != lora_metadata.get("prompt_sha256"):
        raise ValueError("The runs use different effective prompts.")

    icl_tokens = icl_metadata.get("decoding", {}).get("max_new_tokens")
    lora_tokens = lora_metadata.get("decoding", {}).get("max_new_tokens")
    if icl_tokens != lora_tokens:
        raise ValueError("The runs use different max_new_tokens settings.")
    if icl_runtime.get("dtype") != lora_runtime.get("dtype"):
        raise ValueError("The runs use different inference dtypes.")
    if split == "test" and icl_runtime.get("device") != lora_runtime.get("device"):
        raise ValueError("The test runs use different inference devices.")


def _bootstrap_metric_samples(
    actual: np.ndarray,
    icl_predicted: np.ndarray,
    lora_predicted: np.ndarray,
    *,
    resamples: int = 10_000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    sample_indexes = rng.integers(0, len(actual), size=(resamples, len(actual)))
    sampled_actual = actual[sample_indexes]
    metrics_by_model = {}

    for model_name, predictions in (
        ("icl", icl_predicted),
        ("lora", lora_predicted),
    ):
        sampled_predictions = predictions[sample_indexes]
        tp = np.sum((sampled_actual == 1) & (sampled_predictions == 1), axis=1)
        fp = np.sum((sampled_actual == 0) & (sampled_predictions == 1), axis=1)
        tn = np.sum((sampled_actual == 0) & (sampled_predictions == 0), axis=1)
        fn = np.sum((sampled_actual == 1) & (sampled_predictions == 0), axis=1)

        precision = np.divide(
            tp, tp + fp, out=np.zeros(resamples, dtype=float), where=(tp + fp) > 0
        )
        recall = np.divide(
            tp, tp + fn, out=np.zeros(resamples, dtype=float), where=(tp + fn) > 0
        )
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros(resamples, dtype=float),
            where=(precision + recall) > 0,
        )
        fpr = np.divide(
            fp, fp + tn, out=np.full(resamples, np.nan), where=(fp + tn) > 0
        )
        accuracy = (tp + tn) / len(actual)
        metrics_by_model[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
        }

    return {
        metric: {
            "icl": metrics_by_model["icl"][metric],
            "lora": metrics_by_model["lora"][metric],
            "delta_lora_minus_icl": (
                metrics_by_model["lora"][metric]
                - metrics_by_model["icl"][metric]
            ),
        }
        for metric in ("accuracy", "precision", "recall", "f1", "fpr")
    }


def _summarize_bootstrap(samples: dict) -> dict:
    return {
        metric: {
            model: (
                [
                    float(np.nanpercentile(values, 2.5)),
                    float(np.nanpercentile(values, 97.5)),
                ]
                if np.isfinite(values).any()
                else None
            )
            for model, values in metric_samples.items()
        }
        for metric, metric_samples in samples.items()
    }


def calculate_uncertainty(merged: pd.DataFrame) -> dict:
    metric_names = ("accuracy", "precision", "recall", "f1", "fpr")
    overall_samples = _bootstrap_metric_samples(
        merged["actual_icl"].to_numpy(dtype=int),
        merged["predicted_icl"].to_numpy(dtype=int),
        merged["predicted_lora"].to_numpy(dtype=int),
    )
    family_samples = {}
    for index, (family, family_rows) in enumerate(
        merged.groupby("attack_family_icl", dropna=False)
    ):
        family_samples[str(family)] = _summarize_bootstrap(
            _bootstrap_metric_samples(
                family_rows["actual_icl"].to_numpy(dtype=int),
                family_rows["predicted_icl"].to_numpy(dtype=int),
                family_rows["predicted_lora"].to_numpy(dtype=int),
                seed=42 + index,
            )
        )
    return {
        "confidence_level": 0.95,
        "method": "paired nonparametric percentile bootstrap",
        "resamples": 10_000,
        "interval_bounds": ["lower", "upper"],
        "overall": _summarize_bootstrap(overall_samples),
        "attack_family": family_samples,
    }


def compare_predictions(
    icl_frame: pd.DataFrame,
    lora_frame: pd.DataFrame,
) -> dict:
    prediction_columns = ["id", "actual", "predicted", "attack_family"]
    if set(icl_frame["id"]) != set(lora_frame["id"]):
        raise ValueError("The runs do not contain the same sample IDs.")

    merged = icl_frame[prediction_columns].merge(
        lora_frame[prediction_columns],
        on="id",
        suffixes=("_icl", "_lora"),
        validate="one_to_one",
    )
    if not (merged["actual_icl"] == merged["actual_lora"]).all():
        raise ValueError("Ground-truth labels differ between runs.")
    if not (merged["attack_family_icl"] == merged["attack_family_lora"]).all():
        raise ValueError("Attack-family labels differ between runs.")

    icl_rows = [
        {"actual": row.actual_icl, "predicted": row.predicted_icl}
        for row in merged.itertuples()
    ]
    lora_rows = [
        {"actual": row.actual_lora, "predicted": row.predicted_lora}
        for row in merged.itertuples()
    ]
    icl_metrics = calculate_metrics(icl_rows)
    lora_metrics = calculate_metrics(lora_rows)

    correct_icl = merged["predicted_icl"] == merged["actual_icl"]
    correct_lora = merged["predicted_lora"] == merged["actual_lora"]
    fixed = (~correct_icl) & correct_lora
    regressed = correct_icl & (~correct_lora)
    fixed_count = int(fixed.sum())
    regressed_count = int(regressed.sum())
    discordant = fixed_count + regressed_count

    by_family = {}
    for family, family_rows in merged.groupby(
        "attack_family_icl", dropna=False
    ):
        icl_family = [
            {"actual": row.actual_icl, "predicted": row.predicted_icl}
            for row in family_rows.itertuples()
        ]
        lora_family = [
            {"actual": row.actual_lora, "predicted": row.predicted_lora}
            for row in family_rows.itertuples()
        ]
        icl_family_metrics = calculate_metrics(icl_family)
        lora_family_metrics = calculate_metrics(lora_family)
        by_family[str(family)] = {
            "icl": icl_family_metrics,
            "lora": lora_family_metrics,
            "delta_lora_minus_icl": {
                metric: (
                    lora_family_metrics[metric] - icl_family_metrics[metric]
                    if lora_family_metrics[metric] is not None
                    and icl_family_metrics[metric] is not None
                    else None
                )
                for metric in ("f1", "fpr", "recall")
            },
        }

    changed_predictions = [
        {
            "id": str(row.id),
            "actual": int(row.actual_icl),
            "predicted_icl": int(row.predicted_icl),
            "predicted_lora": int(row.predicted_lora),
            "attack_family": str(row.attack_family_icl),
        }
        for row in merged.itertuples()
        if row.predicted_icl != row.predicted_lora
    ]
    return {
        "sample_count": len(merged),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(merged["id"].astype(str))).encode("utf-8")
        ).hexdigest(),
        "icl_metrics": icl_metrics,
        "lora_metrics": lora_metrics,
        "delta_lora_minus_icl": {
            metric: (
                lora_metrics[metric] - icl_metrics[metric]
                if lora_metrics[metric] is not None
                and icl_metrics[metric] is not None
                else None
            )
            for metric in ("f1", "fpr", "recall")
        },
        "paired_errors": {
            "fixed_by_lora": fixed_count,
            "regressed_by_lora": regressed_count,
            "both_correct": int((correct_icl & correct_lora).sum()),
            "both_wrong": int((~correct_icl & ~correct_lora).sum()),
            "mcnemar_exact_p": (
                float(
                    binomtest(
                        min(fixed_count, regressed_count),
                        n=discordant,
                        p=0.5,
                        alternative="two-sided",
                    ).pvalue
                )
                if discordant
                else 1.0
            ),
        },
        "uncertainty": calculate_uncertainty(merged),
        "attack_family_metrics": by_family,
        "changed_predictions": changed_predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare frozen M2 ICL and LoRA predictions on the same split."
    )
    parser.add_argument("--icl-csv", type=Path, required=True)
    parser.add_argument("--lora-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    icl_frame, icl_metadata = load_predictions(args.icl_csv)
    lora_frame, lora_metadata = load_predictions(args.lora_csv)
    validate_matching_runs(icl_metadata, lora_metadata)
    output_path = args.output or (
        MILESTONE_ROOT
        / "data"
        / "reports"
        / f"lora_vs_icl_{icl_metadata['split']}.json"
    )
    if len(icl_frame) != icl_metadata["sample_count"]:
        raise ValueError("The ICL CSV row count does not match its metadata.")
    if len(lora_frame) != lora_metadata["sample_count"]:
        raise ValueError("The LoRA CSV row count does not match its metadata.")
    if icl_metadata.get("failure_count", 0) or lora_metadata.get(
        "failure_count", 0
    ):
        raise ValueError("Cannot compare runs that recorded failed predictions.")
    summary = compare_predictions(icl_frame, lora_frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("ICL:", json.dumps(summary["icl_metrics"], indent=2))
    print("LoRA:", json.dumps(summary["lora_metrics"], indent=2))
    print("LoRA minus ICL:", json.dumps(summary["delta_lora_minus_icl"], indent=2))
    print("Paired errors:", json.dumps(summary["paired_errors"], indent=2))
    print("Attack-family metrics:", json.dumps(summary["attack_family_metrics"], indent=2))
    print("95% paired bootstrap intervals:", json.dumps(summary["uncertainty"], indent=2))
    print(f"Summary: {output_path}")


if __name__ == "__main__":
    main()
