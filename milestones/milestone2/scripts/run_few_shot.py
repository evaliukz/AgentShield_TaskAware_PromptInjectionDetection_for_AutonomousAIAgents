import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
import numpy as np

from milestones.milestone1.src.detectors.prompts import (
    ZERO_SHOT_V1,
    ZERO_SHOT_V2,
)
from milestones.milestone2.src.icl.example_selector import (
    load_manifest,
    normalize_text_hash,
)
from milestones.milestone2.src.icl.dataset import (
    DATASET_NAME,
    DATASET_REVISION,
    load_benchmark_split,
)

MILESTONE_ROOT = Path(__file__).resolve().parents[1]
PROMPTS = {
    "v1": ZERO_SHOT_V1,
    "v2": ZERO_SHOT_V2,
}
DEFAULT_MODELS = {
    "huggingface": "Qwen/Qwen2.5-3B-Instruct",
    "openai": "gpt-4o-mini",
}
CSV_FIELDS = [
    "id",
    "actual",
    "predicted",
    "status",
    "latency_seconds",
    "attack_family",
    "technique",
    "error",
    "input_tokens",
    "output_tokens",
    "raw_output",
]


def slugify_model(model_name: str) -> str:
    short_name = model_name.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9._-]+", "-", short_name).strip("-")


def calculate_metrics(rows):
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for row in rows:
        actual = int(row["actual"])
        predicted = int(row["predicted"])
        if actual == 1 and predicted == 1:
            counts["tp"] += 1
        elif actual == 0 and predicted == 1:
            counts["fp"] += 1
        elif actual == 0 and predicted == 0:
            counts["tn"] += 1
        else:
            counts["fn"] += 1

    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "samples": len(rows),
        **counts,
        "accuracy": (tp + tn) / len(rows) if rows else None,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "fpr": fp / (fp + tn) if fp + tn else None,
    }


def calculate_operational_stats(
    rows,
    backend: str,
    ref_input_price_per_1m: float = 0.15,
    ref_output_price_per_1m: float = 0.60,
):
    latencies = [
        float(row["latency_seconds"])
        for row in rows
        if row.get("latency_seconds") not in ("", None)
    ]
    input_tokens = [
        int(row["input_tokens"])
        for row in rows
        if row.get("input_tokens") not in ("", None)
    ]
    output_tokens = [
        int(row["output_tokens"])
        for row in rows
        if row.get("output_tokens") not in ("", None)
    ]

    stats = {}
    if latencies:
        arr = np.array(latencies, dtype=float)
        mean_lat = float(arr.mean())
        stats["latency_seconds"] = {
            "mean": round(mean_lat, 4),
            "p50": round(float(np.percentile(arr, 50)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
            "max": round(float(arr.max()), 4),
            "total": round(float(arr.sum()), 3),
        }
        stats["throughput_req_per_sec"] = round(1.0 / mean_lat, 3) if mean_lat > 0 else None
        stats["compute_minutes_per_1k"] = round((mean_lat * 1000.0) / 60.0, 2)

    if input_tokens or output_tokens:
        mean_in = sum(input_tokens) / len(input_tokens) if input_tokens else 0.0
        mean_out = sum(output_tokens) / len(output_tokens) if output_tokens else 0.0
        stats["tokens"] = {
            "total_input": sum(input_tokens),
            "total_output": sum(output_tokens),
            "mean_input": round(mean_in, 2) if input_tokens else None,
            "mean_output": round(mean_out, 2) if output_tokens else None,
        }
        equiv_cost_per_1k = (
            (mean_in * 1000.0 / 1_000_000.0) * ref_input_price_per_1m
            + (mean_out * 1000.0 / 1_000_000.0) * ref_output_price_per_1m
        )
        stats["cost_usd_per_1k"] = {
            "direct_api_cost": 0.0 if backend == "huggingface" else round(equiv_cost_per_1k, 4),
            "reference_api_equivalent": round(equiv_cost_per_1k, 4),
            "reference_rates_per_1m": {
                "input_usd": ref_input_price_per_1m,
                "output_usd": ref_output_price_per_1m,
            },
        }
    return stats


def build_metadata(
    rows,
    config,
    runtime_info,
    started_at,
    status,
    ref_input_price_per_1m: float = 0.15,
    ref_output_price_per_1m: float = 0.60,
):
    successful = [row for row in rows if row["status"] == "success"]
    by_family = defaultdict(list)
    by_technique = defaultdict(list)
    for row in successful:
        fam = row["attack_family"] or "unknown"
        tech = row["technique"] or "unknown"
        by_family[fam].append(row)
        by_technique[f"{fam}::{tech}"].append(row)

    return {
        **config,
        "runtime": runtime_info,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "completed_count": len(successful),
        "failure_count": len(rows) - len(successful),
        "metrics": calculate_metrics(successful),
        "operational": calculate_operational_stats(
            successful,
            backend=config["backend"],
            ref_input_price_per_1m=ref_input_price_per_1m,
            ref_output_price_per_1m=ref_output_price_per_1m,
        ),
        "attack_family_metrics": {
            fam: calculate_metrics(fam_rows)
            for fam, fam_rows in sorted(by_family.items())
        },
        "technique_metrics": {
            tech: calculate_metrics(tech_rows)
            for tech, tech_rows in sorted(by_technique.items())
        },
    }


def write_metadata(path: Path, metadata: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def read_checkpoint(csv_path: Path):
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_FIELDS:
            raise RuntimeError(
                f"Checkpoint has unexpected columns: {reader.fieldnames}"
            )
        return list(reader)


def write_checkpoint(csv_path: Path, rows):
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def create_detector(args, model: str, prompt: str, examples):
    if args.backend == "openai":
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing")
        from milestones.milestone2.src.detectors.few_shot import (
            OpenAIFewShotDetector,
        )

        return OpenAIFewShotDetector(
            examples=examples,
            model=model,
            system_prompt=prompt,
            temperature=0.0,
        )

    from milestones.milestone2.src.detectors.few_shot import (
        LocalHFFewShotDetector,
    )

    return LocalHFFewShotDetector(
        examples=examples,
        model=model,
        system_prompt=prompt,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )


def resolve_output_path(
    args, model: str, manifest_name: str, split: str = "validation"
) -> Path:
    reports_dir = MILESTONE_ROOT / "data" / "reports"
    suffix = f"_smoke{args.max_samples}" if args.max_samples else ""
    slug = slugify_model(model)
    stem = f"few_shot_{slug}_{args.prompt_version}_{manifest_name}_{split}{suffix}"
    return reports_dir / f"{stem}.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Run few-shot ICL detection on the validation split."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to versioned training-only JSON example manifest.",
    )
    parser.add_argument(
        "--backend",
        choices=sorted(DEFAULT_MODELS),
        default="huggingface",
        help="Inference backend: huggingface or openai (default: huggingface).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier (defaults: Qwen/Qwen2.5-3B-Instruct or gpt-4o-mini).",
    )
    parser.add_argument(
        "--prompt-version",
        choices=sorted(PROMPTS),
        default="v2",
        help="Base system prompt version (default: v2).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Use test only for the final frozen evaluation.",
    )
    parser.add_argument("--ref-input-price-per-1m", type=float, default=0.15)
    parser.add_argument("--ref-output-price-per-1m", type=float, default=0.60)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from matching checkpoint and retry failed rows.",
    )
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    if args.split == "test" and args.max_samples is not None:
        raise ValueError("--max-samples is not allowed for the final test run.")
    model = args.model or DEFAULT_MODELS[args.backend]
    prompt = PROMPTS[args.prompt_version]
    output_path = resolve_output_path(args, model, manifest.name, args.split)
    metadata_path = output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    validation = load_benchmark_split(
        args.split,
        allow_test=(args.split == "test"),
    )

    # Defense-in-depth runtime leakage guard against validation rows.
    demo_ids = {ex.id for ex in manifest.examples}
    demo_hashes = {ex.text_sha256 for ex in manifest.examples}
    for row in validation:
        if str(row["id"]) in demo_ids:
            raise RuntimeError(f"Leakage guard triggered: validation id={row['id']} in manifest.")
        if normalize_text_hash(row["text"]) in demo_hashes:
            raise RuntimeError(f"Leakage guard triggered: validation text id={row['id']} in manifest.")

    if args.max_samples is not None:
        if args.max_samples <= 0 or args.max_samples > len(validation):
            raise ValueError(f"--max-samples must be between 1 and {len(validation)}")
        validation = validation.select(range(args.max_samples))
    if args.split == "test" and len(validation) != 545:
        raise ValueError(
            f"The final test run must contain exactly 545 rows, got {len(validation)}."
        )

    detector = create_detector(args, model, prompt, manifest.examples)
    runtime_info = detector.get_run_metadata()

    config = {
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "split": args.split,
        "sample_count": len(validation),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(str(row["id"]) for row in validation)).encode(
                "utf-8"
            )
        ).hexdigest(),
        "test_split_accessed": args.split == "test",
        "backend": args.backend,
        "model": model,
        "prompt_version": args.prompt_version,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "manifest_name": manifest.name,
        "manifest_strategy": manifest.strategy,
        "manifest_sha256": manifest.manifest_sha256,
        "num_shots": manifest.num_shots,
        "seed": manifest.seed,
        "demo_ids": [ex.id for ex in manifest.examples],
        "prompt_format": runtime_info["prompt_format"],
        "decoding": runtime_info["decoding"],
    }
    if args.backend == "huggingface":
        config["effective_prompt_sha256"] = runtime_info["effective_prompt_sha256"]
        config["dtype"] = runtime_info["dtype"]

    existing_rows = []
    started_at = datetime.now(timezone.utc).isoformat()
    if output_path.exists() or metadata_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"Output already exists. Use --resume to continue: {output_path}"
            )
        if not output_path.exists() or not metadata_path.exists():
            raise RuntimeError(
                "Cannot resume: both checkpoint CSV and metadata JSON must exist."
            )
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, value in config.items():
            if previous_metadata.get(key) != value:
                raise RuntimeError(
                    f"Cannot resume: checkpoint {key} does not match the requested run."
                )
        started_at = previous_metadata.get("started_at", started_at)
        existing_rows = read_checkpoint(output_path)

    successful_by_id = {
        str(row["id"]): row
        for row in existing_rows
        if row["status"] == "success"
    }
    rows = list(successful_by_id.values())
    write_checkpoint(output_path, rows)

    metadata = build_metadata(
        rows,
        config,
        runtime_info,
        started_at,
        "running",
        args.ref_input_price_per_1m,
        args.ref_output_price_per_1m,
    )
    write_metadata(metadata_path, metadata)

    for index, example in enumerate(validation):
        example_id = str(example["id"])
        if example_id in successful_by_id:
            continue

        start = time.perf_counter()
        input_tokens = ""
        output_tokens = ""
        raw_output = ""
        try:
            pred = detector.predict_with_metadata(example["text"])
            prediction = pred.label
            latency_seconds = round(pred.latency_seconds, 4)
            input_tokens = pred.input_tokens if pred.input_tokens is not None else ""
            output_tokens = pred.output_tokens if pred.output_tokens is not None else ""
            raw_output = pred.raw_output
            status = "success"
            error = ""
        except Exception as exc:
            prediction = ""
            latency_seconds = round(time.perf_counter() - start, 4)
            status = type(exc).__name__
            error = str(exc)
            print(
                f"Error on sample {index + 1}/{len(validation)} "
                f"(id={example_id}): {status}: {error}",
                file=sys.stderr,
            )

        result = {
            "id": example_id,
            "actual": int(example["label"]),
            "predicted": prediction,
            "status": status,
            "latency_seconds": latency_seconds,
            "attack_family": example.get("attack_family", ""),
            "technique": example.get("technique", ""),
            "error": error,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "raw_output": raw_output,
        }
        rows.append(result)
        with output_path.open("a", newline="", encoding="utf-8") as checkpoint:
            writer = csv.DictWriter(checkpoint, fieldnames=CSV_FIELDS)
            if checkpoint.tell() == 0:
                writer.writeheader()
            writer.writerow(result)
            checkpoint.flush()
            os.fsync(checkpoint.fileno())

        metadata = build_metadata(
            rows,
            config,
            runtime_info,
            started_at,
            "running",
            args.ref_input_price_per_1m,
            args.ref_output_price_per_1m,
        )
        write_metadata(metadata_path, metadata)

        print(
            f"Sample {index + 1}/{len(validation)}: "
            f"id={example_id}, actual={result['actual']}, "
            f"predicted={prediction if prediction != '' else 'ERROR'}, "
            f"status={status}, tokens_in={input_tokens}, "
            f"latency={result['latency_seconds']:.3f}s"
        )

    complete = len(rows) == len(validation) and all(
        row["status"] == "success" for row in rows
    )
    metadata = build_metadata(
        rows,
        config,
        runtime_info,
        started_at,
        "complete" if complete else "incomplete",
        args.ref_input_price_per_1m,
        args.ref_output_price_per_1m,
    )
    write_metadata(metadata_path, metadata)

    print("\n=== Few-Shot Validation ===")
    print(f"Backend: {args.backend} | Model: {model}")
    print(f"Manifest: {manifest.name} ({manifest.num_shots}-shot, {manifest.strategy})")
    print(f"Successful: {metadata['completed_count']}/{len(validation)}")
    print(f"Errors: {metadata['failure_count']}")
    print(json.dumps(metadata["metrics"], indent=2))
    print(json.dumps(metadata["operational"], indent=2))
    print(f"CSV: {output_path}")
    print(f"Metadata: {metadata_path}")

    if not complete:
        sys.exit(1)


if __name__ == "__main__":
    main()
