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

from datasets import load_dataset
from dotenv import load_dotenv
import numpy as np

from milestones.milestone1.src.detectors.prompts import (
    ZERO_SHOT_V1,
    ZERO_SHOT_V2,
)

DATASET_NAME = "3nesdeniz/agentic-prompt-injection-5k"
MILESTONE_ROOT = Path(__file__).resolve().parents[1]
PROMPTS = {
    "v1": ZERO_SHOT_V1,
    "v2": ZERO_SHOT_V2,
}
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "huggingface": "Qwen/Qwen2.5-3B-Instruct",
}
LEGACY_CSV_FIELDS = [
    "id",
    "actual",
    "predicted",
    "status",
    "latency_seconds",
    "attack_family",
    "technique",
    "error",
]
CSV_FIELDS = [
    *LEGACY_CSV_FIELDS,
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

    tp, fp, tn, fn = (
        counts["tp"],
        counts["fp"],
        counts["tn"],
        counts["fn"],
    )
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


def calculate_operational_stats(rows):
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
        stats["latency_seconds"] = {
            "mean": round(float(arr.mean()), 4),
            "p50": round(float(np.percentile(arr, 50)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
            "max": round(float(arr.max()), 4),
            "total": round(float(arr.sum()), 3),
        }
    if input_tokens or output_tokens:
        stats["tokens"] = {
            "total_input": sum(input_tokens),
            "total_output": sum(output_tokens),
            "mean_input": (
                round(sum(input_tokens) / len(input_tokens), 2)
                if input_tokens
                else None
            ),
            "mean_output": (
                round(sum(output_tokens) / len(output_tokens), 2)
                if output_tokens
                else None
            ),
        }
    return stats


def build_metadata(rows, config, runtime_info, started_at, status):
    successful = [row for row in rows if row["status"] == "success"]
    grouped = defaultdict(list)
    for row in successful:
        grouped[row["attack_family"] or "unknown"].append(row)

    return {
        **config,
        "runtime": runtime_info,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "completed_count": len(successful),
        "failure_count": len(rows) - len(successful),
        "metrics": calculate_metrics(successful),
        "operational": calculate_operational_stats(successful),
        "attack_family_metrics": {
            family: calculate_metrics(family_rows)
            for family, family_rows in sorted(grouped.items())
        },
    }


def write_metadata(path, metadata):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")
        metadata_file.flush()
        os.fsync(metadata_file.fileno())
    temporary_path.replace(path)


def read_checkpoint(csv_path):
    with csv_path.open(newline="") as checkpoint:
        reader = csv.DictReader(checkpoint)
        if reader.fieldnames not in (CSV_FIELDS, LEGACY_CSV_FIELDS):
            raise RuntimeError(
                f"Checkpoint has unexpected columns: {reader.fieldnames}"
            )
        rows = []
        for row in reader:
            for field in CSV_FIELDS:
                row.setdefault(field, "")
            rows.append(row)
        return rows


def write_checkpoint(csv_path, rows):
    with csv_path.open("w", newline="") as checkpoint:
        writer = csv.DictWriter(checkpoint, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        checkpoint.flush()
        os.fsync(checkpoint.fileno())


def create_detector(args, model, prompt):
    if args.backend == "openai":
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing")
        from milestones.milestone1.src.detectors.zero_shot import (
            ZeroShotDetector,
        )

        return ZeroShotDetector(model=model, system_prompt=prompt)

    from milestones.milestone1.src.detectors.local_hf import (
        LocalHFZeroShotDetector,
    )

    return LocalHFZeroShotDetector(
        model=model,
        system_prompt=prompt,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )


def resolve_output_path(args, model):
    reports_dir = MILESTONE_ROOT / "data" / "reports"
    suffix = f"_smoke{args.max_samples}" if args.max_samples else ""
    if args.output_prefix:
        stem = f"{args.output_prefix}_{args.version}_validation{suffix}"
    elif args.backend == "openai" and model == DEFAULT_MODELS["openai"]:
        stem = f"zero_shot_{args.version}_validation{suffix}"
    else:
        slug = slugify_model(model)
        stem = f"zero_shot_{slug}_{args.version}_validation{suffix}"
    return reports_dir / f"{stem}.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Run zero-shot detection on the validation split."
    )
    parser.add_argument(
        "--version",
        choices=sorted(PROMPTS),
        default="v1",
        help="Prompt version to evaluate (default: v1).",
    )
    parser.add_argument(
        "--backend",
        choices=sorted(DEFAULT_MODELS),
        default="openai",
        help="Inference backend: openai or huggingface (default: openai).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model identifier (defaults: gpt-4o-mini for openai, "
            "Qwen/Qwen2.5-3B-Instruct for huggingface)."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for huggingface backend: auto, mps, cuda, cpu.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Weight dtype for huggingface backend: auto, bfloat16, float16, float32.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=24,
        help="Max new tokens for huggingface backend (default: 24).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional prefix subset size for smoke testing (e.g., 20).",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional custom filename prefix in data/reports/.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the matching CSV checkpoint and retry failed rows.",
    )
    args = parser.parse_args()

    model = args.model or DEFAULT_MODELS[args.backend]
    prompt = PROMPTS[args.version]
    output_path = resolve_output_path(args, model)
    metadata_path = output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_NAME)
    validation = dataset["validation"]
    if args.max_samples is not None:
        if args.max_samples <= 0 or args.max_samples > len(validation):
            raise ValueError(
                f"--max-samples must be between 1 and {len(validation)}"
            )
        validation = validation.select(range(args.max_samples))

    detector = create_detector(args, model, prompt)
    runtime_info = detector.get_run_metadata()

    config = {
        "dataset": DATASET_NAME,
        "split": "validation",
        "sample_count": len(validation),
        "backend": args.backend,
        "model": model,
        "prompt_version": args.version,
        "prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest(),
    }
    if args.backend == "huggingface":
        config["effective_prompt_sha256"] = runtime_info[
            "effective_prompt_sha256"
        ]
        config["decoding"] = runtime_info["decoding"]
        config["dtype"] = runtime_info["dtype"]

    existing_rows = []
    started_at = datetime.now(timezone.utc).isoformat()
    if output_path.exists() or metadata_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"Output already exists. Use --resume to continue: "
                f"{output_path}"
            )
        if not output_path.exists() or not metadata_path.exists():
            raise RuntimeError(
                "Cannot resume: both checkpoint CSV and metadata JSON "
                "must exist."
            )
        with metadata_path.open() as metadata_file:
            previous_metadata = json.load(metadata_file)
        for key, value in config.items():
            if previous_metadata.get(key) != value:
                raise RuntimeError(
                    f"Cannot resume: checkpoint {key} does not match "
                    f"the requested run."
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
        rows, config, runtime_info, started_at, "running"
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
            input_tokens = (
                pred.input_tokens if pred.input_tokens is not None else ""
            )
            output_tokens = (
                pred.output_tokens if pred.output_tokens is not None else ""
            )
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
        with output_path.open("a", newline="") as checkpoint:
            writer = csv.DictWriter(checkpoint, fieldnames=CSV_FIELDS)
            if checkpoint.tell() == 0:
                writer.writeheader()
            writer.writerow(result)
            checkpoint.flush()
            os.fsync(checkpoint.fileno())

        metadata = build_metadata(
            rows, config, runtime_info, started_at, "running"
        )
        write_metadata(metadata_path, metadata)

        print(
            f"Sample {index + 1}/{len(validation)}: "
            f"id={example_id}, actual={result['actual']}, "
            f"predicted={prediction if prediction != '' else 'ERROR'}, "
            f"status={status}, "
            f"latency={result['latency_seconds']:.3f}s"
        )

    complete = (
        len(rows) == len(validation)
        and all(row["status"] == "success" for row in rows)
    )
    metadata = build_metadata(
        rows,
        config,
        runtime_info,
        started_at,
        "complete" if complete else "incomplete",
    )
    write_metadata(metadata_path, metadata)

    print("\n=== Zero-Shot Validation ===")
    print(f"Backend: {args.backend} | Model: {model}")
    print(f"Version: {args.version}")
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
