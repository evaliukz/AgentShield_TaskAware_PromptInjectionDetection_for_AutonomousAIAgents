import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

from milestones.milestone1.src.detectors.prompts import (
    ZERO_SHOT_V1,
    ZERO_SHOT_V2,
)
from milestones.milestone1.src.detectors.zero_shot import ZeroShotDetector

DATASET_NAME = "3nesdeniz/agentic-prompt-injection-5k"
MILESTONE_ROOT = Path(__file__).resolve().parents[1]
PROMPTS = {
    "v1": ZERO_SHOT_V1,
    "v2": ZERO_SHOT_V2,
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
]


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


def build_metadata(rows, config, started_at, status):
    successful = [row for row in rows if row["status"] == "success"]
    grouped = defaultdict(list)
    for row in successful:
        grouped[row["attack_family"] or "unknown"].append(row)

    return {
        **config,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "completed_count": len(successful),
        "failure_count": len(rows) - len(successful),
        "metrics": calculate_metrics(successful),
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
        if reader.fieldnames != CSV_FIELDS:
            raise RuntimeError(
                f"Checkpoint has unexpected columns: {reader.fieldnames}"
            )
        return list(reader)


def write_checkpoint(csv_path, rows):
    with csv_path.open("w", newline="") as checkpoint:
        writer = csv.DictWriter(checkpoint, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        checkpoint.flush()
        os.fsync(checkpoint.fileno())


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
        "--resume",
        action="store_true",
        help="Resume from the matching CSV checkpoint and retry failed rows.",
    )
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing")

    prompt = PROMPTS[args.version]
    model = "gpt-4o-mini"
    output_path = (
        MILESTONE_ROOT
        / "data"
        / "reports"
        / f"zero_shot_{args.version}_validation.csv"
    )
    metadata_path = output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_NAME)
    validation = dataset["validation"]
    config = {
        "dataset": DATASET_NAME,
        "split": "validation",
        "sample_count": len(validation),
        "model": model,
        "prompt_version": args.version,
        "prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest(),
    }

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

    metadata = build_metadata(rows, config, started_at, "running")
    write_metadata(metadata_path, metadata)

    detector = ZeroShotDetector(model=model, system_prompt=prompt)
    for index, example in enumerate(validation):
        example_id = str(example["id"])
        if example_id in successful_by_id:
            continue

        start = time.perf_counter()
        try:
            prediction = detector.predict(example["text"])
            status = "success"
            error = ""
        except Exception as exc:
            prediction = ""
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
            "latency_seconds": round(time.perf_counter() - start, 3),
            "attack_family": example.get("attack_family", ""),
            "technique": example.get("technique", ""),
            "error": error,
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
            rows, config, started_at, "running"
        )
        write_metadata(metadata_path, metadata)

        print(
            f"Sample {index + 1}/{len(validation)}: "
            f"id={example_id}, actual={result['actual']}, "
            f"predicted={prediction if prediction != '' else 'ERROR'}, "
            f"status={status}, "
            f"latency={result['latency_seconds']:.2f}s"
        )

    complete = (
        len(rows) == len(validation)
        and all(row["status"] == "success" for row in rows)
    )
    metadata = build_metadata(
        rows, config, started_at, "complete" if complete else "incomplete"
    )
    write_metadata(metadata_path, metadata)

    print("\n=== Zero-Shot Validation ===")
    print(f"Version: {args.version}")
    print(f"Successful: {metadata['completed_count']}/{len(validation)}")
    print(f"Errors: {metadata['failure_count']}")
    print(json.dumps(metadata["metrics"], indent=2))
    print(f"CSV: {output_path}")
    print(f"Metadata: {metadata_path}")

    if not complete:
        sys.exit(1)


if __name__ == "__main__":
    main()
