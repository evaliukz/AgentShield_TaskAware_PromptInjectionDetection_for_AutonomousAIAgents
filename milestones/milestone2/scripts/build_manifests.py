import argparse
from pathlib import Path

from datasets import load_dataset

from milestones.milestone2.src.icl.example_selector import (
    build_manifest,
    save_manifest,
    select_balanced_examples,
    select_random_balanced_examples,
)

DATASET_NAME = "3nesdeniz/agentic-prompt-injection-5k"
MILESTONE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS_DIR = MILESTONE_ROOT / "manifests"


def main():
    parser = argparse.ArgumentParser(
        description="Build and leakage-audit training-only few-shot example manifests."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_MANIFESTS_DIR),
        help="Directory to write versioned JSON manifests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed seed for random balanced control manifest (default: 42).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    dataset = load_dataset(DATASET_NAME)
    train_rows = list(dataset["train"])
    forbidden = {
        "validation": list(dataset["validation"]),
        "test": list(dataset["test"]),
    }

    specs = [
        (
            "balanced_4shot",
            "balanced",
            None,
            select_balanced_examples(train_rows, num_shots=4),
        ),
        (
            "balanced_8shot",
            "balanced",
            None,
            select_balanced_examples(train_rows, num_shots=8),
        ),
        (
            f"random_seed{args.seed}_8shot",
            "random_balanced",
            args.seed,
            select_random_balanced_examples(
                train_rows, num_shots=8, seed=args.seed
            ),
        ),
    ]

    for name, strategy, seed, examples in specs:
        manifest = build_manifest(
            name=name,
            strategy=strategy,
            examples=examples,
            dataset=DATASET_NAME,
            train_rows=train_rows,
            forbidden_splits=forbidden,
            seed=seed,
        )
        out_path = out_dir / f"{name}.json"
        save_manifest(manifest, out_path)
        print(
            f"Saved {out_path.name}: shots={manifest.num_shots}, "
            f"strategy={manifest.strategy}, sha256={manifest.manifest_sha256[:12]}..."
        )
        for idx, ex in enumerate(manifest.examples, 1):
            snippet = ex.text.replace("\n", " ")[:90]
            print(
                f"  [{idx}] id={ex.id} label={ex.label} "
                f"family={ex.attack_family} technique={ex.technique} | {snippet!r}"
            )


if __name__ == "__main__":
    main()
