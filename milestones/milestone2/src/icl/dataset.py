from typing import Literal

DATASET_NAME = "3nesdeniz/agentic-prompt-injection-5k"
DATASET_REVISION = "37ecd4e7962f6160509752c57379301904f806ac"

_SPLIT_FILES = {
    "train": "data/train.parquet",
    "validation": "data/validation.parquet",
    "test": "data/test.parquet",
}


def load_benchmark_split(
    split: Literal["train", "validation", "test"],
    *,
    allow_test: bool = False,
):
    if split == "test" and not allow_test:
        raise ValueError(
            "The held-out test split is sealed; pass allow_test=True only "
            "for the final frozen evaluation."
        )
    if split not in _SPLIT_FILES:
        raise ValueError(f"Unsupported benchmark split: {split!r}")

    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    parquet_path = hf_hub_download(
        repo_id=DATASET_NAME,
        filename=_SPLIT_FILES[split],
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    return load_dataset(
        "parquet",
        data_files={split: parquet_path},
        split=split,
    )
