import hashlib
import json
from dataclasses import dataclass
from typing import Any

from milestones.milestone1.src.detectors.prompts import ZERO_SHOT_V2
from milestones.milestone2.src.icl.dataset import (
    DATASET_NAME,
    DATASET_REVISION,
)
from milestones.milestone2.src.detectors.few_shot import (
    JSON_SCHEMA_INSTRUCTION,
    format_user_turn,
)

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
BASE_MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
EFFECTIVE_SYSTEM_PROMPT = ZERO_SHOT_V2.rstrip() + "\n" + JSON_SCHEMA_INSTRUCTION


@dataclass(frozen=True)
class EncodedExample:
    example_id: str
    label: int
    input_ids: list[int]
    labels: list[int]


def build_messages(text: str, label: int | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": EFFECTIVE_SYSTEM_PROMPT},
        {"role": "user", "content": format_user_turn(text)},
    ]
    if label is not None:
        messages.append(
            {"role": "assistant", "content": json.dumps({"label": label})}
        )
    return messages


def encode_example(
    tokenizer: Any,
    example: dict[str, Any],
    max_seq_length: int,
) -> EncodedExample:
    label = example["label"]
    if isinstance(label, bool) or label not in (0, 1):
        raise ValueError(f"Invalid label for id={example.get('id')}: {label!r}")

    text = example["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Empty or invalid text for id={example.get('id')}")

    prompt_ids = tokenizer.apply_chat_template(
        build_messages(text),
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        build_messages(text, int(label)),
        tokenize=True,
        add_generation_prompt=False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Chat template does not preserve the generation prefix "
            f"for id={example.get('id')}; cannot safely mask prompt labels."
        )
    if len(full_ids) > max_seq_length:
        raise ValueError(
            f"Tokenized example id={example.get('id')} has {len(full_ids)} "
            f"tokens, exceeding --max-seq-length={max_seq_length}. "
            "Increase the limit explicitly rather than silently truncating."
        )

    target_ids = full_ids[len(prompt_ids) :]
    if not target_ids:
        raise ValueError(f"Empty target completion for id={example.get('id')}")

    return EncodedExample(
        example_id=str(example["id"]),
        label=int(label),
        input_ids=list(full_ids),
        labels=[-100] * len(prompt_ids) + list(target_ids),
    )


def validate_split(rows: list[dict[str, Any]], split_name: str) -> None:
    seen_ids: set[str] = set()
    seen_text_hashes: set[str] = set()
    for row in rows:
        example_id = str(row["id"])
        if example_id in seen_ids:
            raise ValueError(f"Duplicate {split_name} id: {example_id}")
        seen_ids.add(example_id)

        text = row["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty or invalid text in {split_name}: {example_id}")
        text_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        if text_hash in seen_text_hashes:
            raise ValueError(f"Duplicate {split_name} text: {example_id}")
        seen_text_hashes.add(text_hash)

        label = row["label"]
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError(
                f"Invalid label in {split_name} for id={example_id}: {label!r}"
            )


def validate_no_split_overlap(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> None:
    train_ids = {str(row["id"]) for row in train_rows}
    validation_ids = {str(row["id"]) for row in validation_rows}
    id_overlap = train_ids & validation_ids
    if id_overlap:
        raise ValueError(
            f"Train/validation ID overlap detected: {sorted(id_overlap)[:5]}"
        )

    def text_hashes(rows):
        return {
            hashlib.sha256(row["text"].strip().encode("utf-8")).hexdigest()
            for row in rows
        }

    hash_overlap = text_hashes(train_rows) & text_hashes(validation_rows)
    if hash_overlap:
        raise ValueError(
            f"Train/validation exact-text overlap detected: {len(hash_overlap)}"
        )


def split_id_sha256(rows: list[dict[str, Any]]) -> str:
    ids = sorted(str(row["id"]) for row in rows)
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


class CausalCompletionCollator:
    """Pad causal-LM examples while excluding prompt tokens from the loss."""

    def __init__(self, tokenizer):
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define a pad token before collation.")
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(item["input_ids"]) for item in features)
        input_ids = []
        attention_masks = []
        labels = []
        for item in features:
            padding = max_length - len(item["input_ids"])
            input_ids.append(
                item["input_ids"] + [self.pad_token_id] * padding
            )
            attention_masks.append(
                [1] * len(item["input_ids"]) + [0] * padding
            )
            labels.append(item["labels"] + [-100] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
