import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

from milestones.milestone2.src.icl.dataset import load_benchmark_split
from milestones.milestone3.src.training import (
    BASE_MODEL,
    BASE_MODEL_REVISION,
    DATASET_NAME,
    DATASET_REVISION,
    EFFECTIVE_SYSTEM_PROMPT,
    CausalCompletionCollator,
    encode_example,
    split_id_sha256,
    validate_no_split_overlap,
    validate_split,
)

MILESTONE_ROOT = Path(__file__).resolve().parents[1]


def choose_device(torch, requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return requested


def main():
    parser = argparse.ArgumentParser(
        description="Supervised LoRA fine-tuning on train with validation selection."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MILESTONE_ROOT / "artifacts" / "lora-qwen2.5-3b",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Optional optimizer-step cap for resumable training segments.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    import torch
    import transformers
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        EarlyStoppingCallback,
    )

    if args.max_seq_length < 64:
        raise ValueError("--max-seq-length must be at least 64.")
    if (
        args.num_train_epochs <= 0
        or args.learning_rate <= 0
        or args.early_stopping_patience < 1
        or args.max_steps == 0
        or args.max_steps < -1
    ):
        raise ValueError(
            "Epochs/rate must be positive, early-stopping patience >= 1, "
            "and --max-steps must be -1 or positive."
        )
    if args.lora_r <= 0 or args.lora_alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive.")
    if args.resume_from_checkpoint is not None:
        if not args.resume_from_checkpoint.is_dir():
            raise FileNotFoundError(
                f"Checkpoint not found: {args.resume_from_checkpoint}"
            )
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. "
            "Choose a new directory or pass --resume-from-checkpoint."
        )

    device = choose_device(torch, args.device)
    if device == "cpu":
        raise RuntimeError(
            "Refusing to start 3B LoRA training on CPU by default. "
            "Use Apple Silicon MPS or CUDA, or explicitly add a CPU-training "
            "implementation after estimating available memory and runtime."
        )

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_dataset = load_benchmark_split("train")
    validation_dataset = load_benchmark_split("validation")
    train_rows = list(train_dataset)
    validation_rows = list(validation_dataset)
    validate_split(train_rows, "train")
    validate_split(validation_rows, "validation")
    validate_no_split_overlap(train_rows, validation_rows)
    if args.max_train_samples is not None:
        if not 1 <= args.max_train_samples <= len(train_rows):
            raise ValueError("--max-train-samples is outside the train split.")
        train_rows = train_rows[: args.max_train_samples]
    if args.max_eval_samples is not None:
        if not 1 <= args.max_eval_samples <= len(validation_rows):
            raise ValueError("--max-eval-samples is outside the validation split.")
        validation_rows = validation_rows[: args.max_eval_samples]

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_MODEL_REVISION
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_examples = [
        encode_example(tokenizer, row, args.max_seq_length)
        for row in train_rows
    ]
    validation_examples = [
        encode_example(tokenizer, row, args.max_seq_length)
        for row in validation_rows
    ]

    requested_dtype = args.dtype
    if requested_dtype == "auto":
        if device == "cuda" and torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float16
    else:
        model_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[requested_dtype]
    if device == "mps" and model_dtype == torch.bfloat16:
        raise ValueError(
            "bfloat16 LoRA training on MPS is not enabled by this initial "
            "runner. Use --dtype float16 or float32 after confirming support."
        )
    if device == "cuda" and model_dtype == torch.float16 and not torch.cuda.is_available():
        raise RuntimeError("CUDA float16 requested without CUDA.")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        revision=BASE_MODEL_REVISION,
        torch_dtype=model_dtype,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = math.ceil(
        len(train_examples)
        / (args.per_device_batch_size * args.gradient_accumulation_steps)
    )
    target_training_steps = math.ceil(steps_per_epoch * args.num_train_epochs)
    if args.max_steps > 0 and (
        args.max_steps > target_training_steps
        or args.max_steps % steps_per_epoch != 0
    ):
        raise ValueError(
            "--max-steps must be an epoch boundary no greater than the "
            f"training target ({target_training_steps})."
        )

    class StopAtEpochBoundaryCallback(TrainerCallback):
        def on_epoch_end(self, training_args, state, control, **kwargs):
            if args.max_steps > 0 and state.global_step >= args.max_steps:
                control.should_training_stop = True
            return control

    metadata = {
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "datasets_version": importlib.metadata.version("datasets"),
        "train_split": "train",
        "train_count": len(train_rows),
        "train_ids_sha256": split_id_sha256(train_rows),
        "train_dataset_fingerprint": train_dataset._fingerprint,
        "validation_split": "validation",
        "validation_count": len(validation_rows),
        "validation_ids_sha256": split_id_sha256(validation_rows),
        "validation_dataset_fingerprint": validation_dataset._fingerprint,
        "test_split_accessed": False,
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_MODEL_REVISION,
        "prompt_version": "ZERO_SHOT_V2",
        "prompt_sha256": hashlib.sha256(
            EFFECTIVE_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(0) if device == "cuda" else device
        ),
        "cuda_runtime_version": torch.version.cuda if device == "cuda" else None,
        "dtype": str(model_dtype),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": importlib.metadata.version("peft"),
        "seed": args.seed,
        "max_seq_length": args.max_seq_length,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "target_training_steps": target_training_steps,
        "early_stopping_patience": args.early_stopping_patience,
        "learning_rate": args.learning_rate,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    config_path = args.output_dir / "run_config.json"
    if args.resume_from_checkpoint is not None:
        if not config_path.is_file():
            raise RuntimeError(
                "Cannot resume without the original run_config.json."
            )
        trainer_state_path = args.resume_from_checkpoint / "trainer_state.json"
        if not trainer_state_path.is_file():
            raise RuntimeError(
                f"Checkpoint is missing trainer_state.json: "
                f"{args.resume_from_checkpoint}"
            )
        checkpoint_state = json.loads(
            trainer_state_path.read_text(encoding="utf-8")
        )
        resumed_step = int(checkpoint_state["global_step"])
        if args.max_steps > 0 and args.max_steps <= resumed_step:
            raise ValueError(
                "--max-steps must be greater than the resumed checkpoint step."
            )
        previous_metadata = json.loads(config_path.read_text(encoding="utf-8"))
        immutable_fields = set(metadata) - {
            "started_at",
            "status",
            "max_steps",
            "completed_at",
            "best_checkpoint",
            "best_eval_loss",
            "train_runtime_seconds",
            "segment_train_runtime_seconds",
            "cumulative_train_runtime_seconds",
            "train_samples_per_second",
            "peak_gpu_memory_allocated_bytes",
            "peak_gpu_memory_reserved_bytes",
        }
        for key in immutable_fields:
            if key == "target_training_steps" and key not in previous_metadata:
                continue
            if previous_metadata.get(key) != metadata[key]:
                raise RuntimeError(
                    f"Cannot resume: run configuration differs for {key}."
                )
        metadata["started_at"] = previous_metadata["started_at"]
        metadata["cumulative_train_runtime_seconds"] = float(
            previous_metadata.get(
                "cumulative_train_runtime_seconds",
                previous_metadata.get("train_runtime_seconds", 0.0),
            )
        )
        metadata["peak_gpu_memory_allocated_bytes"] = int(
            previous_metadata.get("peak_gpu_memory_allocated_bytes", 0)
        )
        metadata["peak_gpu_memory_reserved_bytes"] = int(
            previous_metadata.get("peak_gpu_memory_reserved_bytes", 0)
        )
    config_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_steps=-1,
        gradient_checkpointing=True,
        fp16=(model_dtype == torch.float16 and device == "cuda"),
        bf16=(model_dtype == torch.bfloat16),
        dataloader_num_workers=0,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=[
            {"input_ids": item.input_ids, "labels": item.labels}
            for item in train_examples
        ],
        eval_dataset=[
            {"input_ids": item.input_ids, "labels": item.labels}
            for item in validation_examples
        ],
        data_collator=CausalCompletionCollator(tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            ),
            StopAtEpochBoundaryCallback(),
        ],
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint
            else None
        )
    )
    trainer.save_model(str(args.output_dir / "best_adapter"))
    tokenizer.save_pretrained(args.output_dir / "best_adapter")

    segment_runtime = next(
        (
            entry["train_runtime"]
            for entry in reversed(trainer.state.log_history)
            if "train_runtime" in entry
        ),
        0.0,
    )
    metadata["segment_train_runtime_seconds"] = segment_runtime
    metadata["cumulative_train_runtime_seconds"] = (
        float(metadata.get("cumulative_train_runtime_seconds", 0.0))
        + segment_runtime
    )
    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    metadata["status"] = (
        "partial"
        if args.max_steps > 0 and args.max_steps < target_training_steps
        else "complete"
    )
    metadata["best_checkpoint"] = trainer.state.best_model_checkpoint
    metadata["best_eval_loss"] = trainer.state.best_metric
    metadata["train_runtime_seconds"] = metadata[
        "cumulative_train_runtime_seconds"
    ]
    metadata["train_samples_per_second"] = next(
        (
            entry["train_samples_per_second"]
            for entry in reversed(trainer.state.log_history)
            if "train_samples_per_second" in entry
        ),
        None,
    )
    metadata["peak_gpu_memory_allocated_bytes"] = max(
        int(metadata.get("peak_gpu_memory_allocated_bytes", 0)),
        torch.cuda.max_memory_allocated() if device == "cuda" else 0,
    )
    metadata["peak_gpu_memory_reserved_bytes"] = max(
        int(metadata.get("peak_gpu_memory_reserved_bytes", 0)),
        torch.cuda.max_memory_reserved() if device == "cuda" else 0,
    )
    config_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved best LoRA adapter: {args.output_dir / 'best_adapter'}")


if __name__ == "__main__":
    main()
