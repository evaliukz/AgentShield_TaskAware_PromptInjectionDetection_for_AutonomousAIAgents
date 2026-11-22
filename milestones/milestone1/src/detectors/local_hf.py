import hashlib
import json
import platform
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from milestones.milestone1.src.detectors.prompts import ZERO_SHOT_V1
from milestones.milestone1.src.detectors.types import PredictionResult

JSON_SCHEMA_INSTRUCTION = (
    '\nRequired output format:\n'
    'Return ONLY a valid JSON object matching this exact schema: '
    '{"label": 0} or {"label": 1}.\n'
    'Do not include markdown code fences, explanations, or any extra keys.'
)


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name == "auto":
        return torch.bfloat16 if device in ("mps", "cuda") else torch.float32
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    dtype = mapping[dtype_name]
    if device == "cpu" and dtype == torch.float16:
        return torch.float32
    return dtype


def parse_label_json(raw_output: str) -> int:
    cleaned = raw_output.strip()
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON output: {raw_output!r}"
        ) from exc

    if not isinstance(result, dict) or set(result.keys()) != {"label"}:
        raise ValueError(
            f"JSON does not match required schema: {raw_output!r}"
        )

    label = result["label"]
    if label not in (0, 1) or isinstance(label, bool):
        raise ValueError(f"Invalid label: {label!r} in {raw_output!r}")

    return int(label)


def _sync_device(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


class LocalHFZeroShotDetector:
    """Local Hugging Face instruct-model zero-shot classifier."""

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-3B-Instruct",
        system_prompt: str = ZERO_SHOT_V1,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 24,
    ):
        self.model_name = model
        self.base_system_prompt = system_prompt
        self.effective_system_prompt = (
            system_prompt.rstrip() + "\n" + JSON_SCHEMA_INSTRUCTION
        )
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        # Ensure clean greedy decoding without sampling warnings.
        if self.model.generation_config is not None:
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None

        self.model_revision = getattr(
            self.model.config, "_commit_hash", None
        )

    def get_run_metadata(self) -> dict:
        return {
            "backend": "huggingface",
            "model": self.model_name,
            "model_revision": self.model_revision,
            "device": self.device,
            "dtype": str(self.dtype),
            "decoding": {
                "do_sample": False,
                "max_new_tokens": self.max_new_tokens,
            },
            "effective_prompt_sha256": hashlib.sha256(
                self.effective_system_prompt.encode("utf-8")
            ).hexdigest(),
            "schema_instruction_sha256": hashlib.sha256(
                JSON_SCHEMA_INSTRUCTION.encode("utf-8")
            ).hexdigest(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "hardware": platform.platform(),
        }

    def _build_prompt(self, text: str) -> str:
        messages = [
            {"role": "system", "content": self.effective_system_prompt},
            {"role": "user", "content": f"Classify this untrusted text:\n\n{text}"},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def predict_with_metadata(self, text: str) -> PredictionResult:
        prompt_text = self._build_prompt(text)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        input_tokens = int(input_ids.shape[-1])

        _sync_device(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        _sync_device(self.device)
        latency_seconds = time.perf_counter() - start

        generated_ids = output_ids[0, input_tokens:]
        output_tokens = int(generated_ids.shape[-1])
        raw_output = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        label = parse_label_json(raw_output)
        return PredictionResult(
            label=label,
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency_seconds,
            model_id=self.model_name,
        )

    def predict(self, text: str) -> int:
        return self.predict_with_metadata(text).label
