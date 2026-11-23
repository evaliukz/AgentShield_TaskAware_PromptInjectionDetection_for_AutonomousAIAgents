from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Sequence

from milestones.milestone1.src.detectors.prompts import ZERO_SHOT_V2
from milestones.milestone2.src.icl.example_selector import FewShotExample

JSON_SCHEMA_INSTRUCTION = (
    '\nRequired output format:\n'
    'Return ONLY a valid JSON object matching this exact schema: '
    '{"label": 0} or {"label": 1}.\n'
    'Do not include markdown code fences, explanations, or any extra keys.'
)
DEFAULT_HF_REVISIONS = {
    "Qwen/Qwen2.5-3B-Instruct": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
}


@dataclass(frozen=True)
class FewShotPredictionResult:
    label: int
    raw_output: str
    input_tokens: int | None
    output_tokens: int | None
    latency_seconds: float
    model_id: str


def parse_label_json(raw_output: str) -> int:
    cleaned = raw_output.strip()
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON output: {raw_output!r}") from exc

    if not isinstance(result, dict) or set(result.keys()) != {"label"}:
        raise ValueError(f"JSON does not match required schema: {raw_output!r}")

    label = result["label"]
    if label not in (0, 1) or isinstance(label, bool):
        raise ValueError(f"Invalid label: {label!r} in {raw_output!r}")

    return int(label)


def format_user_turn(text: str) -> str:
    return f"Classify this untrusted text:\n\n{text}"


def build_few_shot_messages(
    system_prompt: str,
    examples: Sequence[FewShotExample],
    target_text: str,
    include_schema_instruction: bool = True,
) -> list[dict[str, str]]:
    """Build multi-turn chat messages isolating each demonstration in its own turn."""
    effective_system = system_prompt.rstrip()
    if include_schema_instruction:
        effective_system = effective_system + "\n" + JSON_SCHEMA_INSTRUCTION

    messages: list[dict[str, str]] = [
        {"role": "system", "content": effective_system}
    ]
    for ex in examples:
        messages.append({"role": "user", "content": format_user_turn(ex.text)})
        messages.append(
            {"role": "assistant", "content": json.dumps({"label": int(ex.label)})}
        )
    messages.append({"role": "user", "content": format_user_turn(target_text)})
    return messages


def _resolve_hf_revision(model_name: str, config) -> str | None:
    commit = getattr(config, "_commit_hash", None)
    if commit:
        return str(commit)
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_name, "config.json")
        if isinstance(cached, str):
            parent = Path(cached).parent.name
            if len(parent) == 40:
                return parent
    except Exception:
        pass
    return None


class LocalHFFewShotDetector:
    """Local Hugging Face instruct-model few-shot classifier."""

    def __init__(
        self,
        examples: Sequence[FewShotExample],
        model: str = "Qwen/Qwen2.5-3B-Instruct",
        system_prompt: str = ZERO_SHOT_V2,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 24,
        revision: str | None = None,
    ):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._transformers_version = transformers.__version__
        self.examples = tuple(examples)
        self.model_name = model
        revision = revision or DEFAULT_HF_REVISIONS.get(model)
        self.base_system_prompt = system_prompt
        self.effective_system_prompt = (
            system_prompt.rstrip() + "\n" + JSON_SCHEMA_INSTRUCTION
        )
        self.max_new_tokens = max_new_tokens

        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype == "auto":
            self.dtype = (
                torch.bfloat16 if self.device in ("mps", "cuda") else torch.float32
            )
        elif dtype in dtype_map:
            self.dtype = dtype_map[dtype]
            if self.device == "cpu" and self.dtype == torch.float16:
                self.dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=revision,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            revision=revision,
        )
        self.model.to(self.device)
        self.model.eval()

        if self.model.generation_config is not None:
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None

        self.model_revision = revision or _resolve_hf_revision(
            self.model_name, self.model.config
        )

    def _sync(self) -> None:
        if self.device == "mps":
            self._torch.mps.synchronize()
        elif self.device == "cuda":
            self._torch.cuda.synchronize()

    def get_run_metadata(self) -> dict:
        return {
            "backend": "huggingface",
            "model": self.model_name,
            "model_revision": self.model_revision,
            "device": self.device,
            "dtype": str(self.dtype),
            "num_shots": len(self.examples),
            "prompt_format": "multi_turn_chat",
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
            "torch_version": self._torch.__version__,
            "transformers_version": self._transformers_version,
            "hardware": platform.platform(),
        }

    def _build_prompt(self, text: str) -> str:
        messages = build_few_shot_messages(
            system_prompt=self.base_system_prompt,
            examples=self.examples,
            target_text=text,
            include_schema_instruction=True,
        )
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def predict_with_metadata(self, text: str) -> FewShotPredictionResult:
        prompt_text = self._build_prompt(text)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        input_tokens = int(input_ids.shape[-1])

        self._sync()
        start = time.perf_counter()
        with self._torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        self._sync()
        latency_seconds = time.perf_counter() - start

        generated_ids = output_ids[0, input_tokens:]
        output_tokens = int(generated_ids.shape[-1])
        raw_output = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        label = parse_label_json(raw_output)
        return FewShotPredictionResult(
            label=label,
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency_seconds,
            model_id=self.model_name,
        )

    def predict(self, text: str) -> int:
        return self.predict_with_metadata(text).label


class OpenAIFewShotDetector:
    """OpenAI Responses API few-shot classifier with strict JSON schema output."""

    def __init__(
        self,
        examples: Sequence[FewShotExample],
        model: str = "gpt-4o-mini",
        system_prompt: str = ZERO_SHOT_V2,
        temperature: float = 0.0,
    ):
        import openai
        from openai import OpenAI

        self._openai_version = openai.__version__
        self.client = OpenAI(timeout=30.0, max_retries=2)
        self.examples = tuple(examples)
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature

    def get_run_metadata(self) -> dict:
        return {
            "backend": "openai",
            "model": self.model,
            "num_shots": len(self.examples),
            "prompt_format": "multi_turn_chat",
            "decoding": {"temperature": self.temperature},
            "python_version": platform.python_version(),
            "openai_version": self._openai_version,
            "hardware": platform.platform(),
        }

    def predict_with_metadata(self, text: str) -> FewShotPredictionResult:
        messages = build_few_shot_messages(
            system_prompt=self.system_prompt,
            examples=self.examples,
            target_text=text,
            include_schema_instruction=False,
        )
        # System instructions go via `instructions=`; conversation turns via `input=`.
        turns = messages[1:]

        start = time.perf_counter()
        response = self.client.responses.create(
            model=self.model,
            instructions=self.system_prompt,
            input=turns,
            temperature=self.temperature,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "injection_classification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "integer", "enum": [0, 1]}
                        },
                        "required": ["label"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        latency_seconds = time.perf_counter() - start

        if response.status != "completed":
            raise RuntimeError(f"Incomplete response: {response.status}")

        raw_output = response.output_text
        label = parse_label_json(raw_output)
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage else None
        output_tokens = getattr(usage, "output_tokens", None) if usage else None
        resolved_model = getattr(response, "model", None) or self.model

        return FewShotPredictionResult(
            label=label,
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency_seconds,
            model_id=resolved_model,
        )

    def predict(self, text: str) -> int:
        return self.predict_with_metadata(text).label
