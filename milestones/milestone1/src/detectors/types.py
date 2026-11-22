from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionResult:
    label: int
    raw_output: str
    input_tokens: int | None
    output_tokens: int | None
    latency_seconds: float
    model_id: str
