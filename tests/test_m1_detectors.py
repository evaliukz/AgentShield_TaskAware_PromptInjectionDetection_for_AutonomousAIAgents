import argparse
import csv
import hashlib

import pytest

from milestones.milestone1.scripts.run_zero_shot import (
    LEGACY_CSV_FIELDS,
    build_metadata,
    calculate_metrics,
    calculate_operational_stats,
    read_checkpoint,
    resolve_output_path,
    slugify_model,
    write_checkpoint,
)
from milestones.milestone1.src.detectors.local_hf import parse_label_json
from milestones.milestone1.src.detectors.prompts import (
    ZERO_SHOT_V1,
    ZERO_SHOT_V2,
)


def test_m1_prompts_remain_immutable():
    v1_hash = hashlib.sha256(ZERO_SHOT_V1.encode("utf-8")).hexdigest()
    v2_hash = hashlib.sha256(ZERO_SHOT_V2.encode("utf-8")).hexdigest()
    assert (
        v1_hash
        == "b198a3651065211476dd136d764e0fd96a2555037fc1129ac80758b66cdabd82"
    )
    assert (
        v2_hash
        == "398a1eba09b342463d3a78fb6ddd3ef3978d50ab474387da441e83f76bc04fef"
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"label": 0}', 0),
        ('{"label": 1}', 1),
        ('  {\n  "label": 1\n}  ', 1),
    ],
)
def test_parse_label_json_valid(raw, expected):
    assert parse_label_json(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "0",
        "1",
        '{"label": true}',
        '{"label": false}',
        '{"label": 2}',
        '{"label": "1"}',
        '{"label": 1, "extra": "x"}',
        '```json\n{"label": 1}\n```',
        "Label: 1",
    ],
)
def test_parse_label_json_rejects_invalid(raw):
    with pytest.raises(ValueError):
        parse_label_json(raw)


def test_slugify_model():
    assert slugify_model("Qwen/Qwen2.5-3B-Instruct") == "qwen2.5-3b-instruct"
    assert slugify_model("gpt-4o-mini") == "gpt-4o-mini"


def test_resolve_output_path():
    openai_args = argparse.Namespace(
        backend="openai",
        version="v2",
        max_samples=None,
        output_prefix=None,
    )
    assert (
        resolve_output_path(openai_args, "gpt-4o-mini").name
        == "zero_shot_v2_validation.csv"
    )

    hf_args = argparse.Namespace(
        backend="huggingface",
        version="v1",
        max_samples=20,
        output_prefix=None,
    )
    assert (
        resolve_output_path(hf_args, "Qwen/Qwen2.5-3B-Instruct").name
        == "zero_shot_qwen2.5-3b-instruct_v1_validation_smoke20.csv"
    )


def test_checkpoint_roundtrip_and_legacy_compat(tmp_path):
    legacy_csv = tmp_path / "legacy.csv"
    with legacy_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEGACY_CSV_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "id": "11",
                "actual": "1",
                "predicted": "1",
                "status": "success",
                "latency_seconds": "0.5",
                "attack_family": "approval_authority_escalation",
                "technique": "signature_block_esign_approval",
                "error": "",
            }
        )

    rows = read_checkpoint(legacy_csv)
    assert len(rows) == 1
    assert rows[0]["input_tokens"] == ""
    assert rows[0]["output_tokens"] == ""
    assert rows[0]["raw_output"] == ""

    modern_csv = tmp_path / "modern.csv"
    rows[0]["input_tokens"] = "220"
    rows[0]["output_tokens"] = "7"
    rows[0]["raw_output"] = '{"label": 1}'
    write_checkpoint(modern_csv, rows)
    reloaded = read_checkpoint(modern_csv)
    assert reloaded == rows


def test_calculate_metrics_and_operational_stats():
    rows = [
        {
            "actual": 1,
            "predicted": 1,
            "latency_seconds": 0.10,
            "input_tokens": 100,
            "output_tokens": 5,
        },
        {
            "actual": 1,
            "predicted": 0,
            "latency_seconds": 0.20,
            "input_tokens": 120,
            "output_tokens": 5,
        },
        {
            "actual": 0,
            "predicted": 1,
            "latency_seconds": 0.30,
            "input_tokens": 110,
            "output_tokens": 6,
        },
        {
            "actual": 0,
            "predicted": 0,
            "latency_seconds": 0.40,
            "input_tokens": 130,
            "output_tokens": 6,
        },
    ]
    metrics = calculate_metrics(rows)
    assert metrics["tp"] == 1
    assert metrics["fn"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["fpr"] == pytest.approx(0.5)

    ops = calculate_operational_stats(rows)
    assert ops["tokens"]["total_input"] == 460
    assert ops["tokens"]["total_output"] == 22
    assert ops["latency_seconds"]["mean"] == pytest.approx(0.25)
    assert ops["latency_seconds"]["p50"] == pytest.approx(0.25)


def test_build_metadata_excludes_failures_from_metrics():
    rows = [
        {
            "id": "1",
            "actual": 1,
            "predicted": 1,
            "status": "success",
            "latency_seconds": 0.5,
            "attack_family": "tool_function_abuse",
            "technique": "t1",
            "error": "",
            "input_tokens": 200,
            "output_tokens": 7,
            "raw_output": '{"label": 1}',
        },
        {
            "id": "2",
            "actual": 0,
            "predicted": "",
            "status": "ValueError",
            "latency_seconds": 0.4,
            "attack_family": "benign",
            "technique": "t2",
            "error": "Invalid JSON",
            "input_tokens": "",
            "output_tokens": "",
            "raw_output": "",
        },
    ]
    metadata = build_metadata(
        rows,
        config={"model": "Qwen/Qwen2.5-3B-Instruct"},
        runtime_info={"backend": "huggingface"},
        started_at="2026-11-22T00:00:00+00:00",
        status="incomplete",
    )
    assert metadata["completed_count"] == 1
    assert metadata["failure_count"] == 1
    assert metadata["metrics"]["samples"] == 1
    assert metadata["metrics"]["tp"] == 1
    assert "benign" not in metadata["attack_family_metrics"]
