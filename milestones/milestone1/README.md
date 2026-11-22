# Milestone 1 — Reproducible Detection Baseline

This folder contains the Milestone 1 implementation and its generated reports.
The stages follow the experiment workflow:

The zero-shot validation runs and paired comparison are complete. The formal
held-out test is intentionally still pending.

The presentation-ready findings and reusable workflow are documented in
[`Milestone1Report.md`](Milestone1Report.md) (`gpt-4o-mini` baseline) and
[`Milestone1bReport.md`](Milestone1bReport.md) (`Qwen/Qwen2.5-3B-Instruct`
open-weight baseline), with PNG figures in [`assets/`](assets/).

```text
Dataset inspection
    -> leakage audit
    -> majority-class baseline
    -> zero-shot V1/V2 evaluation
    -> error analysis
    -> paired model comparison
```

## Code map

```text
milestone1/
├── scripts/                 # Runnable experiment and analysis entry points
│   ├── inspect_dataset.py   # Schema, class/family distribution, data quality
│   ├── check_leakage.py     # Exact and near-duplicate split audit
│   ├── majority_baseline.py # Train-majority prediction baseline
│   ├── run_zero_shot.py     # V1/V2 validation inference (OpenAI or local HF), checkpoints, metrics
│   ├── analyze_errors.py    # Inspect selected validation examples
│   ├── compare_models.py    # Paired metrics, changed samples, McNemar test
│   └── plot_milestone1b.py  # Generate Milestone 1b PNG figures from saved reports
├── src/
│   └── detectors/
│       ├── prompts.py       # Immutable V1 and V2 prompt definitions
│       ├── types.py         # Shared PredictionResult metadata contract
│       ├── zero_shot.py     # OpenAI-backed binary detector
│       └── local_hf.py      # Local Hugging Face binary detector (Qwen2.5-3B-Instruct)
└── data/
    └── reports/             # Generated CSV and JSON outputs (git-ignored)
```

## Run from the repository root

Use the configured project Python environment and set `OPENAI_API_KEY` in the
repository-root `.env` file before running the zero-shot evaluation.

```bash
python -m milestones.milestone1.scripts.inspect_dataset
python -m milestones.milestone1.scripts.check_leakage
python -m milestones.milestone1.scripts.majority_baseline
python -m milestones.milestone1.scripts.run_zero_shot --version v1 --resume
python -m milestones.milestone1.scripts.run_zero_shot --version v2 --resume
python -m milestones.milestone1.scripts.analyze_errors
python -m milestones.milestone1.scripts.compare_models \
  --v1 milestones/milestone1/data/reports/zero_shot_v1_validation.csv \
  --v2 milestones/milestone1/data/reports/zero_shot_v2_validation.csv
```

`--resume` continues an existing version-specific checkpoint and retries failed
rows. Without it, an existing output is left untouched and the run stops.

### Milestone 1b — Local open-weight baseline (`Qwen/Qwen2.5-3B-Instruct`)

No `OPENAI_API_KEY` is required for the local Hugging Face backend. Install
dependencies from [`pyproject.toml`](../../pyproject.toml) and run:

```bash
# 20-sample smoke test (v1)
python -m milestones.milestone1.scripts.run_zero_shot \
  --backend huggingface --model Qwen/Qwen2.5-3B-Instruct --version v1 --max-samples 20

# Full 467-sample validation runs (v1 and v2)
python -m milestones.milestone1.scripts.run_zero_shot \
  --backend huggingface --model Qwen/Qwen2.5-3B-Instruct --version v1 --resume
python -m milestones.milestone1.scripts.run_zero_shot \
  --backend huggingface --model Qwen/Qwen2.5-3B-Instruct --version v2 --resume

# Paired comparison and figure generation
python -m milestones.milestone1.scripts.compare_models \
  --v1 milestones/milestone1/data/reports/zero_shot_qwen2.5-3b-instruct_v1_validation.csv \
  --v2 milestones/milestone1/data/reports/zero_shot_qwen2.5-3b-instruct_v2_validation.csv \
  --output milestones/milestone1/data/reports/paired_changes_qwen2.5-3b-instruct.csv
python -m milestones.milestone1.scripts.plot_milestone1b

# Unit tests
pytest -v
```
