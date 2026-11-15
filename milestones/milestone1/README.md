# Milestone 1 — Reproducible Detection Baseline

This folder contains the Milestone 1 implementation and its generated reports.
The stages follow the experiment workflow:

The zero-shot validation runs and paired comparison are complete. The formal
held-out test is intentionally still pending.

The presentation-ready findings and reusable workflow are documented in
[`Milestone1Report.md`](Milestone1Report.md), with PNG figures in [`assets/`](assets/).

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
│   ├── run_zero_shot.py     # V1/V2 validation inference, checkpoints, metrics
│   ├── analyze_errors.py    # Inspect selected validation examples
│   └── compare_models.py    # Paired metrics, changed samples, McNemar test
├── src/
│   └── detectors/
│       ├── prompts.py       # Immutable V1 and V2 prompt definitions
│       └── zero_shot.py     # OpenAI-backed binary detector
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
