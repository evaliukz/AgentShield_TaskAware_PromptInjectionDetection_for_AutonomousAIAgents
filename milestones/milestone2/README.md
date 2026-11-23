# Milestone 2 — Few-Shot In-Context Learning

Milestone 2 evaluates controlled few-shot in-context learning (`0-shot` vs.
`4-shot` vs. `8-shot`, plus curated vs. stratified-random example selection) on
the same frozen open-weight base model established in Milestone 1b
(`Qwen/Qwen2.5-3B-Instruct`), strictly selecting demonstrations from the
training split. The frozen `random_seed42_8shot` candidate was later
evaluated once on the held-out test split as part of Milestone 3; see
[`../milestone3/README.md`](../milestone3/README.md) for paired ICL-vs-LoRA
results.

Full findings, accuracy–token–latency trade-offs, subgroup error analysis, and
the frozen ICL configuration selected for Milestone 3 are documented in
[`Milestone2Report.md`](Milestone2Report.md), with PNG figures in
[`assets/`](assets/).

## Code map

```text
milestone2/
├── Milestone2Report.md              # Full M2 findings, cost/latency, frozen M3 ICL config
├── assets/                          # Reproducible PNG figures (01–04)
├── manifests/                       # Versioned, SHA-256-verified training-only manifests
│   ├── balanced_4shot.json
│   ├── balanced_8shot.json
│   └── random_seed42_8shot.json
├── src/
│   ├── icl/
│   │   └── example_selector.py      # Balanced, technique-aware, and seeded-random selectors + leakage checks
│   └── detectors/
│       └── few_shot.py              # Local HF & OpenAI multi-turn few-shot classifiers
├── scripts/
│   ├── build_manifests.py           # Generate & leakage-audit training-only manifests
│   ├── run_few_shot.py              # Checkpointed validation runner with token/latency/cost metadata
│   ├── analyze_icl.py               # Paired McNemar, subgroup, and cost/latency comparison vs. 0-shot
│   └── plot_milestone2.py           # Generate Milestone 2 figures from summary JSON
└── data/
    └── reports/                     # Generated CSV and JSON outputs (git-ignored)
```

## Run from the repository root

Install dependencies from [`pyproject.toml`](../../pyproject.toml) (no
`OPENAI_API_KEY` is required for the local Hugging Face backend):

```bash
# 1. Build and leakage-audit training-only example manifests
python -m milestones.milestone2.scripts.build_manifests

# 2. Smoke tests (20 validation samples)
python -m milestones.milestone2.scripts.run_few_shot \
  --manifest milestones/milestone2/manifests/balanced_4shot.json --max-samples 20
python -m milestones.milestone2.scripts.run_few_shot \
  --manifest milestones/milestone2/manifests/balanced_8shot.json --max-samples 20

# 3. Full 467-sample validation runs (checkpointed & resumable)
python -m milestones.milestone2.scripts.run_few_shot \
  --manifest milestones/milestone2/manifests/balanced_4shot.json --resume
python -m milestones.milestone2.scripts.run_few_shot \
  --manifest milestones/milestone2/manifests/balanced_8shot.json --resume
python -m milestones.milestone2.scripts.run_few_shot \
  --manifest milestones/milestone2/manifests/random_seed42_8shot.json --resume

# 4. Paired/subgroup/cost analysis & figures (requires M1b 0-shot v2 CSV/JSON)
python -m milestones.milestone2.scripts.analyze_icl
python -m milestones.milestone2.scripts.plot_milestone2

# 5. Unit tests
pytest -v
```
