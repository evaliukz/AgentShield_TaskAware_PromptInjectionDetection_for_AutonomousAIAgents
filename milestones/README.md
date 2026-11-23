# Milestone organization

Each milestone keeps its scripts, implementation, and generated reports together.
Run module commands from the repository root.

```text
milestones/
├── milestone1/   # Reproducible detection baselines: M1 (gpt-4o-mini) & M1b (Qwen2.5-3B-Instruct)
├── milestone2/   # Few-shot in-context learning on Qwen2.5-3B-Instruct (validation complete)
└── milestone3/   # LoRA training, paired test, and English report complete
```

See each milestone's README for its code map, status, and run instructions.
Presentation reports and figures live inside each milestone folder
(`Milestone1Report.md`, `Milestone1bReport.md`, `Milestone2Report.md`, and the
planned Milestone 3 report). M3 validation results are documented in
`milestone3/README.md`; the frozen ICL-vs-LoRA test results are recorded there.
