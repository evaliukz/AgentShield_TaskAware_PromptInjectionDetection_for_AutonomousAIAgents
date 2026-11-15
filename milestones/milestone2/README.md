# Milestone 2 — Few-Shot In-Context Learning

Milestone 2 is the next experiment stage. Keep its implementation separate
from Milestone 1; place new work here when it is ready:

```text
milestone2/
├── scripts/       # Few-shot experiment runners and analysis
├── src/           # Reusable Milestone 2 implementation
└── data/reports/  # Generated Milestone 2 outputs (git-ignored)
```

Planned focus: select 4-shot and 8-shot demonstrations from the training split
only, fix the model and prompt protocol, and compare recall, false-positive
rate, token usage, and latency. Do not select demonstrations from validation
or test data.
