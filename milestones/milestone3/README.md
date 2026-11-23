# Milestone 3 — LoRA Fine-Tuning and Final Evaluation

The full research write-up is in [`Milestone3Report.md`](Milestone3Report.md).

## Status

The 32-train/32-validation feasibility pilot completed on an AWS NVIDIA A10G
using CUDA and bfloat16 (validation loss `0.2399`). Full training completed all
three epochs (1,473 optimizer steps). Validation-loss selection chose
`checkpoint-491` (epoch 1, loss `0.0003891`); later losses were `0.0006505`
(epoch 2) and `0.0007722` (epoch 3). The selected adapter and complete
467-example validation predictions are saved under `artifacts/` and
`data/reports/`.

### Paired validation results

The frozen M2 `random_seed42_8shot` candidate was freshly rerun with the same
pinned Qwen revision, CUDA/bfloat16 runtime, prompt, and validation examples as
LoRA. The paired comparison uses these fresh M2 predictions, rather than the
historical MPS-only summary in the M2 report.

| Model | Accuracy | F1 | Recall | FPR |
|---|---:|---:|---:|---:|
| Frozen 8-shot ICL | 0.9422 | 0.9562 | 0.9578 | 0.0881 |
| LoRA, best validation checkpoint | 0.9979 | 0.9984 | 1.0000 | 0.0063 |
| LoRA minus ICL | +5.57 pp | +4.21 pp | +4.22 pp | -8.18 pp |

Across the 467 paired predictions, LoRA fixed 26 ICL errors, introduced no
regressions, and both models were wrong on one example (exact McNemar
`p = 2.98e-08`). Improvements were concentrated in
`approval_authority_escalation` (98 examples; F1 `0.9405 → 1.0000`) and
`memory_trust_boundary` (56; `0.9818 → 1.0000`). Benign FPR fell from 14/159
to 1/159 (`0.0881 → 0.0063`). Both methods were already perfect on
`rag_document_embedded` (42) and `tool_function_abuse` (112).

On this same CUDA run, LoRA mean inference latency was `0.4925s` versus
`0.3035s` for 8-shot ICL; mean input tokens were `345.7` versus `828.7`.
Thus LoRA was more accurate and used fewer prompt tokens, but was slower per
request in this setup. These are validation/runtime observations, not a held-out
generalization claim. The earlier M2 report's MPS result (`F1 0.9577`,
`FPR 0.0755`) differs from the fresh CUDA paired rerun (`F1 0.9562`,
`FPR 0.0881`); only the fresh rerun is used in the paired table.

## Final paired test results

After freezing the epoch-1 LoRA adapter and M2's `random_seed42_8shot`
manifest, both candidates were evaluated once on the same 545 test IDs. Both
runs used the pinned dataset and Qwen revisions, `ZERO_SHOT_V2`, greedy
decoding, CUDA/bfloat16, and `max_new_tokens=24`. Both completed all 545 rows
with zero inference failures. The shared sample-ID SHA-256 is
`b0a4d21d48e0bf7a6fc506324f1a7e4992cbed64daa906ae93be7b30a439b18f`.

| Candidate | Accuracy | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| Frozen 8-shot ICL | 0.9119 | 0.8742 | 0.9679 | 0.9186 | 0.1472 |
| Frozen LoRA | **0.9982** | **0.9964** | **1.0000** | **0.9982** | **0.0038** |
| LoRA minus ICL | +8.62 pp | +12.22 pp | +3.21 pp | +7.96 pp | −14.34 pp |

Paired errors: LoRA fixed 48 ICL errors and regressed on 1, with 496 both
correct and 0 both wrong; exact two-sided McNemar `p = 1.78e-13`.

| Attack family | n | ICL F1 / recall | LoRA F1 / recall | LoRA F1 change (95% bootstrap CI) |
|---|---:|---:|---:|---:|
| `approval_authority_escalation` | 84 | 0.9500 / 0.9048 | 1.0000 / 1.0000 | +0.0500 [0.0182, 0.0909] |
| `memory_trust_boundary` | 56 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.0000 [0.0000, 0.0000] |
| `rag_document_embedded` | 84 | 0.9940 / 0.9881 | 1.0000 / 1.0000 | +0.0060 [0.0000, 0.0182] |
| `tool_function_abuse` | 56 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.0000 [0.0000, 0.0000] |

| Benign test rows | ICL FPR | LoRA FPR | LoRA − ICL FPR (95% bootstrap CI) |
|---:|---:|---:|---:|
| 265 | 0.1472 | 0.0038 | −0.1434 [−0.1887, −0.1019] |

Paired nonparametric percentile bootstrap intervals use 10,000 resamples.
Overall 95% CIs were:

| Metric | ICL | LoRA | Paired LoRA − ICL |
|---|---:|---:|---:|
| Accuracy | [0.8881, 0.9339] | [0.9945, 1.0000] | [0.0642, 0.1119] |
| Precision | [0.8354, 0.9097] | [0.9888, 1.0000] | [0.0861, 0.1613] |
| F1 | [0.8946, 0.9400] | [0.9944, 1.0000] | [0.0580, 0.1037] |
| Recall | [0.9458, 0.9862] | [1.0000, 1.0000] | [0.0138, 0.0542] |
| FPR | [0.1065, 0.1910] | [0.0000, 0.0119] | [−0.1875, −0.1012] |

The point estimates favor LoRA on this fixed test sample, especially for benign
false positives and approval-authority attacks. This is a single benchmark
comparison, not evidence that LoRA will be perfect on future data; bootstrap
intervals describe sampling uncertainty under this test-set design and do not
capture benchmark shift. Do not tune, select, or rerun models based on these
test results. The presentation and complete cost accounting remain to be
completed. The
run metadata records resumed segments but did not capture the original
step-491 segment runtime, so it is not a complete training-cost estimate.

The full comparison (including paired records and subgroup intervals) is in
`data/reports/lora_vs_icl_test.json`; predictions and run metadata are in the
M2/M3 `data/reports/` folders.

## Experiment boundary

- Train on `train` only; use `validation` only for checkpoint selection and
  LoRA candidate evaluation.
- Load only the pinned train or validation Parquet file during development at
  dataset revision `37ecd4e7962f6160509752c57379301904f806ac`; the test split
  was made available only through the explicit final-evaluation opt-in.
- Use the same Qwen 3B base, `ZERO_SHOT_V2` system prompt, JSON label format,
  tokenizer chat template, greedy decoding, and validation examples as M2.
- The training target is the assistant completion `{"label": 0}` or
  `{"label": 1}`. The loss is masked over system/user prompt tokens and is
  computed only on the target assistant completion.
- The one paired test comparison was run only after LoRA and the frozen M2 ICL
  candidate were selected.

An initial failed pilot used the generic Hugging Face dataset loader, which
materialized the test split in the local dataset cache. The run stopped during
train-example tokenization, before loading model weights or training, and did
not use test examples for training, selection, or evaluation. The loader has
since been changed to fetch only the pinned train/validation Parquet files.

## Code map

```text
milestone3/
├── README.md
├── scripts/
│   ├── train_lora.py       # Train LoRA and select checkpoint by validation loss
│   ├── evaluate_lora.py    # Checkpointed evaluation on validation or explicit final test
│   └── compare_lora.py     # Paired M2 ICL vs LoRA validation/test comparison
└── src/
    ├── training.py         # Data validation, chat formatting, masked loss collator
    └── detectors/lora.py   # Greedy JSON-label inference from a saved adapter
```

The split-specific loader lives in
`milestones/milestone2/src/icl/dataset.py` and pins the benchmark revision.

## GPU feasibility and pilot

The Milestone 2 report describes experiments on Apple M3 Pro/MPS. The current
workspace host reports an Intel Core i7-7567U (`x86_64`) and has no CUDA or MPS
available, so training runs on the user-provided AWS A10G instance. The pilot
completed one epoch over 32 train and 32 validation examples (4 optimizer
steps; validation loss `0.2399`). The complete train and validation splits
have maximum tokenized lengths of 414 and 392, respectively, under the
512-token limit. On this EC2 image, CUDA commands must set
`LD_LIBRARY_PATH=/lib/x86_64-linux-gnu` to prefer the installed NVIDIA driver
library over a stub in `/usr/local/lib`.

```bash
python -m pip install -e ".[training]"
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu python -m milestones.milestone3.scripts.train_lora \
  --output-dir milestones/milestone3/artifacts/lora-pilot \
  --num-train-epochs 1 \
  --max-train-samples 32 \
  --max-eval-samples 32 \
  --max-seq-length 512
```

The full baseline was run with the following configuration:

```bash
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu python -m milestones.milestone3.scripts.train_lora \
  --output-dir milestones/milestone3/artifacts/lora-qwen2.5-3b \
  --num-train-epochs 3 \
  --max-seq-length 512 \
  --device cuda \
  --dtype bfloat16

LD_LIBRARY_PATH=/lib/x86_64-linux-gnu python -m milestones.milestone3.scripts.evaluate_lora \
  --adapter milestones/milestone3/artifacts/lora-qwen2.5-3b/best_adapter \
  --dtype bfloat16

# Regenerate the frozen M2 per-example validation predictions for paired analysis.
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu python -m milestones.milestone2.scripts.run_few_shot \
  --manifest milestones/milestone2/manifests/random_seed42_8shot.json \
  --backend huggingface \
  --device cuda \
  --dtype bfloat16 \
  --resume

python -m milestones.milestone3.scripts.compare_lora \
  --icl-csv milestones/milestone2/data/reports/few_shot_qwen2.5-3b-instruct_v2_random_seed42_8shot_validation.csv \
  --lora-csv milestones/milestone3/data/reports/lora_validation_best_adapter.csv
```

The starter configuration uses LoRA rank 8, alpha 16, dropout 0.05, batch size
1, gradient accumulation 8, up to three epochs, and validation-loss early
stopping with patience 2. The pilot command limits the run to 32 train and 32
validation examples and one epoch. Record the actual device,
runtime, peak memory, throughput, model revision, and environment lock before
interpreting performance or comparing compute cost. Use `--dtype bfloat16` for
validation inference on the M2 MPS setup to match its reported inference dtype;
if the selected device/runtime does not support it, explicitly record the
replacement dtype and rerun the ICL comparator under the same dtype.

## Resuming a segmented training run

When a long run needs to be resumed, use `--max-steps` to stop at an epoch
boundary. The cap must be greater than the resumed checkpoint step and no
greater than the full target; all other training configuration must remain
unchanged. The cap stops training without shortening the scheduler horizon.
For the current 3-epoch run, step 491 is one epoch and the target is step 1473:

```bash
# Resume the existing epoch-1 checkpoint and stop after epoch 2.
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu python -m milestones.milestone3.scripts.train_lora \
  --output-dir milestones/milestone3/artifacts/lora-qwen2.5-3b \
  --num-train-epochs 3 \
  --max-seq-length 512 \
  --device cuda \
  --dtype bfloat16 \
  --max-steps 982 \
  --resume-from-checkpoint milestones/milestone3/artifacts/lora-qwen2.5-3b/checkpoint-491

# If needed, continue from epoch 2 to the original 3-epoch target.
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu python -m milestones.milestone3.scripts.train_lora \
  --output-dir milestones/milestone3/artifacts/lora-qwen2.5-3b \
  --num-train-epochs 3 \
  --max-seq-length 512 \
  --device cuda \
  --dtype bfloat16 \
  --max-steps 1473 \
  --resume-from-checkpoint milestones/milestone3/artifacts/lora-qwen2.5-3b/checkpoint-982
```

Each segment updates `run_config.json` with its status and cap while retaining
cumulative runtime and peak GPU memory. A `partial` status means only that the
requested step cap was reached before the full training target.

## Final evaluation record

The one-time paired final test evaluation is complete. Preserve the frozen
adapter and manifest; do not make test-driven changes or rerun inference.
