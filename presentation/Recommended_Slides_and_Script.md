# AgentShield — Recommended Presentation Slides and Script

**CMU 11-967 HW12 | In-class presentation: November 25, 2026**  
**Suggested length:** 7–8 minutes, plus questions  
**Purpose:** Present the research question, controlled ICL-versus-fine-tuning comparison, one-time paired test result, and limitations. The report is due December 11; this is a concise presentation of the work, not a substitute for the full report.

## Presentation guardrails

- Label all 467-row validation metrics as **development results**.
- State that the ICL and LoRA candidates were selected using validation, then frozen before the **single paired test evaluation**.
- Do not use the test set for additional analysis, prompt changes, threshold selection, or model tuning.
- Describe the evaluated task precisely as **text-only prompt-injection classification**. The project does not test a separate user task, live agent, tool execution, or end-to-end authorization.
- Present near-perfect test performance as a result on one fixed benchmark split, not a guarantee of security or future performance.
- Use the prepared PNG charts from `milestones/milestone3/assets/`. Avoid dense tables on slides; show only the key numbers and keep details for questions.

---

## Slide 1 — Detecting prompt injection in untrusted text

**On-slide**

- AgentShield: ICL vs. LoRA for prompt-injection detection
- CMU 11-967 HW12
- Research question: Does adapting model weights improve detection over frozen few-shot prompting?

**Visual:** A simple flow: `Untrusted text → classifier → benign / injection label`. Label it “text classification, not action authorization.”

**Speaker script (~40 seconds)**

> Autonomous assistants read retrieved documents, emails, and tool outputs. Some of that content may try to redirect the assistant or claim authority it does not have. My project asks a focused question: given one piece of untrusted text, can a language model classify it as benign or as a prompt-injection attempt? I compare two ways to use the same compact model: put labeled examples in the prompt, or train a small LoRA adapter.

**Transition:** Explain the benchmark and what the task does—and does not—represent.

---

## Slide 2 — Dataset and evaluation protocol

**On-slide**

| Train | Validation | Test |
|---:|---:|---:|
| 3,923 | 467 | 545 |

- Source-provided splits; fixed dataset revision
- Train for LoRA and ICL demonstrations
- Validation for candidate selection
- One paired test run after both candidates were frozen

**Visual:** A split-count bar chart or compact three-part diagram. Source: `3nesdeniz/agentic-prompt-injection-5k`.

**Speaker script (~45 seconds)**

> I used the dataset’s existing train, validation, and test splits: 3,923, 467, and 545 examples. LoRA training and the few-shot demonstrations used training data. I used validation for selecting the ICL manifest and LoRA checkpoint. Only after those choices were frozen did I evaluate both candidates on the same 545 test IDs. This dataset supports text-only binary classification. It does not provide a separate legitimate user task or measure whether a real agent safely executes tools.

**Transition:** Show how the two methods were made comparable.

---

## Slide 3 — Two methods, one controlled comparison

**On-slide**

- Same pinned base: `Qwen/Qwen2.5-3B-Instruct`
- Same `ZERO_SHOT_V2` system instruction and JSON label contract
- ICL: eight training-only demonstrations (`random_seed42_8shot`)
- LoRA: adapter trained on train; checkpoint chosen by validation loss
- Greedy decoding; same test IDs and runtime precision

**Visual:** Side-by-side prompt-context diagram:

```text
ICL: system prompt + 8 labeled examples + target text → JSON label
LoRA: adapted Qwen + system prompt + target text → JSON label
```

**Speaker script (~50 seconds)**

> To isolate the adaptation approach, both candidates use the same pinned Qwen 2.5 3B model, the same trust-boundary prompt, and the same JSON output format. The ICL candidate appends eight labeled training examples. The LoRA candidate uses the prompt without those demonstrations, but with adapter weights learned from training data. We decode greedily and compare predictions row-by-row. The ICL manifest and LoRA checkpoint were both selected on validation, not test.

**Transition:** Briefly explain what was varied during development.

---

## Slide 4 — Development experiments and selection

**On-slide**

- M1: prompt v1 vs. v2; GPT-4o-mini and same-base Qwen baselines
- M2: 4-shot and two 8-shot demonstration strategies
- Best frozen M2 arm: random-seed 8-shot, chosen on validation
- M3: LoRA rank 8; three epochs; selected epoch 1 by validation loss

**Visual:** Small validation-only comparison:

| Candidate | Validation F1 | Validation FPR |
|---|---:|---:|
| 8-shot random seed 42 | 0.9562* | 0.0881* |
| LoRA, selected epoch 1 | 0.9984 | 0.0063 |

`*` Fresh CUDA paired rerun for the controlled M3 comparison. The earlier M2 MPS report differs and is not the paired-comparison number.

The earlier M1b Qwen zero-shot validation reference was F1 `0.9063`, FPR `0.0818`; it was not freshly rerun as part of this CUDA pair.

**Speaker script (~55 seconds)**

> Development was staged rather than decided from test. M1 compared prompt wording and established zero-shot baselines. M2 tested shot count and example selection; selection mattered a lot, so the 8-shot random manifest was frozen using validation. In M3, I trained LoRA for three epochs and chose epoch 1 because it had the lowest validation loss. The table shows development results only. The fresh CUDA ICL run is the one I compare against LoRA, because the historical M2 report used Apple MPS and is not the same runtime.

**Transition:** Give the LoRA setup in one slide.

---

## Slide 5 — LoRA fine-tuning setup

**On-slide**

- NVIDIA A10G, CUDA, bfloat16
- 3,923 train examples; 467 validation examples
- 3 epochs / 1,473 optimizer steps; seed 42
- LoRA `r=8`, `alpha=16`, dropout `0.05`
- Loss only on assistant JSON label completion
- Selected `checkpoint-491`; validation loss `0.0003891`

**Visual:** A compact training pipeline: `chat-formatted training row → frozen Qwen + trainable LoRA matrices → label completion`.

**Speaker script (~45 seconds)**

> Fine-tuning used a single NVIDIA A10G. Each example was formatted as a chat conversation, but the loss was masked over the system and user prompt and computed only on the assistant’s JSON label. The run covered three epochs and 1,473 optimizer steps. The best validation loss was at step 491, so I selected that checkpoint. I did not run a broad hyperparameter sweep; the configuration was intentionally modest and compute was limited.

**Transition:** Now the headline result, from the frozen test comparison.

---

## Slide 6 — Final paired test result

**On-slide**

| Metric | 8-shot ICL | LoRA |
|---|---:|---:|
| Accuracy | 0.9119 | 0.9982 |
| Precision | 0.8742 | 0.9964 |
| Recall | 0.9679 | 1.0000 |
| F1 | 0.9186 | 0.9982 |
| Benign FPR | 0.1472 | 0.0038 |

- Same 545 examples; zero inference failures
- 48 ICL errors fixed; 1 regression
- Exact McNemar `p = 1.78e-13`

**Visual:** `milestones/milestone3/assets/01_test_overall_metrics.png`.

**Speaker script (~60 seconds)**

> On the same 545 test examples, LoRA achieved an F1 of 0.9982 compared with 0.9186 for ICL. Recall increased from 0.9679 to 1.0, while benign false-positive rate fell from 0.1472 to 0.0038. Paired analysis found 48 cases where LoRA corrected an ICL error and one regression; 496 were correct for both and none were wrong for both. The exact McNemar test is strongly inconsistent with equal paired correctness on this sample, but it does not establish performance beyond this benchmark.

**Transition:** Show which types of errors changed.

---

## Slide 7 — Error analysis: family results and paired changes

**On-slide**

- Biggest observed family gain: approval/authority escalation
- ICL missed 8/84; LoRA missed 0/84
- Benign false positives: ICL 39/265; LoRA 1/265
- Both models perfect on observed memory-trust and tool-function rows
- Bootstrap intervals and subgroup counts matter

**Visuals:** `02_test_attack_families.png` and, if space permits, `03_test_paired_outcomes.png`.

**Speaker script (~55 seconds)**

> The largest family-level change was in approval and authority escalation: ICL recall was 0.9048, while LoRA classified all 84 examples correctly. On benign examples, ICL raised 39 false alarms and LoRA raised one. Both methods were perfect on the observed memory-trust and tool-function subgroups, but those groups contain only 56 examples each. I therefore show the uncertainty intervals and avoid treating a perfect subgroup score as proof of zero future risk. For qualitative error examples, the recurring development-set ICL patterns were benign quoted attack text being flagged and authority-spoofing content being missed.

**Transition:** Address operational trade-offs and limits.

---

## Slide 8 — Trade-offs, limitations, and responsible use

**On-slide**

- Validation A10G: LoRA used 346 vs. ICL 829 mean input tokens
- Mean latency: LoRA 0.4925 s vs. ICL 0.3035 s per example
- Training-time accounting incomplete; no complete dollar cost
- One dataset/test split; subgroup sizes are modest
- Detector output is not authorization; use independent policy gates

**Speaker script (~55 seconds)**

> The shorter LoRA prompt used fewer input tokens, but the measured single-example latency was higher on this setup: about 0.49 seconds versus 0.30 for ICL. That is not a production serving benchmark, and the resumed training metadata does not capture the full original epoch-one segment, so I cannot claim a complete training cost. The dataset is one benchmark, and a text classifier is not a security boundary by itself. A deployed agent would still need deterministic permission checks and least privilege.

**Transition:** End with the narrow conclusion.

---

## Slide 9 — Conclusion and next steps

**On-slide**

- On this test split, frozen LoRA outperformed frozen 8-shot ICL
- Particularly lower benign FPR; no claim of general perfection
- Next: finish course report/presentation, verify dataset license, improve cost accounting
- Future work: external/agent-level evaluation with independent policy enforcement

**Speaker script (~35 seconds)**

> My narrow conclusion is that, under this controlled setup and on this fixed test split, the validation-selected LoRA adapter outperformed the frozen 8-shot ICL candidate, especially on benign false positives. The result is promising, not a guarantee of generalization or agent security. The next steps are to finish the full rubric-aligned report, verify licensing details, and evaluate broader settings only under a separately designed protocol.

---

## Likely questions and suggested answers

**Why does the project title say “Task-Aware” if the experiment is text-only?**  
The title describes the motivating application. The current benchmark task has only a text field and a binary label; it does not provide a separate user task or proposed action. The measured result should therefore be described as text-only, trust-boundary-aware classification. Truly task-conditioned detection remains future work.

**Did you tune on the test set?**  
No. Both candidates were frozen from validation decisions before either was evaluated on the same 545 test IDs. The test comparison was run once. No test-driven changes or reruns followed.

**Does `p = 1.78e-13` mean LoRA will be better in production?**  
No. It summarizes paired correctness differences on these 545 examples. It does not address distribution shift, dataset artifacts, or agent-level outcomes.

**Why is ICL latency lower if it has more prompt tokens?**  
The observed times are specific to this sequential run and runtime. Differences can depend on implementation, GPU state, tokenization, and measurement noise. A controlled batched serving benchmark is needed before drawing production latency conclusions.

**What would more compute enable?**  
Multiple seeds and a small pre-registered hyperparameter/model-size matrix selected on validation; stronger template/group-aware evaluation; and separate external benchmark or agent-behavior evaluation. Test data from this study would remain closed to further tuning.
