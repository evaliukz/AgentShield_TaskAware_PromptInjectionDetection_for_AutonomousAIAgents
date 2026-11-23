# AgentShield — Complete Project Milestones & Substeps

> **Purpose:** A concise, execution-oriented roadmap for the CMU research project **and** the extended Senior SDE / Microsoft Project Perception portfolio.
>
> **Status legend:** `[x]` completed · `[ ]` to do · `[~]` partial / needs verification.  
> **Tracks:** **CORE** = research deliverable; **EXTENSION** = production-system portfolio.  
> **Important:** A development-set result is not a final generalization claim. The official test set was accessed once after model selection; do not use its results for further tuning.

## At a glance

| Milestone | Track | What you learn/build | Completion artifact | Status |
|---|---|---|---|---|
| **M1 — Data, Baselines & Zero-Shot Evaluation** | CORE | Dataset reliability, leakage, inference, metrics | Reproducible evaluation + v1/v2 comparison | **Completed (development)** |
| **M2 — Few-Shot In-Context Learning** | CORE | Example selection, prompting, token-cost trade-offs | 0/4/8-shot comparison on validation + frozen ICL candidate | **Completed (development)** |
| **M3 — LoRA Fine-Tuning & Research Conclusions** | CORE | Parameter-efficient tuning, fair ICL comparison, final generalization | LoRA model + locked test evaluation + research report | **In progress (training, paired validation/test, and written report complete; presentation/cost accounting pending)** |
| **M4 — Agent Runtime & Security Workflow** | EXTENSION | Agent orchestration, state, tools, policy gates | Working defender/red/green sandbox demo | Not started |
| **M5 — Context, Database & RAG** | EXTENSION | Storage, retrieval, tenant-aware context | Persistent workflow + permission-aware RAG | Not started |
| **M6 — Production AI Infrastructure** | EXTENSION | Scalability, observability, cost, safe deployment | Deployable service with traces, tests and SLOs | Not started |
| **M7 — End-to-End Portfolio & Interview Readiness** | EXTENSION | Integration, trade-offs, communication | Demo, architecture docs, benchmark and presentation | Not started |

---

## M1 — Data, Baselines & Zero-Shot Evaluation ✅

**Goal:** Establish a trustworthy baseline *before* attempting to improve the model.

- [x] **1. Inspect dataset schema and quality.** Load the Hugging Face dataset; inspect fields, types, labels, nulls, empty strings, duplicates, class counts, attack families and text lengths. **Why:** Ensure experiments use valid and understood data. **Code:** `milestones/milestone1/scripts/inspect_dataset.py`.
- [x] **2. Audit data leakage; establish train/dev/test protocol.** Check exact cross-split duplicates and train-to-dev/test character TF-IDF similarity; retain the original train/validation/test splits. **Why:** Avoid misleading performance from overlapping examples. **Code:** `milestones/milestone1/scripts/check_leakage.py`. **Limit:** Semantic/template leakage not ruled out.
- [x] **3. Implement majority-class baseline.** Predict the label most common in *train* for each validation item. **Why:** Provide a trivial reference showing why accuracy/F1 alone can mislead. **Code:** `milestones/milestone1/scripts/majority_baseline.py`.
- [x] **4. Build Zero-Shot LLM detector (v1 and v2).** Use a frozen model, structured binary output, fixed prompt versions and an API smoke test. **Why:** Measure detection without training model weights. **Code:** `milestones/milestone1/src/detectors/zero_shot.py`, `milestones/milestone1/src/detectors/prompts.py`, `milestones/milestone1/scripts/run_zero_shot.py`.
- [x] **5. Evaluate and analyze errors.** Compute confusion matrix, accuracy, precision, recall, F1 and FPR on all **467 validation** examples; inspect false positives/negatives by attack family. **Why:** Understand model behavior and security trade-offs. **Code:** `milestones/milestone1/scripts/analyze_errors.py`, evaluation logic in `milestones/milestone1/scripts/run_zero_shot.py`.
- [x] **6. Compare prompt versions on paired examples.** Join v1/v2 by sample ID; count fixes/regressions and run exploratory exact McNemar test. **Why:** See *which predictions* changed, not just aggregate metrics. **Code:** `milestones/milestone1/scripts/compare_models.py`.
- [~] **7. Reproducibility hardening (partial).** Validation runs have per-sample checkpoints/resume and JSON metadata with model alias, prompt version/hash, dataset, split, timestamps and metrics. Token usage, SDK/environment lock, immutable model revision and a unified evaluator remain to be added where relevant. **Why:** Make comparisons auditable and repeatable. **Code:** `milestones/milestone1/scripts/run_zero_shot.py`.

**Validated M1 results (validation, 467 rows):**

| Metric | Majority baseline | Zero-shot v1 | Zero-shot v2 |
|---|---:|---:|---:|
| F1 | 0.7948 | 0.9474 | **0.9630** |
| Recall | 1.0000 | 0.9351 | **0.9708** |
| Precision | 0.6595 | **0.9600** | 0.9553 |
| FPR | 1.0000 | **0.0755** | 0.0881 |

**Observed:** v2 corrected 11 v1 mistakes, introduced 2 new mistakes; both wrong on 21; both correct on 433. Exact McNemar p = 0.02246 **on the development set**. v2 increased recall but also false alarms; quoting-for-analysis benign examples remained difficult. The M1 models were not included in the later paired test evaluation; these M1 results remain development-only.

**M1 exit criterion:** Development baseline and artifacts are complete. The 545-row test split was evaluated once after M2/M3 candidates were frozen; final paired results are reported in the [M3 README](milestones/milestone3/README.md).

---

## M2 — Few-Shot In-Context Learning (ICL) · CORE ✅

**Goal:** Test whether labeled examples in the prompt improve detection *without changing model weights*. **Status:** Validation study complete; the frozen candidate's one-time test evaluation is included in M3.

- [x] **1. Define the ICL research question and controlled matrix.** Compare 0-shot, 4-shot and two 8-shot selection strategies on `Qwen/Qwen2.5-3B-Instruct`. **Code/report:** `milestones/milestone2/Milestone2Report.md`.
- [x] **2. Build and verify training-only manifests.** All examples come from `train`; automated ID and normalized-text leakage checks found zero overlap with validation/test. **Code:** `milestones/milestone2/scripts/build_manifests.py`; **artifacts:** `milestones/milestone2/manifests/`.
- [x] **3. Implement selectors and the few-shot detector.** Support deterministic balanced and seeded-random manifests with a shared strict JSON label contract. **Code:** `milestones/milestone2/src/icl/example_selector.py`, `milestones/milestone2/src/detectors/few_shot.py`.
- [x] **4. Run all validation comparisons.** Three few-shot arms completed 467/467 each with zero format errors; compare with the 0-shot v2 control. **Code:** `milestones/milestone2/scripts/run_few_shot.py`, `milestones/milestone2/scripts/analyze_icl.py`.
- [x] **5. Measure cost, latency and subgroup behavior.** Capture token counts, MPS latency, estimated API-equivalent cost, attack-family and technique-level errors. **Evidence:** `milestones/milestone2/Milestone2Report.md` and `milestones/milestone2/assets/`.
- [x] **6. Freeze the ICL candidate for M3.** Select `random_seed42_8shot` using validation only; manifest SHA-256 prefix `f87a16166299`. **Control metrics:** F1 `0.9577`, recall `0.9545`, precision `0.9608`, FPR `0.0755`, p50 latency `1.353 s`.

**Validated M2 results (467 rows; same Qwen base, MPS and greedy decoding):**

| Configuration | F1 | Recall | FPR | Paired result vs. 0-shot v2 |
|---|---:|---:|---:|---|
| 0-shot v2 | 0.9063 | 0.8636 | 0.0818 | Baseline |
| 4-shot balanced | 0.9109 | 0.8799 | 0.1006 | 17 fixed / 15 regressed; McNemar p = 0.8601 |
| 8-shot balanced | 0.8703 | 0.7955 | 0.0629 | 10 fixed / 28 regressed; p = 0.0051 |
| **8-shot random seed 42 (frozen)** | **0.9577** | **0.9545** | **0.0755** | **30 fixed / 1 regressed; p = 2.98e-08** |

**Finding:** Shot count alone did not predict performance; two 8-shot manifests differed by 8.74 F1 points. The selected random 8-shot arm improved validation F1 by 5.14 pp over the same-base zero-shot control, but raised p50 latency by 66% and input tokens by 2.40×. It still misclassified 11/12 `quoting_attack_for_analysis` examples. Because the candidate was selected on validation, treat these as exploratory results with selection optimism.

**M2 exit criterion:** Met. The frozen manifest and experiment protocol were used in M3's one-time paired test evaluation; do not make further validation- or test-driven ICL changes.

---

## M3 — LoRA Fine-Tuning & Final Research Evaluation · CORE · IN PROGRESS (PAIRED TEST AND REPORT COMPLETE)

**Goal:** Fairly compare **In-Context Learning vs Fine-Tuning** and complete the academic research deliverable. This is the next core milestone.

- [x] **1. Choose the base model and initial hardware candidate.** The controlled target is the same pinned `Qwen/Qwen2.5-3B-Instruct` revision as M2. The user-provided AWS instance has an NVIDIA A10G (23 GB VRAM); a 32-example pilot confirmed CUDA/bfloat16 feasibility.
- [x] **2. Prepare training data.** Implement `text → JSON label` chat formatting, strict label/text/duplicate validation, sequence-length checks, prompt-token loss masking, and train/validation ID hashes. Full train/validation tokenization is verified; the maximum lengths are 414 and 392. Dataset loading is pinned to explicit train/validation Parquet files. **Code:** `milestones/milestone3/src/training.py`, `milestones/milestone2/src/icl/dataset.py`.
- [x] **3. Implement supervised LoRA training.** The full 3,923-example run completed 3 epochs / 1,473 optimizer steps on the A10G. Validation-loss selection chose epoch 1 (`checkpoint-491`, loss `0.0003891`); later epoch losses were higher. The run configuration, selected adapter, and hardware metadata are saved. **Code:** `milestones/milestone3/scripts/train_lora.py`; optional dependency: `.[training]`.
- [x] **4. Run validation-only LoRA inference.** All 467 validation examples completed with zero inference failures. The selected LoRA model scored F1 `0.9984`, recall `1.0000`, and FPR `0.0063`. **Code:** `milestones/milestone3/src/detectors/lora.py`, `milestones/milestone3/scripts/evaluate_lora.py`.
- [x] **5. Select the LoRA checkpoint using validation, not test.** Selected the best of three epoch checkpoints from one training configuration; no hyperparameter sweep was run. **Deliverable:** Frozen adapter at `milestones/milestone3/artifacts/lora-qwen2.5-3b/best_adapter`.
- [x] **6. Compare frozen ICL and LoRA on paired validation outputs.** The fresh, pinned M2 8-shot rerun and LoRA predictions share the same 467 IDs, Qwen revision, prompt hash, CUDA/bfloat16 runtime, and greedy decoding. **Code:** `milestones/milestone3/scripts/compare_lora.py`.
- [x] **7. Run the one-time held-out test evaluation.** Both frozen candidates completed the same 545 test IDs with zero inference failures. The paired report includes overall and family metrics, paired errors, exact McNemar testing, and 10,000-resample paired bootstrap intervals. **Deliverable:** `milestones/milestone3/data/reports/lora_vs_icl_test.json` (local, git-ignored) and the [M3 README summary](milestones/milestone3/README.md).
- [~] **8. Compare quality vs operational cost.** The one-time test comparison favors LoRA on quality: F1 `0.9186 → 0.9982`, recall `0.9679 → 1.0000`, and FPR `0.1472 → 0.0038`; 48 ICL errors were fixed and 1 regression occurred (McNemar `p = 1.78e-13`). On validation, LoRA used fewer mean input tokens (`345.7` vs. `828.7`) but was slower per request (`0.4925s` vs. `0.3035s`) than 8-shot ICL on the A10G. Training-cost accounting and a complete trade-off matrix remain pending. Do not use test results to tune. **Deliverable:** ICL vs LoRA trade-off matrix.
- [x] **9. Write the research reports.** Document the M3 experiment and prepare the integrated HW12 report covering Problems 1–4, including ethics, task formulation, ICL/model decisions, fine-tuning choices, test results, and qualitative error analysis. **Deliverables:** `milestones/milestone3/Milestone3Report.md` and [`final_report/Final_Report.md`](final_report/Final_Report.md).
- [ ] **10. Prepare and present findings.** Adapt the report into the CMU presentation, including key figures, results, limitations, and an honest summary of the one-time test comparison. **Deliverable:** CMU presentation.

**Paired validation snapshot (467 examples; development only):**

| Candidate | F1 | Recall | FPR | Accuracy |
|---|---:|---:|---:|---:|
| Frozen 8-shot ICL | 0.9562 | 0.9578 | 0.0881 | 0.9422 |
| LoRA, selected epoch 1 | 0.9984 | 1.0000 | 0.0063 | 0.9979 |
| Difference (LoRA − ICL) | +4.21 pp | +4.22 pp | −8.18 pp | +5.57 pp |

LoRA fixed 26 ICL errors with no regressions (exact McNemar `p = 2.98e-08`).
Attack-family gains were strongest for `approval_authority_escalation`
(F1 `0.9405 → 1.0000`) and `memory_trust_boundary`
(`0.9818 → 1.0000`); benign FPR fell from 14/159 to 1/159. Both candidates
were perfect on `rag_document_embedded` and `tool_function_abuse`. The historical
M2 report's MPS score (`F1 0.9577`, `FPR 0.0755`) differs from the fresh CUDA
rerun (`F1 0.9562`, `FPR 0.0881`); the fresh rerun is used for the paired
comparison. See `milestones/milestone3/README.md` and the ignored JSON/CSV
artifacts under `milestones/milestone3/data/reports/`.

**Paired test snapshot (545 frozen examples; final one-time evaluation):**

| Candidate | Accuracy | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| Frozen 8-shot ICL | 0.9119 | 0.8742 | 0.9679 | 0.9186 | 0.1472 |
| Frozen LoRA | 0.9982 | 0.9964 | 1.0000 | 0.9982 | 0.0038 |
| LoRA − ICL | +8.62 pp | +12.22 pp | +3.21 pp | +7.96 pp | −14.34 pp |

Paired errors: 48 fixed, 1 regression, 496 both correct, and 0 both wrong.
Exact two-sided McNemar `p = 1.78e-13`. Paired 95% bootstrap CIs for the
LoRA-minus-ICL differences were precision `[+8.61, +16.13]` pp, F1
`[+5.80, +10.37]` pp, recall
`[+1.38, +5.42]` pp, FPR `[−18.75, −10.12]` pp, and accuracy
`[+6.42, +11.19]` pp. Attack-family point estimates and intervals are in the
[M3 README](milestones/milestone3/README.md). These are benchmark-specific
estimates; they do not establish perfect future performance or robustness to
distribution shift.

**M3 design constraints / next actions:**
- Keep the base model, tokenizer, label mapping, JSON output contract and validation/test examples identical between ICL and LoRA where applicable; report training and inference costs separately.
- Train only on the 3,923-row training split. Validation was used for checkpoint selection; both candidates are frozen and the single paired test phase is complete. Do not use test outcomes for tuning or model changes.
- The local execution host is an Intel Core i7-7567U (`x86_64`) without CUDA or MPS. The user-provided AWS A10G is the verified training target; record its GPU/runtime, peak memory, throughput, sequence length and actual device in run metadata.
- The 32-train/32-validation pilot has passed on CUDA. The full baseline uses the pinned train/validation Parquet files only; a failed early loader attempt materialized test data in the HF cache but did not use test examples for training, selection or evaluation.
- Treat validation-selected ICL and LoRA results as development results with selection optimism; make the final claim from the locked, paired 545-row test evaluation. Training runtime metadata includes resumed/recovery segments and does not capture the original step-491 segment, so it is not a complete training-cost estimate.

**M3 exit criterion:** A complete, reproducible research comparison with a paired test-set conclusion (and honest limitations). Training, validation, the one-time paired test, and the written report are complete; presentation and cost accounting remain.

---

## M4 — Agent Runtime & Security Workflow · EXTENSION

**Goal:** Turn the standalone detector into a controlled AI-agent security component.

- [ ] **1. Define a narrow, authorized security scenario.** Example: simulated suspicious sign-in investigation; use synthetic data only. **Deliverable:** Threat model, allowed tools and success criteria.
- [ ] **2. Implement a minimal Defender Agent.** Task → simulated security tool → untrusted result → reasoning → proposed action. **Deliverable:** Runnable safe-agent demo.
- [ ] **3. Place AgentShield at the trust boundary.** Scan external/tool text before it enters the agent context, and inspect high-risk proposed actions independently. **Deliverable:** Detector middleware.
- [ ] **4. Add deterministic policy enforcement.** Enforce tool allowlists, least privilege, approval requirements and deny-by-default on sensitive actions; **never authorize solely from a model's “benign” label**. **Deliverable:** `src/policy/engine.py` + policy tests.
- [ ] **5. Introduce workflow state and orchestration.** Track steps, retries, timeouts, cancellation, idempotency and checkpoints. **Deliverable:** State machine / orchestrator.
- [ ] **6. Add constrained Red and Green agents.** Red generates **simulated** attack cases in a sandbox; Green proposes remediation with human approval for any privileged action. **Deliverable:** Red–Blue–Green demonstration.
- [ ] **7. Run end-to-end safety tests.** Measure attack success rate, unauthorized tool execution, approval bypass, false blocks and recovery after failure. **Deliverable:** Integration test suite.

**M4 exit criterion:** A reproducible sandbox workflow that demonstrates *detection + independent authorization controls*.

---

## M5 — Context, Database & RAG · EXTENSION

**Goal:** Add persistent state and relevant enterprise context, without losing tenant or permission boundaries.

- [ ] **1. Define domain entities and access patterns.** `Task`, `AgentRun`, `Step`, `ToolCall`, `PolicyDecision`, `Detection`, `AuditEvent`, `Document`. **Deliverable:** Data model and query patterns.
- [ ] **2. Choose the system of record.** Start with SQLite/Postgres for structured workflow state; justify transactions and indexing. **Deliverable:** Schema + migrations.
- [ ] **3. Persist run/checkpoint/audit data.** Track correlation IDs, model/prompt versions, decisions and errors. **Deliverable:** Durable execution history.
- [ ] **4. Build a small security knowledge corpus.** Use synthetic policies, tool documentation and investigation playbooks with source metadata. **Deliverable:** Versioned corpus.
- [ ] **5. Implement baseline retrieval.** Chunk/index docs, compute embeddings, compare lexical vs vector vs hybrid search and optional reranking. **Deliverable:** RAG retrieval module.
- [ ] **6. Enforce permission-aware retrieval.** Filter by tenant and authorization *before* content is provided to the model; defend against prompt injection within retrieved documents. **Deliverable:** Access-control tests.
- [ ] **7. Integrate task context into detection.** Provide **explicit legitimate task + source + untrusted content** and compare task-aware vs text-only detection on a properly constructed contextual benchmark. **Deliverable:** Context-aware experiment.
- [ ] **8. Evaluate retrieval.** Measure Recall@k, relevance, freshness, context cost and whether retrieved content improves downstream security outcomes. **Deliverable:** RAG evaluation report.

**M5 exit criterion:** A functioning context-backed pipeline with isolated data access and measurable retrieval quality.

---

## M6 — Production AI Infrastructure · EXTENSION

**Goal:** Make AgentShield measurable, reliable, scalable and safe to operate.

- [ ] **1. Define API contracts and service boundaries.** Detection, policy decision and run-status endpoints; validate inputs and return typed errors. **Deliverable:** API spec / FastAPI service.
- [ ] **2. Add async processing and backpressure.** Queues/workers, concurrency limits, rate limiting, cancellation and retry policy. **Deliverable:** Load-testable execution path.
- [ ] **3. Add resilience controls.** Timeouts, bounded retries, idempotency, circuit breaker/fallback, graceful degradation and **safe handling of failed security checks**. **Deliverable:** Failure-injection tests.
- [ ] **4. Add observability.** Structured logs, metrics, distributed traces and correlation IDs across agent/tool/model/policy steps; redact secrets and sensitive content. **Deliverable:** Trace dashboard + runbook.
- [ ] **5. Measure performance and cost.** P50/P95 latency, throughput, failure rates, tokens/run, cost/run and resource usage. **Deliverable:** Benchmark report and budgets.
- [ ] **6. Containerize and deploy.** Docker locally; optionally Kubernetes with health checks, autoscaling and resource limits if it adds real learning value. **Deliverable:** Reproducible deployment.
- [ ] **7. Implement secure operations.** Secrets management, identity/RBAC, tenant isolation, least-privileged service accounts, audit retention and CI security checks. **Deliverable:** Security checklist and test evidence.
- [ ] **8. Define SLOs and safe rollout.** Example: availability, p95 latency, maximum false-block budget; canary/shadow mode and rollback. **Deliverable:** Operational readiness checklist.

**M6 exit criterion:** A deployment/demo that can be stress-tested, traced, recovered and safely rolled back.

---

## M7 — Integration, Portfolio & Senior SDE Interview · EXTENSION

**Goal:** Present AgentShield as an evidence-backed AI security system, not just a model benchmark.

- [ ] **1. Integrate all modules behind one demo workflow.** Run a synthetic attack/benign scenario end-to-end. **Deliverable:** `make demo` or one documented command.
- [ ] **2. Finalize architecture and sequence diagrams.** Show trust boundaries, detector, agent runtime, policy engine, database, RAG and observability. **Deliverable:** `docs/architecture.md`.
- [ ] **3. Record key architecture decisions.** Why LoRA vs ICL, SQL vs alternatives, static vs retrieved examples, sync vs async, block vs review. **Deliverable:** ADRs with trade-offs.
- [ ] **4. Consolidate evaluation dashboards.** Research test results separate from system-level attack, latency, error and cost results. **Deliverable:** Benchmark summary.
- [ ] **5. Produce a short demo and readable README.** Reproduction steps, scope, demo video/screenshots (optional), limitations and security guardrails. **Deliverable:** Portfolio-quality GitHub repo.
- [ ] **6. Prepare interview stories.** Practice requirements, design, trade-offs, debugging, failure modes, observability, scalability and collaboration narratives. **Deliverable:** 10-minute system design walkthrough + deep-dive Q&A.

**M7 exit criterion:** A reviewer can reproduce the core experiment, run the safe system demo and understand every major design choice.

---

## Dependency order & recommended scope

```text
M1 Data & zero-shot evaluation (done)
  ↓
M2 Few-shot ICL (done; random_seed42_8shot frozen on validation)
  ↓
M3 LoRA + locked paired test + CMU report   ← CURRENT PRIORITY
  ↓
M4 Safe agent runtime
  ↓
M5 Database + RAG + actual task context
  ↓
M6 Productionization
  ↓
M7 Portfolio + interview story
```

**Scope control:** Finish the **M3 research report, presentation, and cost accounting** before expanding deeply into M4–M7. M1/M2 development work and the one-time M3 test comparison are complete; preserve the frozen candidates and do not reopen test data for further tuning. The core project asks *which detector approach works better and at what cost?* The extension asks *how do we safely operate that detector inside a real agentic system?*

## Reusable engineering pattern (apply to every milestone)

1. **Specify:** Problem, inputs/outputs, assumptions, trust boundaries, success metrics.
2. **Baseline:** Implement the simplest meaningful reference.
3. **Build:** Separate interfaces, logic, data and experiments.
4. **Measure:** Use controlled comparisons and realistic failure cases.
5. **Diagnose:** Inspect individual mistakes, regressions and subgroup outcomes.
6. **Freeze & document:** Version configuration, keep artifacts, state limitations.
7. **Operationalize:** Add observability, reliability and safe recovery only where needed.

**Interview shorthand:** *Problem → Baseline → Architecture → Trade-offs → Evaluation → Failures → Production readiness.*

### Code index — current / planned

| File/module | Role | Status |
|---|---|---|
| `milestones/milestone1/scripts/inspect_dataset.py` | Schema, counts, data quality | M1 implemented |
| `milestones/milestone1/scripts/check_leakage.py` | Exact/near-duplicate cross-split audit | M1 implemented |
| `milestones/milestone1/scripts/majority_baseline.py` | Trivial no-ML baseline | M1 implemented |
| `milestones/milestone1/src/detectors/prompts.py` | Versioned zero-shot instructions | M1 implemented |
| `milestones/milestone1/src/detectors/zero_shot.py` | Structured-output model inference | M1 implemented |
| `milestones/milestone1/scripts/run_zero_shot.py` | Checkpointed inference, metadata and metrics | M1 implemented; token/environment pinning remains partial |
| `milestones/milestone1/scripts/analyze_errors.py` | Inspect FP/FN examples | M1 implemented |
| `milestones/milestone1/scripts/compare_models.py` | Paired evaluation, McNemar | M1 implemented |
| `milestones/milestone2/src/icl/example_selector.py` | Training-only shot selection and leakage checks | M2 implemented |
| `milestones/milestone2/src/detectors/few_shot.py` | Local HF and OpenAI few-shot wrapper | M2 implemented |
| `milestones/milestone2/scripts/run_few_shot.py` | Checkpointed ICL validation runner | M2 implemented; now loads only validation Parquet |
| `milestones/milestone2/src/icl/dataset.py` | Pinned split-specific benchmark loader | Train/validation by default; explicit final-test opt-in |
| `milestones/milestone2/scripts/analyze_icl.py` | Paired, subgroup and cost/latency analysis | M2 implemented |
| `milestones/milestone2/manifests/random_seed42_8shot.json` | Frozen training-only ICL candidate | M2 selected on validation; evaluated once in final paired test |
| `milestones/milestone3/scripts/train_lora.py` | LoRA training pipeline | 3 epochs / 1,473 steps complete; best checkpoint-491 |
| `milestones/milestone3/scripts/evaluate_lora.py` | Checkpointed adapter evaluation on validation or explicit final test | 467/467 validation and 545/545 test complete |
| `milestones/milestone3/scripts/compare_lora.py` | Paired M2 ICL vs LoRA metrics, family analysis and uncertainty | Validation and one-time test comparisons complete |
| `milestones/milestone3/src/training.py` | Train/validation formatting, checks and masked completion loss | Implemented and verified on pinned Qwen tokenizer |
| `milestones/milestone3/Milestone3Report.md` | Final English M3 research report | Written report complete; presentation and cost accounting pending |
| `src/policy/engine.py` | Deterministic authorization | M4 planned |
| `src/agents/` | Agent runtime / sandbox tools | M4 planned |
| `src/storage/`, `src/rag/` | State and contextual retrieval | M5 planned |
| `src/api/`, `src/observability/` | Service and production monitoring | M6 planned |

*Milestone 1 and 2 paths are organized under `milestones/milestone1/` and `milestones/milestone2/`; M3 training, paired validation, the one-time paired test, and the written report are complete under `milestones/milestone3/`. The remaining work is the presentation and complete training-cost accounting. The initial failed pilot materialized the test parquet in the Hugging Face cache but did not use test examples for training, selection, or evaluation; the later final evaluation used the pinned test data only after both candidates were frozen. Do not tune from the test results.*
