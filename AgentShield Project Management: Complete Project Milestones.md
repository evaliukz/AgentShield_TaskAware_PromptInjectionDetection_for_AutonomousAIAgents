# AgentShield — Complete Project Milestones & Substeps

> **Purpose:** A concise, execution-oriented roadmap for the CMU research project **and** the extended Senior SDE / Microsoft Project Perception portfolio.
>
> **Status legend:** `[x]` completed · `[ ]` to do · `[~]` partial / needs verification.  
> **Tracks:** **CORE** = research deliverable; **EXTENSION** = production-system portfolio.  
> **Important:** A development-set result is not a final generalization claim. Keep the official test set untouched until the final model selection.

## At a glance

| Milestone | Track | What you learn/build | Completion artifact | Status |
|---|---|---|---|---|
| **M1 — Data, Baselines & Zero-Shot Evaluation** | CORE | Dataset reliability, leakage, inference, metrics | Reproducible evaluation + v1/v2 comparison | **Completed (development)** |
| **M2 — Few-Shot In-Context Learning** | CORE | Example selection, prompting, token-cost trade-offs | 0/4/8-shot comparison on validation | Not started |
| **M3 — Fine-Tuning & Research Conclusions** | CORE | LoRA, fair evaluation, model selection, generalization | Fine-tuned model + locked test evaluation + research report | Not started |
| **M4 — Agent Runtime & Security Workflow** | EXTENSION | Agent orchestration, state, tools, policy gates | Working defender/red/green sandbox demo | Not started |
| **M5 — Context, Database & RAG** | EXTENSION | Storage, retrieval, tenant-aware context | Persistent workflow + permission-aware RAG | Not started |
| **M6 — Production AI Infrastructure** | EXTENSION | Scalability, observability, cost, safe deployment | Deployable service with traces, tests and SLOs | Not started |
| **M7 — End-to-End Portfolio & Interview Readiness** | EXTENSION | Integration, trade-offs, communication | Demo, architecture docs, benchmark and presentation | Not started |

---

## M1 — Data, Baselines & Zero-Shot Evaluation ✅

**Goal:** Establish a trustworthy baseline *before* attempting to improve the model.

- [x] **1. Inspect dataset schema and quality.** Load the Hugging Face dataset; inspect fields, types, labels, nulls, empty strings, duplicates, class counts, attack families and text lengths. **Why:** Ensure experiments use valid and understood data. **Code:** `scripts/inspect_dataset.py`.
- [x] **2. Audit data leakage; establish train/dev/test protocol.** Check exact cross-split duplicates and train-to-dev/test character TF-IDF similarity; retain the original train/validation/test splits. **Why:** Avoid misleading performance from overlapping examples. **Code:** `scripts/check_leakage.py`. **Limit:** Semantic/template leakage not ruled out.
- [x] **3. Implement majority-class baseline.** Predict the label most common in *train* for each validation item. **Why:** Provide a trivial reference showing why accuracy/F1 alone can mislead. **Code:** `scripts/majority_baseline.py`.
- [x] **4. Build Zero-Shot LLM detector (v1 and v2).** Use a frozen model, structured binary output, fixed prompt versions and an API smoke test. **Why:** Measure detection without training model weights. **Code:** `src/detectors/zero_shot.py`, `src/detectors/prompts.py`, `scripts/run_zero_shot.py`.
- [x] **5. Evaluate and analyze errors.** Compute confusion matrix, accuracy, precision, recall, F1 and FPR on all **467 validation** examples; inspect false positives/negatives by attack family. **Why:** Understand model behavior and security trade-offs. **Code:** `scripts/analyze_errors.py`, evaluation logic in `scripts/run_zero_shot.py`.
- [x] **6. Compare prompt versions on paired examples.** Join v1/v2 by sample ID; count fixes/regressions and run exploratory exact McNemar test. **Why:** See *which predictions* changed, not just aggregate metrics. **Code:** `scripts/compare_models.py`.
- [ ] **7. Reproducibility hardening (recommended).** Pin dependencies; save model name, prompt version/hash, dataset revision, decoding parameters, outputs, timestamps, errors, latency and token usage; create a shared `evaluate.py` module and a one-command run script. **Why:** Make comparisons auditable and repeatable. **Code:** suggested `src/evaluation/metrics.py`, `scripts/evaluate.py`, `requirements.txt` or `pyproject.toml`.

**Validated M1 results (validation, 467 rows):**

| Metric | Majority baseline | Zero-shot v1 | Zero-shot v2 |
|---|---:|---:|---:|
| F1 | 0.7948 | 0.9474 | **0.9630** |
| Recall | 1.0000 | 0.9351 | **0.9708** |
| Precision | 0.6595 | **0.9600** | 0.9553 |
| FPR | 1.0000 | **0.0755** | 0.0881 |

**Observed:** v2 corrected 11 v1 mistakes, introduced 2 new mistakes; both wrong on 21; both correct on 433. Exact McNemar p = 0.02246 **on the development set**. v2 increased recall but also false alarms; quoting-for-analysis benign examples remained difficult. No independent test-set performance claim yet.

**M1 exit criterion:** Re-run the validation benchmark reproducibly, preserve artifacts and freeze baseline prompts. **Do not use the test set to tune them.**

---

## M2 — Few-Shot In-Context Learning (ICL) · CORE

**Goal:** Test whether labeled examples in the prompt improve detection *without changing model weights*.

- [ ] **1. Define the ICL research question.** Compare zero-shot vs 4-shot vs 8-shot on the same model, validation data and metrics. **Deliverable:** Experiment matrix and fixed hypotheses.
- [ ] **2. Build a training-only example pool.** Select examples strictly from `train`, check IDs/duplicates, and include both benign (especially quotation/analysis) and attack scenarios. **Deliverable:** Versioned example manifest.
- [ ] **3. Implement example selectors.** Start with deterministic balanced examples; optionally compare random and technique-aware selection using fixed seeds. **Deliverable:** `src/icl/example_selector.py`.
- [ ] **4. Implement Few-Shot detector.** Reuse the zero-shot output schema; prepend example text + correct labels without including validation labels. **Deliverable:** `src/detectors/few_shot.py`.
- [ ] **5. Evaluate 4-shot and 8-shot.** Run all 467 validation rows with checkpoint/resume and per-sample provenance. **Deliverable:** Predictions and paired metric table.
- [ ] **6. Evaluate cost and latency.** Record input/output tokens, per-item latency, API errors and estimated price per 1,000 classifications. **Deliverable:** Accuracy–cost–latency comparison.
- [ ] **7. Analyze subgroup behavior.** Report performance on `approval_authority_escalation`, `memory_trust_boundary`, quoting-for-analysis benign examples, etc.; compare paired regressions. **Deliverable:** ICL findings and limitations.

**M2 exit criterion:** Choose a fixed ICL configuration for the final research comparison *using validation only*.

---

## M3 — LoRA Fine-Tuning & Final Research Evaluation · CORE

**Goal:** Fairly compare **In-Context Learning vs Fine-Tuning** and complete the academic research deliverable.

- [ ] **1. Choose a feasible open-weight base model and compute budget.** Match inference/evaluation conditions where possible; distinguish a cross-model comparison from a controlled same-base-model comparison. **Deliverable:** Model/hardware rationale.
- [ ] **2. Prepare training data.** Map `text → label`, validate labels, deduplicate as appropriate within the training partition, format classification targets and tokenize. **Deliverable:** Versioned training dataset/preprocessing script.
- [ ] **3. Implement supervised LoRA (or QLoRA) training.** Configure adapters, checkpoints, seeds, batch size, learning rate and early stopping. **Deliverable:** `scripts/train_lora.py`, adapter artifact, training logs.
- [ ] **4. Tune using validation, not test.** Select checkpoint/hyperparameters and document trial count. **Deliverable:** Final frozen fine-tuned model.
- [ ] **5. Build one common inference/evaluation interface.** Ensure zero-shot, few-shot and fine-tuned models return comparable labels; use identical preprocessing and metrics. **Deliverable:** Unified evaluator.
- [ ] **6. Run final, one-time held-out test evaluation.** Freeze model/prompt/threshold selection first; compute global and family-level metrics with uncertainty intervals where appropriate. **Deliverable:** Locked test results.
- [ ] **7. Compare quality vs operational cost.** Include model size, training time/cost, inference latency, token use and deployment complexity. **Deliverable:** ICL vs LoRA trade-off matrix.
- [ ] **8. Write and present findings.** Explain the research question, methods, leakage controls, fairness caveats, results, failure modes and threats to validity. **Deliverable:** CMU presentation + final report.

**M3 exit criterion:** A complete, reproducible research comparison with a test-set conclusion (and honest limitations).

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
M2 Few-shot ICL
  ↓
M3 LoRA + locked final test + CMU report   ← FIRST PRIORITY
  ↓
M4 Safe agent runtime
  ↓
M5 Database + RAG + actual task context
  ↓
M6 Productionization
  ↓
M7 Portfolio + interview story
```

**Scope control:** Finish **M1 reproducibility cleanup + M2 + M3** before expanding deeply into M4–M7. The core project asks *which detector approach works better and at what cost?* The extension asks *how do we safely operate that detector inside a real agentic system?*

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
| `scripts/inspect_dataset.py` | Schema, counts, data quality | M1 implemented |
| `scripts/check_leakage.py` | Exact/near-duplicate cross-split audit | M1 implemented |
| `scripts/majority_baseline.py` | Trivial no-ML baseline | M1 implemented |
| `src/detectors/prompts.py` | Versioned zero-shot instructions | M1 implemented |
| `src/detectors/zero_shot.py` | Structured-output model inference | M1 implemented |
| `scripts/run_zero_shot.py` | Inference loop, results/latency | M1 implemented; harden for reproducibility |
| `scripts/analyze_errors.py` | Inspect FP/FN examples | M1 implemented |
| `scripts/compare_models.py` | Paired evaluation, McNemar | M1 implemented |
| `src/icl/example_selector.py` | Training-only shot selection | M2 planned |
| `src/detectors/few_shot.py` | Few-shot model wrapper | M2 planned |
| `scripts/train_lora.py` | LoRA model adaptation | M3 planned |
| `src/policy/engine.py` | Deterministic authorization | M4 planned |
| `src/agents/` | Agent runtime / sandbox tools | M4 planned |
| `src/storage/`, `src/rag/` | State and contextual retrieval | M5 planned |
| `src/api/`, `src/observability/` | Service and production monitoring | M6 planned |

*Status reflects what has been discussed/reported in this conversation; inspect the repository before treating any code path as verified in GitHub.*
