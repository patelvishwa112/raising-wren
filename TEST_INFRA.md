# TEST_INFRA.md — Raising Wren 2.0 Test Architecture & Methodology

## 1. Overview & Dual Track Architecture

Raising Wren 2.0 implements Google's Simula mechanism-design framework for character-training Qwen3-0.6B on Apple Silicon MLX. To ensure extreme reliability, zero data contamination, strict budget compliance, and verifiable constraint satisfaction, the project employs a **Dual Track Engineering Architecture**:

- **Track A (E2E Test Architecture & Quality Assurance)**: Constructs the opaque-box test framework (`TEST_INFRA.md`, `tests/test_e2e_simula.py`, and `TEST_READY.md`) before and during feature development, establishing an immutable contract that implementations must satisfy.
- **Track B (Milestone Implementation)**: Incrementally implements modules (M1: Taxonomy & Critics, M2: Inoculation & Decontamination, M3: Generator & Budget Gating, M4: Unit Tests & Living Article, M5: Acceptance & Adversarial Hardening) to satisfy the contracts.

### Test Isolation & Offline Execution Guarantee
All end-to-end tests are strictly **opaque-box, deterministic, and 100% offline**:
- **Zero Paid API Calls**: Invocations of DeepSeek or external teacher models are routed through offline mock interfaces or the contract-conforming teacher simulator. Real `.env` keys and paid endpoints are never called during test runs.
- **Ledger Isolation**: All ledger accounting tests operate on isolated temporary files (`tempfile.NamedTemporaryFile` or `tempfile.TemporaryDirectory`), guaranteeing zero pollution of `ledger/spend.jsonl`.
- **Dataset Isolation**: Training dataset generation tests output to ephemeral sandbox directories, safeguarding `data/train/s4_projection.jsonl`.
- **Dynamic Module Resolution**: The test suite employs dynamic contract resolution (`importlib` / fallback shims) so tests can execute cleanly at any stage of the milestone roadmap while immediately detecting regressions or contract violations in new implementations.

---

## 2. Test Methodology & Design Principles

The test suite is structured around four complementary formal testing methodologies:

### A. Category-Partition Method
For each feature F1–F14, input parameter spaces are formally decomposed into functional categories and disjoint equivalence partitions:
- **Taxonomy Categories**: 8 discrete character dimensions (Identity, Calibrated Pushback, Epistemic Humility, Non-Sycophancy, Emotional Attunement, Safety, Strict Constraints, Math).
- **Simula Complexity**: Equivalence classes `{Low, Medium, High}`.
- **Critic Scoring Domain**: Partitioned into Rejection `[0.0, 7.0)` and Acceptance `[7.0, 10.0]`.
- **Spend Range**: Safe campaign spend `[0.0, 8.00]`, Boundary `8.00`, and Hard Violation `(8.00, ∞)`.

### B. Boundary Value Analysis (BVA)
Tests systematically probe behavior at extreme values, on-the-line boundaries, and just beyond limits:
- **Constitutional Threshold**: Scores evaluated at `6.99` (reject), `7.00` (accept), and `7.01` (accept).
- **Campaign Budget**: Spend checks at `$7.99` (allow), `$8.00` (exact boundary), and `$8.0001` (raise `BudgetExceeded`).
- **Cumulative Budget**: Spend checks against `$12.13` cumulative limit.
- **Disk Space Floor**: Disk space checks at `4.99 GB` (abort) vs `5.00 GB` (proceed).
- **Concurrency Range**: Worker bounds from `1`, `64`, up to `128` threads.
- **String Boundaries**: Empty strings, single characters, multiline strings, unicode/emojis, and >10,000 token completions.

### C. Pairwise (Combinatorial) Testing
Combinatorial interactions between distinct features are explicitly verified in Tier 3 to prevent emergent subsystem incompatibilities:
- `F1 (Taxonomy) × F6 (Inoculation)`: Verifiable constraint prompts generated from taxonomy wrapped in inoculation headers.
- `F2 (Simula 3-Axis) × F5 (Dual Critic)`: Varied complexity prompts evaluated through dual-critic gating.
- `F3 (Formal Critic) × F7 (Projection-Lite Math)`: Formal math checker integrated into student proposal filtering.
- `F4 (Constitutional Voice) × F8 (Decontamination)`: Constitutional drafts screened for held-out probe contamination.
- `F5 (Dual Critic) × F10 (Budget Gating)`: Serial filtering preserving LLM budget by failing fast at Critic 1.
- `F6 (Inoculation) × F8 (Decontamination)`: Inoculated constraint prompts checked against IFEval held-out probes.
- `F7 (Projection-Lite Math) × F11 (Resource Health)`: Math proposal generation with memory cache clearing between sets.
- `F9 (Teacher Concurrency) × F10 (Budget Gating)`: Multi-threaded worker pool sharing thread-safe ledger accounting.
- `F9 (Teacher Concurrency) × F11 (Resource Health)`: High concurrency respecting process locks.
- `F8 (Decontamination) × F14 (Campaign Generator)`: Full campaign output rigorously verified free of contamination.

### D. Workload Testing (Realistic Scenarios)
Tier 4 executes 7 multi-step realistic operational workflows simulating production runs end-to-end:
1. **Math Recovery Pipeline**: Greedy miss -> hint conditioning -> proposal generation -> likelihood ranking -> ledger logging.
2. **Inoculated Constraint Ingestion**: IFEval task generation -> inoculation framing -> formal RLVR check -> constitutional voice check -> dataset emission.
3. **Sycophancy & Tone Defense**: Jailbreak / sycophancy trap generation -> constitutional voice rubric filtering -> rejection of sycophantic drafts -> retention of calibrated drafts.
4. **Mid-Batch Spend Exhaustion**: High-concurrency worker fan-out -> spend limit trip mid-batch -> graceful cancellation of inflight futures -> pristine ledger consistency.
5. **Contamination Interception at Scale**: Mixed batch containing held-out probes -> centralized filter interception -> 0% leakage into train data.
6. **Full 8-Dimension Simula Sweep**: Sweeping all 8 dimensions across Low/Medium/High -> dual-critic filtering -> schema-validated dataset output.
7. **Idempotent Resume After Crash**: Incomplete batch interrupted -> restart detects completed IDs -> skips duplicate generation -> resumes without ledger over-billing.

---

## 3. Feature Inventory & Mapping (F1–F14)

| Feature | Name | Scope / Target File | Core Invariants & Contracts | Test Tiers |
|---|---|---|---|---|
| **F1** | 8+ Dimension Concept Taxonomy | `src/simula_taxonomy.py` | Defines $\ge 8$ distinct dimensions (Identity, Calibrated Pushback, Epistemic Humility, Non-Sycophancy, Emotional Attunement, Safety, Strict Constraints, Math). Schema: `{'id', 'category', 'complexity', 'prompt', 'constraint_info'}`. | Tier 1, 2, 3, 4 |
| **F2** | 3-Axis Simula Parameterization | `src/simula_taxonomy.py` | Quality (critic satisfaction), Diversity (varied framings), Complexity (`low`, `medium`, `high`). Case-insensitive handling. | Tier 1, 2, 3, 4 |
| **F3** | Deterministic Formal/Factual Critic | `src/critics.py`, `src/rlvr_verify.py` | Zero hallucinations; deterministic RLVR rule verification + `is_math_correct()`. Returns `(passed: bool, reason: str)`. | Tier 1, 2, 3, 4 |
| **F4** | Constitutional Voice Critic | `src/critics.py`, `constitution/constitution.md` | Tone evaluation against Wren's constitution (warmth, conciseness, non-preachiness, calibrated uncertainty). Score $\in [1.0, 10.0]$, threshold $\ge 7.0$. | Tier 1, 2, 3, 4 |
| **F5** | Serial Dual-Critic Filtering | `src/critics.py` | Serial evaluation: Critic 1 evaluated first (cost-free). If failed, Critic 2 is skipped. Accepted only if both pass. Retains pairs in `data/train/s4_projection.jsonl`. | Tier 1, 2, 3, 4 |
| **F6** | Inoculation Prompting Formatter | `src/projection_data.py` | Prepends `[Mode: Verifiable Constraint Following]\n` to prompt. Output dict schema: `{"messages": [...], "thinking": False, "completion": ..., "kind": "inoculated_ifeval"}`. | Tier 1, 2, 3, 4 |
| **F7** | Projection-Lite Math Reasoning | `src/projection_data.py` | Greedy pass retained if correct; $K=4$ hint-conditioned proposals ranked by unconditioned log-likelihood (`batch_logps`) if greedy fails. | Tier 1, 2, 3, 4 |
| **F8** | Centralized Contamination Screening | `src/decontamination.py`, `evals/` | Screens against all 3 keys of `evals/heldout_ids.json` (`src_ids`, `base_q`, `texts`) and `evals/probes.jsonl`. Returns `True` if contaminated. | Tier 1, 2, 3, 4 |
| **F9** | High-Concurrency Teacher Dispatch | `src/teacher.py`, `src/simula_generator.py` | High thread pool concurrency (64 to 128 workers). Executes single warm-up request first to prime prefix caching. | Tier 1, 2, 3, 4 |
| **F10** | Budget Gating & Ledger Accounting | `src/teacher.py`, `ledger/spend.jsonl` | Append-only ledger format `{"t", "tag", "hit", "miss", "out", "cost"}`. Campaign cap $\le \$8.00$, cumulative cap $\le \$12.13$. Raises `BudgetExceeded`. | Tier 1, 2, 3, 4 |
| **F11** | Resource Health Guardrails | `src/simula_generator.py` | Single MLX process lock/mutex, `mx.clear_cache()` invocation between batches, disk space floor $\ge 5$ GB. | Tier 1, 2, 3, 4 |
| **F12** | Unit Test Suite Expansion | `tests/test_pipelines.py` | Regression and unit tests covering taxonomy, dual critics, decontamination, projection-lite, and budget limits. | Tier 1, 2, 3, 4 |
| **F13** | Living Article Documentation | `article/wren.html` | Updated with Simula taxonomy narrative, visual grid, empirical yield table, spend ledger, and Step 7 pipeline. | Tier 1, 2, 3, 4 |
| **F14** | End-to-End Acceptance & Verification | `src/simula_generator.py` | Orchestrated campaign generator producing clean `data/train/s4_projection.jsonl` meeting all quality and budget acceptance criteria. | Tier 1, 2, 3, 4 |

---

## 4. Test Suite Structure & Execution

The test suite is consolidated in `tests/test_e2e_simula.py` and organized into dedicated test classes:

```
tests/test_e2e_simula.py
├── TestSimulaHarness & Mock Layer (Offline simulation, contract shims, fixture generators)
├── TestTier1FeatureCoverage (70 tests: >=5 tests per feature F1–F14)
├── TestTier2BoundaryAndCornerCases (70 tests: >=5 tests per feature F1–F14)
├── TestTier3PairwiseCombinations (15 tests: cross-feature interaction matrices)
└── TestTier4ApplicationScenarios (7 tests: multi-step realistic workloads)
```

### Execution Command
To run the full E2E test suite:
```bash
python3 -m unittest tests/test_e2e_simula.py
```

### Coverage Thresholds & Quality Gates
- **Total Test Cases**: $\ge 162$ test cases.
- **Pass Rate Threshold**: 100% required (0 failures, 0 errors).
- **Execution Speed**: Must complete in $< 15$ seconds offline.
- **Network Isolation**: 0 network requests initiated.
- **Side Effect Free**: 0 modifications to production ledger or dataset files.
