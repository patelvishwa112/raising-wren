# Project: Raising Wren 2.0 Synthetic Data Generation & Character Pipeline

## Architecture
Raising Wren 2.0 implements Google's Simula mechanism-design framework to train Qwen3-0.6B on Apple Silicon MLX toward the constitution character. The architecture consists of:
1. **Taxonomy & Concept Engine (`src/simula_taxonomy.py`)**: Defines an 8+ dimensional domain concept taxonomy parameterized across Quality, Diversity, and Complexity (Low/Medium/High).
2. **Dual-Critic Rejection Pipeline (`src/critics.py`)**:
   - *Critic 1 (Factual / Formal)*: Deterministic constraint and math verification via `src/rlvr_verify.py` and `is_math_correct()`.
   - *Critic 2 (Constitutional Voice)*: LLM rubric scoring ($\ge 7/10$) against `constitution/constitution.md` (warmth, conciseness, absence of moralizing preaching, calibrated uncertainty).
3. **Inoculation & Projection-Lite Engine (`src/projection_data.py`, `src/decontamination.py`)**:
   - Inoculation headers (`[Mode: Verifiable Constraint Following]\n`) for strict formatting tasks to prevent persona pollution.
   - Projection-Lite sampling for math reasoning: student-native greedy solutions where correct; $K=4$ hint-conditioned student proposals ranked by unconditioned likelihood via `batch_logps` for missed items.
   - Comprehensive decontamination screening against `evals/heldout_ids.json` (all 3 keys: `src_ids`, `base_q`, `texts`) and `evals/probes.jsonl`.
4. **Budget-Gated Concurrency Driver (`src/simula_generator.py`)**:
   - High thread concurrency (64 to 128 workers) via `src/teacher.py:run_batch()`.
   - Ledger accounting in `ledger/spend.jsonl`, strictly enforcing an $8.00 campaign spend cap and $12.13 cumulative ceiling.
   - Health guardrails: single MLX process lock, `mx.clear_cache()` between batches, disk space $\ge 5$ GB.
5. **Living Article & Verification (`article/wren.html`, `tests/test_pipelines.py`)**:
   - Comprehensive unit and E2E test coverage.
   - Living write-up documenting Simula taxonomy, empirical yield table, spend ledger, and pipeline updates.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | 8+ Dimension Concept Taxonomy | 8+ character and capability dimensions (Identity & Persona, Calibrated Pushback, Epistemic Humility, Non-Sycophantic Feedback, Emotional Attunement, Safe Edge-Case Handling, Strict Verifiable Constraints, Step-by-Step Math) | M1 | ORIGINAL_REQUEST §R1 |
| F2 | 3-Axis Simula Parameterization | Parameterized along Quality, Diversity (diverse prompts/framings), and Complexity (low/medium/high) | M1 | ORIGINAL_REQUEST §R1 |
| F3 | Deterministic Formal/Factual Critic | Verifies task correctness, zero hallucinations, or deterministic constraints (`src/rlvr_verify.py` + `is_math_correct`) | M1 | ORIGINAL_REQUEST §R1 |
| F4 | Constitutional Voice Critic | Evaluates tone, warmth, conciseness, non-preachiness against `constitution/constitution.md` with threshold $\ge 7/10$ | M1 | ORIGINAL_REQUEST §R1 |
| F5 | Serial Dual-Critic Filtering | Pipeline rejects invalid candidates at Critic 1 before invoking Critic 2, retaining valid pairs in `data/train/s4_projection.jsonl` | M1 | ORIGINAL_REQUEST §R1 |
| F6 | Inoculation Prompting Formatter | Prepends `[Mode: Verifiable Constraint Following]\n` to all verifiable formatting tasks | M2 | ORIGINAL_REQUEST §R3 |
| F7 | Projection-Lite Math Reasoning | Keeps student greedy when correct; generates $K=4$ hint-guided proposals and ranks by unconditioned log-likelihood (`batch_logps`) | M2 | ORIGINAL_REQUEST §R3 |
| F8 | Centralized Contamination Screening | Rigorous exclusion against `evals/heldout_ids.json` (`src_ids`, `base_q`, `texts`) and `evals/probes.jsonl` | M2 | ORIGINAL_REQUEST §R3 |
| F9 | High-Concurrency Teacher Dispatch | Executes batch calls through `src/teacher.py` with 64–128 worker threads and cached warm-up prefix | M3 | ORIGINAL_REQUEST §R2 |
| F10 | Budget Gating & Ledger Accounting | Strict ledger logging in `ledger/spend.jsonl`; explicit campaign spend cap $\le \$8.00$ and cumulative spend $\le \$12.13$ | M3 | ORIGINAL_REQUEST §R2 |
| F11 | Resource Health Guardrails | Single MLX process enforcement, `mx.clear_cache()` between runs, disk space floor $\ge 5$ GB | M3 | ORIGINAL_REQUEST §R4 |
| F12 | Unit Test Suite Expansion | Unit tests in `tests/test_pipelines.py` covering taxonomy, dual critics, decontamination, projection-lite, and budget limits | M4 | ORIGINAL_REQUEST §R4 |
| F13 | Living Article Documentation | Update `article/wren.html` with Simula taxonomy narrative, visual grid, empirical yield table, spend ledger, and Step 7 pipeline | M4 | ORIGINAL_REQUEST §R4 |
| F14 | End-to-End Acceptance & Verification | 100% pass on comprehensive E2E test suite (Tiers 1–4) followed by Tier 5 adversarial hardening | M5 | ORIGINAL_REQUEST Acceptance Criteria |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Simula Taxonomy & Dual-Critic Architecture | Implement `src/simula_taxonomy.py` (8+ dimensions, 3 axes) and `src/critics.py` (Formal Critic + Constitutional Voice Critic $\ge 7/10$) | none | **DONE** (`src/simula_taxonomy.py`, `src/critics.py`, 239/239 tests pass) |
| M2 | Inoculation, Projection-Lite & Decontamination | Enhance `src/projection_data.py` (inoculation + $K=4$ proposal likelihood ranking via `batch_logps`) and `src/decontamination.py` (`heldout_ids.json` + `probes.jsonl`) | none | **DONE** (`src/decontamination.py`, `src/projection_data.py`, 318/318 tests pass) |
| M3 | Concurrency Driver, Budget Gating & Generation | Implement `src/simula_generator.py` integrating teacher batch generation (64–128 workers), budget cap ($\le \$8.00$ campaign, $\le \$12.13$ cumulative), resource locks, and output to `data/train/s4_projection.jsonl` | M1, M2 | **DONE** (`src/simula_generator.py`, 357/357 tests pass) |
| M4 | Test Expansion & Living Article Update | Expand `tests/test_pipelines.py` with full component test coverage; update `article/wren.html` with Simula narrative, taxonomy, yield metrics, and spend ledger | M1, M2, M3 | **DONE** (`article/wren.html`, `tests/test_pipelines.py`, 413/413 tests pass) |
| M5 | Final E2E Test Pass & Adversarial Hardening | Phase 1: 100% E2E test pass across Tiers 1–4; Phase 2: Tier 5 adversarial testing and capability validation | M4 | **IN PROGRESS** |

---

## Interface Contracts

### `src/simula_taxonomy.py`
```python
def get_taxonomy_categories() -> list[str]:
    """Returns list of 8+ character and capability dimensions."""
    ...

def generate_prompts_for_category(category: str, complexity: str, n: int) -> list[dict]:
    """
    Returns list of prompt metadata dicts:
    {'id': str, 'category': str, 'complexity': 'low'|'medium'|'high', 'prompt': str, 'constraint_info': dict | None}
    """
    ...
```

### `src/critics.py`
```python
def evaluate_formal_critic(prompt: str, completion: str, constraint_info: dict | None) -> tuple[bool, str]:
    """Returns (passed: bool, reason: str). Uses rlvr_verify and is_math_correct."""
    ...

def evaluate_constitutional_critic(prompt: str, completion: str) -> tuple[float, str]:
    """Returns (score: float [1.0 - 10.0], rationale: str). Threshold for retention is >= 7.0."""
    ...

def dual_critic_pipeline(candidate: dict) -> tuple[bool, dict]:
    """
    Runs Formal Critic first (cost-free). If passed, runs Constitutional Voice Critic.
    Returns (accepted: bool, metadata: dict with critic scores).
    """
    ...
```

### `src/decontamination.py`
```python
class ContaminationFilter:
    def __init__(self, heldout_path: str = "evals/heldout_ids.json", probes_path: str = "evals/probes.jsonl"):
        ...
    def is_contaminated(self, text: str, src_id: str | None = None) -> bool:
        """Returns True if sample matches any held-out probe text, base_q substring, or src_id."""
        ...
```

### `src/projection_data.py`
```python
def format_inoculated_ifeval(prompt: str, completion: str, constraint_info: dict = None) -> dict:
    """Prepends '[Mode: Verifiable Constraint Following]\n' to prompt."""
    ...

def rank_math_proposals_by_likelihood(model, tok, prompt: str, candidates: list[str]) -> str:
    """Ranks candidate solutions using unconditioned log-likelihood from batch_logps, returning best."""
    ...
```

### `src/simula_generator.py`
```python
def run_simula_campaign(
    max_campaign_spend: float = 8.00,
    cumulative_spend_limit: float = 12.13,
    concurrency_workers: int = 64,
    dry_run: bool = False
) -> dict:
    """Executes high-concurrency generation, dual-critic filtering, budget gating, and outputs to data/train/s4_projection.jsonl."""
    ...
```

---

## Code Layout
- `src/simula_taxonomy.py`: Simula concept taxonomy definitions, prompt templates, and 3-axis parameterization.
- `src/critics.py`: Deterministic formal critic + Constitutional voice critic with dual-critic evaluation pipeline.
- `src/decontamination.py`: Screening against `evals/heldout_ids.json` and `evals/probes.jsonl`.
- `src/projection_data.py`: Inoculation formatting and Projection-Lite math reasoning ($K$-sample proposals + log-likelihood ranking).
- `src/simula_generator.py`: Concurrency runner, budget gating, resource mutex, and dataset builder.
- `tests/test_pipelines.py`: Unit test suite covering all pipeline components.
- `article/wren.html`: Living article presentation.
- `data/train/s4_projection.jsonl`: Final decontaminated, dual-critic-approved training dataset.
