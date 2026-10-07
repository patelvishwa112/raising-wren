"""Empirical Adversarial Stress Harness for Milestone 3 Remediation (Iteration 2).
Authored by Challenger 2 (Empirical Challenger).

Executes comprehensive stress tests across:
1. Zero and negative budget limits:
   - max_campaign_spend <= 0.0 (0.0, -0.0, -1.0, -100.0)
   - cumulative_spend_limit > 12.13 and max_campaign_spend > 8.00 raise ValueError
   - micro-budgets (less than 1 item)
   - concurrency budget exhaustion: asserts total_campaign_spend <= max_campaign_spend
   - cumulative spend limit enforcement: asserts initial + total <= cumulative_spend_limit
2. Large concurrency worker scaling:
   - 64 workers with 64 clean prompts
   - 128 workers with 128 clean prompts
   - 128 workers processing 256 clean items
   - Multi-campaign lock serialization (_MLX_PROCESS_LOCK)
3. Verifiable constraints and math accuracy:
   - All 20 RLVR constraint types across low/medium/high
   - Step-by-step math reasoning across integer, float, scientific, negative, and zero golds
   - Formal critic verification
   - Offline constitutional critic threshold compliance (>= 7.0)
   - Inoculation prefix single-prepend invariant
   - Contamination filtering for held-out IDs and probe texts
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.critics import (
    evaluate_constitutional_critic,
    evaluate_formal_critic,
    dual_critic_pipeline,
    CONSTITUTIONAL_THRESHOLD,
)
from src.decontamination import get_contamination_filter
from src.projection_data import INOCULATION_PREFIX, is_math_correct
from src.rlvr_verify import verify as rlvr_verify
from src.simula_generator import (
    DEFAULT_MAX_CAMPAIGN_SPEND,
    DEFAULT_CUMULATIVE_SPEND_LIMIT,
    SimulaBudgetTracker,
    BudgetExceeded,
    _MLX_PROCESS_LOCK,
    check_disk_space,
    check_resource_guardrails,
    clear_mlx_cache,
    generate_mock_completion,
    get_ledger_spend,
    run_simula_campaign,
)
from src.simula_taxonomy import (
    _generate_constraint_item,
    generate_full_campaign_prompt_pool,
    generate_prompts_for_category,
    get_taxonomy_categories,
)


def run_budget_stress_tests() -> list[tuple[str, bool, str]]:
    results = []

    # Test 1.1: Zero and negative campaign spend
    for val in [0.0, -0.0, -0.001, -1.0, -50.0]:
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "out.jsonl"
            led_p = Path(td) / "led.jsonl"
            res = run_simula_campaign(
                max_campaign_spend=val,
                dry_run=True,
                output_path=out_p,
                ledger_path=led_p,
                total_prompts=10,
            )
            passed = (
                res["status"] == "success"
                and res["generated_count"] == 0
                and res["accepted_count"] == 0
                and res["total_campaign_spend"] == 0.0
                and out_p.exists()
                and out_p.read_text().strip() == ""
            )
            results.append((f"Budget <= 0.0 ({val}) handled cleanly", passed, f"res={res}"))

    # Test 1.2: Upper boundary validations
    for bad_cum in [12.1301, 13.0, 100.0]:
        try:
            run_simula_campaign(cumulative_spend_limit=bad_cum, dry_run=True)
            results.append((f"Cumulative spend > 12.13 ({bad_cum}) raises ValueError", False, "No error raised"))
        except ValueError:
            results.append((f"Cumulative spend > 12.13 ({bad_cum}) raises ValueError", True, "Raised ValueError"))

    for bad_camp in [8.0001, 8.5, 20.0]:
        try:
            run_simula_campaign(max_campaign_spend=bad_camp, dry_run=True)
            results.append((f"Campaign spend > 8.00 ({bad_camp}) raises ValueError", False, "No error raised"))
        except ValueError:
            results.append((f"Campaign spend > 8.00 ({bad_camp}) raises ValueError", True, "Raised ValueError"))

    # Test 1.3: Micro-budget smaller than 1 item (cost_per_item is 0.0005)
    with tempfile.TemporaryDirectory() as td:
        out_p = Path(td) / "out.jsonl"
        led_p = Path(td) / "led.jsonl"
        pool = [{"id": "p1", "category": "balanced_perspectives", "prompt": "Examine perspectives on solar."}]
        res = run_simula_campaign(
            max_campaign_spend=0.0001,
            dry_run=True,
            output_path=out_p,
            ledger_path=led_p,
            prompt_pool=pool,
        )
        passed = (
            res["accepted_count"] == 0
            and res["total_campaign_spend"] == 0.0
            and out_p.read_text().strip() == ""
        )
        results.append(("Micro-budget (< 1 item cost) halts with 0 items", passed, f"res={res}"))

    # Test 1.4: Budget cap enforcement under concurrency (must not exceed max_campaign_spend)
    with tempfile.TemporaryDirectory() as td:
        out_p = Path(td) / "out.jsonl"
        led_p = Path(td) / "led.jsonl"
        pool = generate_full_campaign_prompt_pool(total_n=100, seed=42)[:60]
        # Target limit: 0.0030 USD (max 6 items)
        target_limit = 0.0030
        res = run_simula_campaign(
            max_campaign_spend=target_limit,
            concurrency_workers=32,
            dry_run=True,
            output_path=out_p,
            ledger_path=led_p,
            prompt_pool=pool,
        )
        lines = [json.loads(l) for l in out_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        # DEFECT CHECK: Spend must NEVER exceed the configured cap
        spend_within_cap = res["total_campaign_spend"] <= target_limit
        accounting_synced = len(lines) == res["accepted_count"]
        passed = spend_within_cap and accounting_synced
        results.append((
            f"Campaign spend cap strictly enforced under concurrency (spend <= {target_limit})",
            passed,
            f"target_limit={target_limit}, actual_spend={res['total_campaign_spend']}, accepted={res['accepted_count']}",
        ))

    # Test 1.5: Cumulative spend ceiling enforcement under concurrency
    with tempfile.TemporaryDirectory() as td:
        led_p = Path(td) / "spend.jsonl"
        out_p = Path(td) / "out.jsonl"
        # Seed ledger with 12.1260 USD (remaining headroom = 0.0040 USD, max 8 items)
        initial_spend = 12.1260
        cum_limit = 12.1300
        led_p.write_text(json.dumps({"cost": initial_spend}) + "\n")
        pool = generate_full_campaign_prompt_pool(total_n=100, seed=42)[:60]
        res = run_simula_campaign(
            max_campaign_spend=8.00,
            cumulative_spend_limit=cum_limit,
            concurrency_workers=32,
            dry_run=True,
            output_path=out_p,
            ledger_path=led_p,
            prompt_pool=pool,
        )
        total_cum_spend = initial_spend + res["total_campaign_spend"]
        # DEFECT CHECK: Cumulative spend must NEVER exceed 12.13
        passed = round(total_cum_spend, 6) <= cum_limit
        results.append((
            f"Cumulative spend limit strictly enforced under concurrency (cumulative <= {cum_limit})",
            passed,
            f"cum_limit={cum_limit}, initial={initial_spend}, campaign_spend={res['total_campaign_spend']}, total_cum={total_cum_spend}",
        ))

    return results


def run_concurrency_stress_tests() -> list[tuple[str, bool, str]]:
    results = []
    cfilter = get_contamination_filter()

    # Generate verified clean prompts for pure concurrency stress
    clean_pool = [
        {"id": f"clean_{i:04d}", "category": "balanced_perspectives", "prompt": f"Discuss aspect number {i} of clean renewable technology."}
        for i in range(256)
    ]
    assert all(not cfilter.is_contaminated(p["prompt"]) for p in clean_pool)

    # Test 2.1: 64 workers with 64 prompts
    with tempfile.TemporaryDirectory() as td:
        out_p = Path(td) / "out_64.jsonl"
        led_p = Path(td) / "led_64.jsonl"
        res = run_simula_campaign(
            concurrency_workers=64,
            dry_run=True,
            output_path=out_p,
            ledger_path=led_p,
            prompt_pool=clean_pool[:64],
        )
        lines = [json.loads(l) for l in out_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        passed = (
            res["status"] == "success"
            and res["accepted_count"] == 64
            and len(lines) == 64
            and all("messages" in l and "completion" in l and "kind" in l for l in lines)
            and len({l["messages"][0]["content"] for l in lines}) == 64
        )
        results.append(("64 concurrent workers with 64 prompts: zero corruption & 100% throughput", passed, f"lines={len(lines)}"))

    # Test 2.2: 128 workers with 128 prompts
    with tempfile.TemporaryDirectory() as td:
        out_p = Path(td) / "out_128.jsonl"
        led_p = Path(td) / "led_128.jsonl"
        res = run_simula_campaign(
            concurrency_workers=128,
            dry_run=True,
            output_path=out_p,
            ledger_path=led_p,
            prompt_pool=clean_pool[:128],
        )
        lines = [json.loads(l) for l in out_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        passed = (
            res["status"] == "success"
            and res["accepted_count"] == 128
            and len(lines) == 128
            and all("messages" in l and "completion" in l and "kind" in l for l in lines)
            and len({l["messages"][0]["content"] for l in lines}) == 128
        )
        results.append(("128 concurrent workers with 128 prompts: zero corruption & 100% throughput", passed, f"lines={len(lines)}"))

    # Test 2.3: 128 workers with 256 prompts
    with tempfile.TemporaryDirectory() as td:
        out_p = Path(td) / "out_256.jsonl"
        led_p = Path(td) / "led_256.jsonl"
        res = run_simula_campaign(
            concurrency_workers=128,
            dry_run=True,
            output_path=out_p,
            ledger_path=led_p,
            prompt_pool=clean_pool[:256],
        )
        lines = [json.loads(l) for l in out_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        passed = (
            res["status"] == "success"
            and res["accepted_count"] == 256
            and len(lines) == 256
            and len({l["messages"][0]["content"] for l in lines}) == 256
        )
        results.append(("128 workers processing 256 items: complete throughput & no dropped rows", passed, f"lines={len(lines)}"))

    # Test 2.4: Multi-campaign serialization under _MLX_PROCESS_LOCK
    with tempfile.TemporaryDirectory() as td:
        num_campaigns = 4
        threads = []
        campaign_results = [None] * num_campaigns
        errs = []

        def worker(idx):
            out = Path(td) / f"campaign_{idx}.jsonl"
            led = Path(td) / f"spend_{idx}.jsonl"
            try:
                r = run_simula_campaign(
                    concurrency_workers=16,
                    dry_run=True,
                    output_path=out,
                    ledger_path=led,
                    total_prompts=8,
                )
                campaign_results[idx] = r
            except Exception as e:
                errs.append((idx, e))

        for i in range(num_campaigns):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=15.0)

        passed = (
            len(errs) == 0
            and all(r is not None and r["status"] == "success" for r in campaign_results)
            and all((Path(td) / f"campaign_{i}.jsonl").exists() for i in range(num_campaigns))
        )
        results.append(("Multi-campaign lock serialization across 4 concurrent threads", passed, f"errs={errs}"))

    return results


def run_verifiable_constraints_and_math_tests() -> list[tuple[str, bool, str]]:
    results = []

    # Test 3.1: All 20 RLVR constraint types
    constraints = [
        ("validate_lowercase", {"func_name": "validate_lowercase"}),
        ("validate_uppercase", {"func_name": "validate_uppercase"}),
        ("validate_no_commas", {"func_name": "validate_no_commas"}),
        ("validate_title", {"func_name": "validate_title"}),
        ("validate_end", {"func_name": "validate_end", "end_phrase": "That concludes this summary."}),
        ("validate_quotation", {"func_name": "validate_quotation"}),
        ("verify_bullet_points_3", {"func_name": "verify_bullet_points", "N": 3}),
        ("verify_bullet_points_4", {"func_name": "verify_bullet_points", "N": 4}),
        ("verify_bullet_points_5", {"func_name": "verify_bullet_points", "N": 5}),
        ("validate_word_constraint_at_most", {"func_name": "validate_word_constraint", "N": 40, "quantifier": "at most"}),
        ("validate_word_constraint_at_least", {"func_name": "validate_word_constraint", "N": 10, "quantifier": "at least"}),
        ("verify_sentence_constraint_2", {"func_name": "verify_sentence_constraint", "N": 2, "quantifier": "exactly"}),
        ("verify_sentence_constraint_3", {"func_name": "verify_sentence_constraint", "N": 3, "quantifier": "exactly"}),
        ("verify_keywords", {"func_name": "verify_keywords", "keyword_list": ["energy", "system", "process"]}),
        ("validate_json_format", {"func_name": "validate_json_format"}),
        ("validate_forbidden_words", {"func_name": "validate_forbidden_words", "forbidden_words": ["very", "really", "important"]}),
        ("validate_two_responses", {"func_name": "validate_two_responses"}),
        ("verify_paragraph_count", {"func_name": "verify_paragraph_count", "N": 3}),
        ("verify_postscript", {"func_name": "verify_postscript", "postscript_marker": "P.S."}),
        ("validate_repeat_prompt", {"func_name": "validate_repeat_prompt", "original_prompt": "First, repeat the request for details."}),
    ]

    all_c_passed = True
    c_failures = []
    for name, cinfo in constraints:
        meta = {
            "category": "strict_verifiable_constraints",
            "prompt": f"Fulfill {name}",
            "constraint_info": cinfo,
        }
        comp = generate_mock_completion(meta)
        v_ok = rlvr_verify(comp, cinfo)
        f_ok, f_reason = evaluate_formal_critic(meta["prompt"], comp, cinfo)
        if not (v_ok and f_ok):
            all_c_passed = False
            c_failures.append((name, v_ok, f_ok, f_reason))

    results.append((
        f"All {len(constraints)} RLVR constraints pass both rlvr_verify and formal critic (100%)",
        all_c_passed,
        f"failures={c_failures}",
    ))

    # Test 3.2: Diverse math reasoning gold answers
    diverse_golds = [
        0, 0.0, -0.0, 1, 42, -42, -999, 1000000, 9876543210,
        3.14159, 0.0001, -273.15, 0.5, "42", "0", "-15", "3.14",
        "1,000,000", 1e6, -1e4, 1e-4, "0.0", "-0.0"
    ]
    all_math_passed = True
    math_failures = []
    for g in diverse_golds:
        meta = {
            "category": "step_by_step_math",
            "prompt": "Calculate step by step.",
            "constraint_info": {"gold": g},
        }
        comp = generate_mock_completion(meta)
        m_ok = is_math_correct(comp, g)
        f_ok, f_reason = evaluate_formal_critic(meta["prompt"], comp, {"gold": g})
        if not (m_ok and f_ok):
            all_math_passed = False
            math_failures.append((g, m_ok, f_ok, f_reason, comp))

    results.append((
        f"Diverse math gold values ({len(diverse_golds)} values incl 0, -0, negatives, decimals, scientific)",
        all_math_passed,
        f"failures={math_failures}",
    ))

    # Test 3.3: Constitutional voice critic score >= 7.0 for all taxonomy categories
    categories = get_taxonomy_categories()
    voice_scores = {}
    voice_passed = True
    for cat in categories:
        meta = {"category": cat, "prompt": f"Explain key considerations regarding {cat}."}
        comp = generate_mock_completion(meta)
        score, rationale = evaluate_constitutional_critic(meta["prompt"], comp, offline=True)
        voice_scores[cat] = score
        if score < CONSTITUTIONAL_THRESHOLD:
            voice_passed = False

    results.append((
        f"Constitutional Voice Critic >= {CONSTITUTIONAL_THRESHOLD} for all {len(categories)} taxonomy dimensions",
        voice_passed,
        f"scores={voice_scores}",
    ))

    # Test 3.4: Inoculation Prefix isolation and single prepend
    with tempfile.TemporaryDirectory() as td:
        out_p = Path(td) / "inoc_test.jsonl"
        led_p = Path(td) / "spend_test.jsonl"
        p_verifiable = generate_prompts_for_category("strict_verifiable_constraints", "medium", 5)
        p_math = generate_prompts_for_category("step_by_step_math", "medium", 3)
        p_persona = generate_prompts_for_category("identity_persona", "medium", 3)
        pool = p_verifiable + p_math + p_persona
        res = run_simula_campaign(
            prompt_pool=pool,
            output_path=out_p,
            ledger_path=led_p,
            dry_run=True,
        )
        lines = [json.loads(l) for l in out_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        inoc_ok = True
        inoc_errs = []
        for r in lines:
            txt = r["messages"][0]["content"]
            kind = r["kind"]
            if kind == "inoculated_ifeval":
                if not txt.startswith(INOCULATION_PREFIX):
                    inoc_ok = False
                    inoc_errs.append(f"Missing prefix: {txt[:40]}")
                if txt.count(INOCULATION_PREFIX) != 1:
                    inoc_ok = False
                    inoc_errs.append(f"Prefix count != 1 ({txt.count(INOCULATION_PREFIX)}): {txt[:50]}")
            else:
                if txt.startswith(INOCULATION_PREFIX):
                    inoc_ok = False
                    inoc_errs.append(f"Unexpected prefix on non-verifiable: {txt[:40]}")

        results.append((
            "Inoculation prefix isolation & kind=inoculated_ifeval single-prepend verification",
            inoc_ok,
            f"errors={inoc_errs}",
        ))

    # Test 3.5: Decontamination filtering against heldout_ids & probes
    cfilter = get_contamination_filter()
    with tempfile.TemporaryDirectory() as td:
        out_p = Path(td) / "decontam_test.jsonl"
        led_p = Path(td) / "spend_test.jsonl"
        heldout_id = next(iter(cfilter.src_ids)) if cfilter.src_ids else "heldout-probe-01"
        pool = [
            {"id": "clean_1", "category": "balanced_perspectives", "prompt": "Examine perspectives on solar energy deployment."},
            {"id": "probe_1", "category": "identity_persona", "prompt": "Who are you?"},
            {"id": heldout_id, "category": "identity_persona", "prompt": "Novel prompt with heldout id."},
        ]
        res = run_simula_campaign(
            prompt_pool=pool,
            output_path=out_p,
            ledger_path=led_p,
            dry_run=True,
        )
        lines = [json.loads(l) for l in out_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        decontam_ok = (
            res["accepted_count"] == 1
            and res["rejected_count"] == 2
            and len(lines) == 1
            and lines[0]["messages"][0]["content"] == "Examine perspectives on solar energy deployment."
        )
        results.append((
            "Decontamination screening: probes and heldout IDs rejected from campaign output",
            decontam_ok,
            f"accepted={res['accepted_count']}, rejected={res['rejected_count']}",
        ))

    return results


def main() -> int:
    print("=" * 80)
    print("EMPIRICAL ADVERSARIAL STRESS HARNESS — MILESTONE 3 REMEDIATION (ITERATION 2)")
    print("=" * 80)

    all_suites = [
        ("1. Zero and Negative Budget Limits & Caps", run_budget_stress_tests),
        ("2. Large Concurrency Worker Counts (64, 128, 256)", run_concurrency_stress_tests),
        ("3. Verifiable Constraints and Math Accuracy", run_verifiable_constraints_and_math_tests),
    ]

    total_tests = 0
    total_passed = 0
    total_failed = 0

    t0 = time.time()

    for suite_name, suite_fn in all_suites:
        print(f"\n--- Running Suite: {suite_name} ---")
        st_start = time.time()
        suite_results = suite_fn()
        st_elapsed = time.time() - st_start

        for desc, passed, detail in suite_results:
            total_tests += 1
            if passed:
                total_passed += 1
                status_str = "[PASS]"
            else:
                total_failed += 1
                status_str = "[FAIL]"
            print(f"  {status_str} {desc}")
            if not passed:
                print(f"         Detail: {detail}")

        print(f"Suite completed in {st_elapsed:.3f}s")

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"SUMMARY: Ran {total_tests} adversarial stress tests in {elapsed:.3f}s")
    print(f"PASSED:  {total_passed}/{total_tests}")
    print(f"FAILED:  {total_failed}/{total_tests}")
    print("=" * 80)

    if total_failed == 0:
        print("\nOVERALL VERDICT: APPROVE")
        return 0
    else:
        print("\nOVERALL VERDICT: CHALLENGE_FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
