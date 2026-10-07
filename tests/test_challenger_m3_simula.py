"""Adversarial Verification Suite for Milestone 3 (Simula Generator & Concurrency).
Authored by Challenger 1 (Empirical Challenger).

Stress-tests:
1. Budget boundary conditions (8.00 vs 8.0001, 12.13 vs 12.1301, zero, negative).
2. Resource guardrails (_MLX_PROCESS_LOCK non-reentrancy, disk floor 5.0 GB, clear_mlx_cache resilience).
3. Thread safety and JSONL data integrity under 64 and 128 concurrent workers.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.simula_generator as simgen
from src.simula_generator import (
    DEFAULT_MAX_CAMPAIGN_SPEND,
    DEFAULT_CUMULATIVE_SPEND_LIMIT,
    MIN_DISK_FREE_GB,
    SimulaBudgetTracker,
    _MLX_PROCESS_LOCK,
    check_disk_space,
    check_resource_guardrails,
    clear_mlx_cache,
    generate_mock_completion,
    get_ledger_spend,
    run_simula_campaign,
)
from src.simula_taxonomy import (
    generate_full_campaign_prompt_pool,
    generate_prompts_for_category,
    get_taxonomy_categories,
)
from src.projection_data import INOCULATION_PREFIX


class TestBudgetBoundaryAdversarial(unittest.TestCase):
    """Stress tests on budget limits and boundary conditions."""

    def test_campaign_spend_boundary_exact_limits(self):
        """Verify max_campaign_spend boundary behavior."""
        # Exact upper limit: 8.00 must succeed
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "out.jsonl"
            led_p = Path(td) / "led.jsonl"
            res = run_simula_campaign(
                max_campaign_spend=8.00,
                dry_run=True,
                output_path=out_p,
                ledger_path=led_p,
                total_prompts=2,
            )
            self.assertEqual(res["status"], "success")

        # Exceeding by tiny epsilons must raise ValueError
        invalid_campaign_spends = [8.0001, 8.001, 8.01, 8.1, 9.0, 100.0]
        for val in invalid_campaign_spends:
            with self.assertRaises(ValueError, msg=f"Spend {val} should raise ValueError"):
                run_simula_campaign(max_campaign_spend=val)
            with self.assertRaises(ValueError, msg=f"Tracker with spend {val} should raise ValueError"):
                SimulaBudgetTracker(max_campaign_spend=val)

    def test_cumulative_spend_boundary_exact_limits(self):
        """Verify cumulative_spend_limit boundary behavior."""
        # Exact upper limit: 12.13 must succeed
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "out.jsonl"
            led_p = Path(td) / "led.jsonl"
            res = run_simula_campaign(
                max_campaign_spend=1.00,
                cumulative_spend_limit=12.13,
                dry_run=True,
                output_path=out_p,
                ledger_path=led_p,
                total_prompts=2,
            )
            self.assertEqual(res["status"], "success")

        # Exceeding by tiny epsilons must raise ValueError
        invalid_cumulative_spends = [12.1301, 12.131, 12.14, 12.5, 15.0, 20.0]
        for val in invalid_cumulative_spends:
            with self.assertRaises(ValueError, msg=f"Cumulative {val} should raise ValueError"):
                run_simula_campaign(cumulative_spend_limit=val)
            with self.assertRaises(ValueError, msg=f"Tracker with cumulative {val} should raise ValueError"):
                SimulaBudgetTracker(cumulative_spend_limit=val)

    def test_zero_and_negative_spend_values_handled_gracefully(self):
        """Verify zero and negative spend values execute gracefully without unhandled exceptions."""
        test_values = [0.0, -0.0, -0.001, -1.0, -100.0]
        for val in test_values:
            with tempfile.TemporaryDirectory() as td:
                out_p = Path(td) / "s4_out.jsonl"
                led_p = Path(td) / "spend.jsonl"

                res = run_simula_campaign(
                    max_campaign_spend=val,
                    output_path=out_p,
                    ledger_path=led_p,
                    total_prompts=10,
                )
                self.assertEqual(res["status"], "success")
                self.assertEqual(res["generated_count"], 0)
                self.assertEqual(res["accepted_count"], 0)
                self.assertEqual(res["total_campaign_spend"], 0.0)
                self.assertTrue(out_p.exists(), f"Output file must exist for spend {val}")
                self.assertTrue(led_p.exists(), f"Ledger file must exist for spend {val}")

                # Check ledger record is valid json with cost 0.0
                lines = [l.strip() for l in led_p.read_text(encoding="utf-8").splitlines() if l.strip()]
                self.assertGreaterEqual(len(lines), 1)
                rec = json.loads(lines[-1])
                self.assertEqual(rec["cost"], 0.0)

    def test_budget_tracker_reservation_overflow_rejection(self):
        """Verify tracker halts and stops reservations when budget is depleted."""
        with tempfile.TemporaryDirectory() as td:
            led_p = Path(td) / "spend.jsonl"
            tracker = SimulaBudgetTracker(
                max_campaign_spend=0.002,
                cumulative_spend_limit=10.0,
                ledger_path=led_p,
            )
            # 1st reservation: 0.001 (OK)
            self.assertTrue(tracker.check_and_reserve(0.001))
            # 2nd reservation: 0.001 (OK, total reserved = 0.002)
            self.assertTrue(tracker.check_and_reserve(0.001))
            # 3rd reservation: 0.0001 (exceeds 0.002, must fail)
            self.assertFalse(tracker.check_and_reserve(0.0001))
            self.assertTrue(tracker.stop_event.is_set())

            # Release one reservation
            tracker.release_reservation(0.001)
            self.assertEqual(tracker.reserved, 0.001)


class TestResourceGuardrailsAdversarial(unittest.TestCase):
    """Stress tests on resource guardrails (disk floor, lock, MLX cache)."""

    def test_mlx_process_lock_non_reentrancy_and_exclusivity(self):
        """Verify _MLX_PROCESS_LOCK is non-reentrant on same thread and exclusive across threads."""
        # 1. Non-reentrancy on same thread
        acquired = _MLX_PROCESS_LOCK.acquire(blocking=False)
        self.assertTrue(acquired, "First acquisition must succeed")
        try:
            second = _MLX_PROCESS_LOCK.acquire(blocking=False)
            self.assertFalse(second, "Second acquisition on same thread must fail (non-reentrant)")
        finally:
            _MLX_PROCESS_LOCK.release()

        # Re-acquisition after release succeeds
        reacquired = _MLX_PROCESS_LOCK.acquire(blocking=False)
        self.assertTrue(reacquired, "Acquisition after release must succeed")
        _MLX_PROCESS_LOCK.release()

        # 2. Cross-thread exclusivity
        lock_held_event = threading.Event()
        release_event = threading.Event()
        thread_acquired = [False]
        thread2_attempted = [False]

        def holder():
            with _MLX_PROCESS_LOCK:
                thread_acquired[0] = True
                lock_held_event.set()
                release_event.wait(timeout=2.0)

        t1 = threading.Thread(target=holder)
        t1.start()
        lock_held_event.wait(timeout=2.0)
        self.assertTrue(thread_acquired[0])

        # Attempt to acquire from main thread while held by t1
        attempt = _MLX_PROCESS_LOCK.acquire(blocking=False)
        self.assertFalse(attempt, "Should not acquire lock when held by another thread")

        # Let t1 release
        release_event.set()
        t1.join(timeout=2.0)

        # Now main thread can acquire
        can_acquire = _MLX_PROCESS_LOCK.acquire(blocking=False)
        self.assertTrue(can_acquire, "Should acquire lock after thread release")
        _MLX_PROCESS_LOCK.release()

    def test_disk_space_floor_adversarial_mock(self):
        """Verify disk space floor strictly rejects free space below 5.0 GB."""
        # Below threshold: 4.99 GB free
        usage_4_99_gb = MagicMock()
        usage_4_99_gb.free = int(4.99 * (1024**3))

        with patch("shutil.disk_usage", return_value=usage_4_99_gb):
            with self.assertRaises(RuntimeError) as ctx:
                check_resource_guardrails(min_disk_gb=5.0)
            self.assertIn("below 5.0 GB threshold", str(ctx.exception))

            with self.assertRaises(RuntimeError):
                check_disk_space(min_free_gb=5.0)

            with tempfile.TemporaryDirectory() as td:
                out_p = Path(td) / "out.jsonl"
                led_p = Path(td) / "led.jsonl"
                with self.assertRaises(RuntimeError):
                    run_simula_campaign(
                        output_path=out_p,
                        ledger_path=led_p,
                        dry_run=True,
                    )

        # At or above threshold: 5.00 GB and 5.01 GB free must succeed
        usage_5_01_gb = MagicMock()
        usage_5_01_gb.free = int(5.01 * (1024**3))

        with patch("shutil.disk_usage", return_value=usage_5_01_gb):
            # Should not raise
            check_resource_guardrails(min_disk_gb=5.0)
            avail = check_disk_space(min_free_gb=5.0)
            self.assertAlmostEqual(avail, 5.01, places=2)

    def test_clear_mlx_cache_resilience(self):
        """Verify clear_mlx_cache handles present, missing, and broken MLX core safely."""
        # 1. Direct call in current environment
        result = clear_mlx_cache()
        self.assertIsInstance(result, bool)

        # 2. When mlx.core is missing (ImportError)
        with patch.dict(sys.modules, {"mlx.core": None, "mlx": None}):
            # In Python, setting a module to None in sys.modules raises ModuleNotFoundError
            res_missing = clear_mlx_cache()
            self.assertFalse(res_missing, "Missing MLX must safely return False")

        # 3. When mlx.core exists but lacks clear_cache attribute
        parent_mock = MagicMock()
        mock_broken_mlx = MagicMock(spec=[])  # no clear_cache attribute
        parent_mock.core = mock_broken_mlx
        with patch.dict(sys.modules, {"mlx": parent_mock, "mlx.core": mock_broken_mlx}):
            res_broken = clear_mlx_cache()
            self.assertFalse(res_broken, "MLX without clear_cache must return False")

        # 4. When mlx.core exists and has clear_cache
        parent_mock_ok = MagicMock()
        mock_working_mlx = MagicMock()
        parent_mock_ok.core = mock_working_mlx
        mock_working_mlx.clear_cache = MagicMock()
        with patch.dict(sys.modules, {"mlx": parent_mock_ok, "mlx.core": mock_working_mlx}):
            res_working = clear_mlx_cache()
            self.assertTrue(res_working, "Valid MLX clear_cache must return True")
            mock_working_mlx.clear_cache.assert_called_once()


class TestHighConcurrencyThreadSafetyAdversarial(unittest.TestCase):
    """Stress tests verifying JSONL integrity and thread safety under 64 and 128 workers."""

    def test_concurrent_worker_stress_64_workers(self):
        """Run campaign with 64 workers and verify zero JSON line corruption or interleaving."""
        total_items = 64
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "s4_stress_64.jsonl"
            led_p = Path(td) / "spend_stress_64.jsonl"

            # Generate pool of 64 distinct prompts
            pool = generate_full_campaign_prompt_pool(total_n=80, seed=101)[:total_items]
            self.assertEqual(len(pool), total_items)
            self.assertEqual(len({p["prompt"] for p in pool}), total_items)

            res = run_simula_campaign(
                max_campaign_spend=8.00,
                cumulative_spend_limit=12.13,
                concurrency_workers=64,
                dry_run=True,
                output_path=out_p,
                ledger_path=led_p,
                prompt_pool=pool,
            )

            self.assertEqual(res["status"], "success")
            self.assertEqual(res["generated_count"], total_items)
            self.assertEqual(res["accepted_count"] + res["rejected_count"], total_items)

            # Verification of output file
            self.assertTrue(out_p.exists())
            raw_content = out_p.read_text(encoding="utf-8")
            raw_lines = [l for l in raw_content.splitlines() if l.strip()]

            self.assertEqual(len(raw_lines), res["accepted_count"])

            seen_prompts = set()
            for idx, line in enumerate(raw_lines):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as err:
                    self.fail(f"Corrupted or interleaved JSON line at index {idx}: {line!r} (Error: {err})")

                # Schema checks
                self.assertIn("messages", obj)
                self.assertIn("completion", obj)
                self.assertIn("kind", obj)
                self.assertFalse(obj.get("thinking", True))

                prompt_content = obj["messages"][0]["content"]
                self.assertNotIn(prompt_content, seen_prompts, f"Duplicate prompt emitted: {prompt_content[:50]}")
                seen_prompts.add(prompt_content)

                # Check inoculation formatting consistency
                if obj["kind"] == "inoculated_ifeval":
                    self.assertTrue(
                        prompt_content.startswith(INOCULATION_PREFIX),
                        f"Inoculated item must start with {INOCULATION_PREFIX}",
                    )
                else:
                    self.assertTrue(obj["kind"].startswith("simula_"))

            # Verification of ledger file
            self.assertTrue(led_p.exists())
            ledger_lines = [l for l in led_p.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertGreaterEqual(len(ledger_lines), 1)
            for idx, l in enumerate(ledger_lines):
                try:
                    rec = json.loads(l)
                except json.JSONDecodeError as err:
                    self.fail(f"Corrupted ledger JSON line {idx}: {l!r} ({err})")
                for k in ("t", "tag", "hit", "miss", "out", "cost"):
                    self.assertIn(k, rec)

    def test_concurrent_worker_stress_128_workers(self):
        """Run campaign with 128 workers and verify zero JSON line corruption or interleaving."""
        total_items = 128
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "s4_stress_128.jsonl"
            led_p = Path(td) / "spend_stress_128.jsonl"

            # Generate pool of 128 prompts across all categories and complexities
            test_pool = generate_full_campaign_prompt_pool(total_n=150, seed=202)[:total_items]
            self.assertEqual(len(test_pool), total_items)
            self.assertEqual(len({p["prompt"] for p in test_pool}), total_items)

            res = run_simula_campaign(
                max_campaign_spend=8.00,
                cumulative_spend_limit=12.13,
                concurrency_workers=128,
                dry_run=True,
                output_path=out_p,
                ledger_path=led_p,
                prompt_pool=test_pool,
            )

            self.assertEqual(res["status"], "success")
            self.assertEqual(res["generated_count"], total_items)
            self.assertEqual(res["accepted_count"] + res["rejected_count"], total_items)

            # Verification of output file
            self.assertTrue(out_p.exists())
            raw_content = out_p.read_text(encoding="utf-8")
            raw_lines = [l for l in raw_content.splitlines() if l.strip()]

            self.assertEqual(
                len(raw_lines),
                res["accepted_count"],
                f"Expected exactly {res['accepted_count']} JSON lines, got {len(raw_lines)}",
            )

            seen_prompts = set()
            for idx, line in enumerate(raw_lines):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as err:
                    self.fail(f"Corrupted or interleaved JSON line at index {idx}: {line!r} (Error: {err})")

                # Schema checks
                self.assertIn("messages", obj)
                self.assertIn("completion", obj)
                self.assertIn("kind", obj)
                self.assertFalse(obj.get("thinking", True))

                prompt_content = obj["messages"][0]["content"]
                self.assertNotIn(prompt_content, seen_prompts, f"Duplicate prompt emitted: {prompt_content[:50]}")
                seen_prompts.add(prompt_content)

            # Verification of ledger file
            self.assertTrue(led_p.exists())
            ledger_lines = [l for l in led_p.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertGreaterEqual(len(ledger_lines), 1)
            for idx, l in enumerate(ledger_lines):
                try:
                    rec = json.loads(l)
                except json.JSONDecodeError as err:
                    self.fail(f"Corrupted ledger JSON line {idx}: {l!r} ({err})")
                for k in ("t", "tag", "hit", "miss", "out", "cost"):
                    self.assertIn(k, rec)

    def test_multi_campaign_concurrency_serialization(self):
        """Verify multiple threads running run_simula_campaign simultaneously are safely serialized by _MLX_PROCESS_LOCK."""
        num_campaign_threads = 3
        threads = []
        errors = []
        results = [None] * num_campaign_threads

        with tempfile.TemporaryDirectory() as td:
            def run_single(idx):
                out = Path(td) / f"out_{idx}.jsonl"
                led = Path(td) / f"led_{idx}.jsonl"
                try:
                    r = run_simula_campaign(
                        max_campaign_spend=8.00,
                        concurrency_workers=16,
                        dry_run=True,
                        output_path=out,
                        ledger_path=led,
                        total_prompts=8,
                    )
                    results[idx] = r
                except Exception as ex:
                    errors.append((idx, ex))

            for i in range(num_campaign_threads):
                t = threading.Thread(target=run_single, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join(timeout=10.0)

            self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
            for i in range(num_campaign_threads):
                self.assertIsNotNone(results[i])
                self.assertEqual(results[i]["status"], "success")
                out_file = Path(td) / f"out_{i}.jsonl"
                lines = [l for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
                self.assertEqual(len(lines), results[i]["accepted_count"])
                for l in lines:
                    json.loads(l)  # must not raise

    def test_budget_halt_accounting_synchronization(self):
        """Stress-test that when budget limit halts a campaign, generated lines on disk match reported counts.
        
        FAILING TEST EXPOSING CRITICAL DEFECT:
        When as_completed breaks on the first 'budget_stopped' future (which completes
        in microseconds), in-flight generation tasks write lines to out_file that are
        never aggregated into res['accepted_count'] or current_campaign_spend.
        """
        with tempfile.TemporaryDirectory() as td:
            led_p = Path(td) / "spend.jsonl"
            out_p = Path(td) / "out.jsonl"
            pool = generate_full_campaign_prompt_pool(total_n=80, seed=101)[:40]
            res = run_simula_campaign(
                max_campaign_spend=0.002,
                concurrency_workers=32,
                dry_run=True,
                output_path=out_p,
                ledger_path=led_p,
                prompt_pool=pool,
            )
            self.assertIn(res["status"], ("success", "budget_stopped"))
            lines = [json.loads(l) for l in out_p.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(
                res["accepted_count"],
                len(lines),
                f"Accounting desync: reported accepted_count ({res['accepted_count']}) != lines written to disk ({len(lines)})",
            )
            self.assertAlmostEqual(
                res["total_campaign_spend"],
                len(lines) * 0.0005,
                places=5,
                msg="Reported spend must accurately match all records written to disk",
            )

    def test_teacher_budget_exceeded_error_propagation(self):
        """Verify that BudgetExceeded from teacher is not silently swallowed as mock generation.
        
        FAILING TEST EXPOSING MAJOR DEFECT:
        src/simula_generator.py:459 catches Exception broadly and falls back to
        generate_mock_completion even when teacher raises BudgetExceeded.
        """
        mock_teacher = MagicMock()
        mock_teacher.chat.side_effect = simgen.BudgetExceeded("Teacher API budget limit exceeded")

        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "out.jsonl"
            led_p = Path(td) / "spend.jsonl"
            pool = [{"id": "p_clean", "category": "identity_persona", "prompt": "Explain the fundamental theorem of calculus in two sentences."}]

            with patch.dict(sys.modules, {"src.teacher": mock_teacher}):
                # A hard budget exceeded error must either raise or halt generation with 0 accepted
                res = run_simula_campaign(
                    max_campaign_spend=8.00,
                    concurrency_workers=1,
                    mock_mode=False,
                    output_path=out_p,
                    ledger_path=led_p,
                    prompt_pool=pool,
                )
                self.assertEqual(
                    res["accepted_count"],
                    0,
                    "BudgetExceeded from teacher must not silently fall back to mock completion",
                )

    def test_double_ledger_accounting_in_live_dispatch(self):
        """Verify that live mode generation does not double-count teacher per-call spend with aggregate batch spend.
        
        FAILING TEST EXPOSING MAJOR DEFECT:
        src/simula_generator.py:533 logs aggregate batch cost into ledger/spend.jsonl,
        while teacher.chat logs each call individually, causing get_ledger_spend to sum both.
        """
        mock_teacher = MagicMock()
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "out.jsonl"
            led_p = Path(td) / "spend.jsonl"

            def fake_chat(**kwargs):
                # Simulates teacher.chat logging individual transaction
                with open(led_p, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"tag": "teacher_chat", "cost": 0.01}) + "\n")
                return (
                    "The fundamental theorem of calculus connects differentiation with integration. It shows they are inverse operations.",
                    {"cost": 0.01},
                )

            mock_teacher.chat.side_effect = fake_chat
            pool = [{"id": "p_clean", "category": "identity_persona", "prompt": "Explain the fundamental theorem of calculus in two sentences."}]

            with patch.dict(sys.modules, {"src.teacher": mock_teacher}):
                res = run_simula_campaign(
                    max_campaign_spend=8.00,
                    concurrency_workers=1,
                    mock_mode=False,
                    output_path=out_p,
                    ledger_path=led_p,
                    prompt_pool=pool,
                )

            total_ledger = get_ledger_spend(led_p)
            self.assertAlmostEqual(
                total_ledger,
                0.01,
                places=4,
                msg=f"Ledger double accounting: expected $0.01, but ledger records ${total_ledger:.4f}",
            )


if __name__ == "__main__":
    unittest.main()
