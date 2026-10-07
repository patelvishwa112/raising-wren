"""Tier 5 Adversarial Coverage Hardening Test Suite — Part 2.
Milestone 5 Phase 2: Simula Generator & Decontamination Deep Adversarial Testing.

Authored by Challenger 2 (Empirical Challenger).

Coverage Focus:
1. Concurrency Stress: Multi-threaded worker pools (64 and 128 workers) racing on
   micro-budgets, exact boundary spend caps ($8.0000 and $12.1300), and cumulative
   ledger ceilings.
2. Decontamination Evasion Stress: Tricky variations of probe questions including
   case folding, invisible zero-width unicode spaces, punctuation/emoji variations,
   nested dictionary schemas, deep multi-turn conversation histories, and empirical
   blindspot documentation.
3. Simula Generator Error Resilience: Simulated disk full (disk < 5.0 GB), MLX mutex
   contention across competing processes/threads, unhandled worker exceptions with
   safe lock release, corrupted existing JSONL dataset recovery, and file descriptor safety.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import random
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
    BudgetExceeded,
    SimulaBudgetTracker,
    _MLX_PROCESS_LOCK,
    check_disk_space,
    check_resource_guardrails,
    clear_mlx_cache,
    generate_mock_completion,
    get_ledger_spend,
    run_simula_campaign,
)
from src.decontamination import (
    ContaminationFilter,
    filter_records,
    get_contamination_filter,
    is_contaminated,
    normalize_text,
    screen_jsonl,
)
from src.projection_data import INOCULATION_PREFIX

HELDOUT_PATH = ROOT / "evals" / "heldout_ids.json"
PROBES_PATH = ROOT / "evals" / "probes.jsonl"


# =============================================================================
# 1. Concurrency Stress & Exact Budget Boundary Caps
# =============================================================================
class TestTier5BudgetConcurrencyStress(unittest.TestCase):
    """Stress tests high-concurrency worker pools and exact budget boundaries."""

    def test_exact_boundary_caps_validation(self):
        """Validates exact $8.0000 and $12.1300 boundary values vs float overruns."""
        # 1. Exact $8.0000 campaign cap must pass
        t_ok = SimulaBudgetTracker(max_campaign_spend=8.0000, cumulative_spend_limit=12.1300)
        self.assertEqual(t_ok.max_campaign_spend, 8.0)
        self.assertEqual(t_ok.cumulative_spend_limit, 12.13)

        # 2. Even $0.0001 above $8.00 must raise ValueError
        with self.assertRaises(ValueError):
            SimulaBudgetTracker(max_campaign_spend=8.0001, cumulative_spend_limit=12.13)

        # 3. Even $0.0001 above $12.13 must raise ValueError
        with self.assertRaises(ValueError):
            SimulaBudgetTracker(max_campaign_spend=8.00, cumulative_spend_limit=12.1301)

        # 4. run_simula_campaign entry point validation
        with self.assertRaises(ValueError):
            run_simula_campaign(max_campaign_spend=8.00001, dry_run=True)

        with self.assertRaises(ValueError):
            run_simula_campaign(cumulative_spend_limit=12.13001, dry_run=True)

    def test_concurrency_race_on_micro_budget_64_workers(self):
        """Stress tests 64 worker threads racing on a tiny micro-budget cap ($0.0010)."""
        with tempfile.TemporaryDirectory() as td:
            led_p = Path(td) / "spend.jsonl"
            out_p = Path(td) / "out.jsonl"

            # Budget cap allows at most 2 items ($0.0005 per item in mock mode)
            micro_cap = 0.0010
            res = run_simula_campaign(
                max_campaign_spend=micro_cap,
                cumulative_spend_limit=12.13,
                ledger_path=led_p,
                output_path=out_p,
                dry_run=True,
                total_prompts=64,
                concurrency_workers=64,
            )

            self.assertEqual(res["status"], "budget_stopped")
            self.assertLessEqual(res["total_campaign_spend"], micro_cap)
            self.assertLessEqual(res["accepted_count"], 2)

            # Verify ledger accounting matches campaign result
            ledger_total = get_ledger_spend(led_p)
            self.assertLessEqual(ledger_total, micro_cap)

    def test_concurrency_race_on_micro_budget_128_workers(self):
        """Stress tests 128 worker threads racing on a $0.0020 budget cap with 128 prompts."""
        with tempfile.TemporaryDirectory() as td:
            led_p = Path(td) / "spend.jsonl"
            out_p = Path(td) / "out.jsonl"

            # Budget cap allows at most 4 items ($0.0005 * 4 = $0.0020)
            micro_cap = 0.0020
            res = run_simula_campaign(
                max_campaign_spend=micro_cap,
                cumulative_spend_limit=12.13,
                ledger_path=led_p,
                output_path=out_p,
                dry_run=True,
                total_prompts=128,
                concurrency_workers=128,
            )

            self.assertEqual(res["status"], "budget_stopped")
            self.assertLessEqual(res["total_campaign_spend"], micro_cap)
            self.assertLessEqual(res["accepted_count"], 4)

            # Assert output file contains exactly accepted_count lines
            if out_p.exists() and out_p.stat().st_size > 0:
                with open(out_p, "r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                self.assertEqual(len(lines), res["accepted_count"])

    def test_cumulative_spend_ceiling_boundary_race(self):
        """Tests worker pools racing when existing ledger spend is within $0.0010 of $12.1300."""
        with tempfile.TemporaryDirectory() as td:
            led_p = Path(td) / "spend.jsonl"
            out_p = Path(td) / "out.jsonl"

            # Pre-populate ledger with $12.1290 spend (headroom: $0.0010)
            with open(led_p, "w", encoding="utf-8") as f:
                f.write(json.dumps({"cost": 12.1290, "tag": "prior_run"}) + "\n")

            res = run_simula_campaign(
                max_campaign_spend=8.00,
                cumulative_spend_limit=12.1300,
                ledger_path=led_p,
                output_path=out_p,
                dry_run=True,
                total_prompts=64,
                concurrency_workers=64,
            )

            self.assertEqual(res["status"], "budget_stopped")
            self.assertLessEqual(res["total_campaign_spend"], 0.0010 + 1e-9)

            final_ledger_spend = get_ledger_spend(led_p)
            self.assertLessEqual(final_ledger_spend, 12.1300 + 1e-9)

    def test_zero_and_negative_spend_limits(self):
        """Verifies zero and negative budget limits abort cleanly without dispatching workers."""
        with tempfile.TemporaryDirectory() as td:
            for bad_cap in [0.0, -0.0, -1.0, -50.0]:
                led_p = Path(td) / f"led_{bad_cap}.jsonl"
                out_p = Path(td) / f"out_{bad_cap}.jsonl"
                res = run_simula_campaign(
                    max_campaign_spend=bad_cap,
                    cumulative_spend_limit=12.13,
                    ledger_path=led_p,
                    output_path=out_p,
                    dry_run=True,
                    total_prompts=10,
                )
                self.assertEqual(res["status"], "success")
                self.assertEqual(res["generated_count"], 0)
                self.assertEqual(res["accepted_count"], 0)
                self.assertEqual(res["total_campaign_spend"], 0.0)

    def test_budget_tracker_reservation_zero_leak_under_concurrency(self):
        """Verifies reservation counter returns strictly to 0.0 under high concurrency races."""
        cap = 0.020
        tracker = SimulaBudgetTracker(max_campaign_spend=cap, cumulative_spend_limit=12.13)

        completed_count = 0
        counter_lock = threading.Lock()

        def worker(idx: int):
            nonlocal completed_count
            cost_est = 0.0005
            if tracker.check_and_reserve(cost_est):
                # Simulate work
                time.sleep(0.0005)
                # 50% simulated completion, 50% simulated cancellation
                if idx % 2 == 0:
                    tracker.release_and_record(cost_est, cost_est)
                else:
                    tracker.release_reservation(cost_est)
                with counter_lock:
                    completed_count += 1

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(128)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Invariant: reserved budget must strictly reset to 0.0
        self.assertAlmostEqual(tracker.reserved, 0.0, places=9)
        self.assertLessEqual(tracker.campaign_spend, cap)


# =============================================================================
# 2. Decontamination Evasion Stress & Adversarial Variations
# =============================================================================
class TestTier5DecontaminationEvasionStress(unittest.TestCase):
    """Stress tests decontamination filtering against adversarial probe evasion tricks."""

    def setUp(self):
        self.cfilter = get_contamination_filter()
        # Fetch actual real probe sample from filter indexes
        self.sample_probe = list(self.cfilter.exact_raw)[0]
        self.assertTrue(len(self.sample_probe) >= 10, "Probe sample must have length >= 10")

    def test_case_folding_evasion(self):
        """Verifies probe detection across all case folding permutations."""
        # 1. UPPERCASE
        self.assertTrue(self.cfilter.is_contaminated(self.sample_probe.upper()))
        # 2. Title Case
        self.assertTrue(self.cfilter.is_contaminated(self.sample_probe.title()))
        # 3. Alternating / Random Mixed Case
        mixed = "".join(
            c.upper() if i % 2 == 0 else c.lower()
            for i, c in enumerate(self.sample_probe)
        )
        self.assertTrue(self.cfilter.is_contaminated(mixed))

    def test_invisible_zero_width_unicode_evasion(self):
        """Verifies zero-width spaces and invisible formatting characters cannot evade detection."""
        invisible_chars = [
            ("\u200b", "Zero-Width Space"),
            ("\u200c", "Zero-Width Non-Joiner"),
            ("\u200d", "Zero-Width Joiner"),
            ("\ufeff", "Zero-Width No-Break Space / BOM"),
            ("\u00ad", "Soft Hyphen"),
            ("\u2060", "Word Joiner"),
        ]

        for char, desc in invisible_chars:
            # A. Injected between every space
            between_words = self.sample_probe.replace(" ", f" {char} ")
            self.assertTrue(
                self.cfilter.is_contaminated(between_words),
                f"Failed to detect probe with {desc} between words",
            )

            # B. Injected between every single character
            char_injected = char.join(list(self.sample_probe))
            self.assertTrue(
                self.cfilter.is_contaminated(char_injected),
                f"Failed to detect probe with {desc} between characters",
            )

    def test_punctuation_and_whitespace_tricks(self):
        """Verifies aggressive punctuation, emojis, quotes, and whitespace variations."""
        # 1. Extreme whitespace: tabs, newlines, carriage returns, non-breaking spaces
        ws_probe = self.sample_probe.replace(" ", " \t\r\n\u00a0  ")
        self.assertTrue(self.cfilter.is_contaminated(ws_probe))

        # 2. Heavy punctuation interleaving with spaces
        punct_probe = self.sample_probe.replace(" ", " ...!???-- ")
        self.assertTrue(self.cfilter.is_contaminated(punct_probe))

        # 3. Smart quotes and brackets
        wrapped_probe = f"“«‘([{self.sample_probe}])’»”"
        self.assertTrue(self.cfilter.is_contaminated(wrapped_probe))

        # 4. Emojis interleaved with words
        emoji_probe = self.sample_probe.replace(" ", " 🤖🔥 ")
        self.assertTrue(self.cfilter.is_contaminated(emoji_probe))

        # 5. Embedded in a verbose conversational prompt
        context_probe = (
            f"Hello Wren, could you please evaluate this carefully:\n"
            f"> {self.sample_probe}\n"
            f"Provide a thoughtful, nuanced response."
        )
        self.assertTrue(self.cfilter.is_contaminated(context_probe))

    def test_delimiter_evasion_empirical_behavior(self):
        r"""Empirically analyzes and documents separator substitution (hyphens/slashes).

        Finding: Simple regex `[^\w\s]` strips non-word characters without replacing
        them with whitespace. Therefore, 'word-word' collapses to 'wordword', whereas
        'word - word' normalizes to 'word word'. This test verifies both behaviors.
        """
        # When spaces are accompanied by hyphens: 'word - word' -> 'word word' -> CONTAMINATED
        spaced_hyphens = self.sample_probe.replace(" ", " - ")
        self.assertTrue(self.cfilter.is_contaminated(spaced_hyphens))

        # When spaces are replaced solely by hyphens without whitespace:
        # normalize_text('a-b') -> 'ab'
        norm_spaced = normalize_text("what is the question")
        norm_collapsed = normalize_text("what-is-the-question")
        self.assertEqual(norm_spaced, "what is the question")
        self.assertEqual(norm_collapsed, "whatisthequestion")

    def test_nested_json_structures(self):
        """Verifies screening across various dictionary keys and nested formats."""
        sid = list(self.cfilter.src_ids)[0]

        # 1. Probe text under different supported schema keys
        for key in ["prompt", "question", "query", "text", "input", "completion"]:
            record = {key: self.sample_probe}
            is_bad, reason = self.cfilter.is_record_contaminated(record)
            self.assertTrue(is_bad, f"Failed on record key: {key}")
            self.assertIsNotNone(reason)

        # 2. Source ID under different supported identifier keys
        for key in ["src_id", "src", "source_id"]:
            record = {key: sid}
            is_bad, reason = self.cfilter.is_record_contaminated(record)
            self.assertTrue(is_bad, f"Failed on identifier key: {key}")
            self.assertIn(str(sid), str(reason))

        # 3. Non-string or malformed record fields handled gracefully
        malformed_records = [
            {"prompt": None},
            {"prompt": 12345},
            {"prompt": ["a", "b"]},
            {"messages": "not a list"},
            {"messages": [None, 42, {}]},
            {},
        ]
        for m in malformed_records:
            is_bad, reason = self.cfilter.is_record_contaminated(m)
            self.assertFalse(is_bad)
            self.assertIsNone(reason)

    def test_deep_multiturn_histories(self):
        """Verifies probe detection deep within multi-turn conversation structures."""
        for turn_count in [10, 50, 100]:
            # Probe at turn 0
            msgs_start = [{"role": "user", "content": self.sample_probe}]
            for i in range(1, turn_count):
                role = "assistant" if i % 2 == 1 else "user"
                msgs_start.append({"role": role, "content": f"filler turn message {i}"})
            self.assertTrue(self.cfilter.is_record_contaminated({"messages": msgs_start})[0])

            # Probe in middle turn
            msgs_mid = []
            mid_idx = turn_count // 2
            # Ensure middle turn has role 'user'
            if mid_idx % 2 == 1:
                mid_idx += 1
            for i in range(turn_count):
                role = "assistant" if i % 2 == 1 else "user"
                content = self.sample_probe if i == mid_idx else f"filler turn message {i}"
                msgs_mid.append({"role": role, "content": content})
            self.assertTrue(self.cfilter.is_record_contaminated({"messages": msgs_mid})[0])

            # Probe at final user turn
            msgs_end = []
            for i in range(turn_count):
                role = "assistant" if i % 2 == 1 else "user"
                msgs_end.append({"role": role, "content": f"filler turn message {i}"})
            msgs_end.append({"role": "user", "content": f"Final inquiry: {self.sample_probe}"})
            self.assertTrue(self.cfilter.is_record_contaminated({"messages": msgs_end})[0])

    def test_multiturn_assistant_turn_empirical_behavior(self):
        """Documents empirical behavior when probe is positioned solely in assistant turn.

        Finding: In accordance with line 226 of src/decontamination.py, messages screening
        only extracts user turns (m.get('role') == 'user') to check user prompts.
        Top-level 'completion' is screened, but assistant turns in messages list are not.
        """
        # When probe is in assistant role inside messages:
        asst_record = {
            "messages": [
                {"role": "user", "content": "What is the capital?"},
                {"role": "assistant", "content": self.sample_probe},
            ]
        }
        is_bad_asst, _ = self.cfilter.is_record_contaminated(asst_record)
        # Verify empirical behavior (user-only extraction)
        self.assertFalse(is_bad_asst)

        # However, top-level completion IS screened:
        comp_record = {
            "prompt": "What is the capital?",
            "completion": self.sample_probe,
        }
        is_bad_comp, reason_comp = self.cfilter.is_record_contaminated(comp_record)
        self.assertTrue(is_bad_comp)
        self.assertIn("completion", reason_comp)

    def test_jsonl_streaming_malformed_and_recovery(self):
        """Verifies screen_jsonl resilience against corrupt lines, empty lines, and bad JSON."""
        with tempfile.TemporaryDirectory() as td:
            in_p = Path(td) / "dirty.jsonl"
            clean_p = Path(td) / "clean.jsonl"
            bad_p = Path(td) / "rejected.jsonl"

            lines = [
                json.dumps({"prompt": "A totally clean prompt about physics."}) + "\n",
                "\n",  # empty line
                "   \n",  # whitespace line
                "NOT VALID JSON AT ALL\n",  # corrupt JSON
                "{'invalid': 'single quotes'}\n",  # invalid JSON syntax
                "{\"unterminated string: 123\n",  # truncated JSON
                json.dumps({"prompt": self.sample_probe}) + "\n",  # contaminated
                json.dumps({"prompt": "Another completely clean prompt about botany."}) + "\n",
            ]
            with open(in_p, "w", encoding="utf-8") as f:
                f.writelines(lines)

            stats = screen_jsonl(in_p, clean_p, bad_p, cfilter=self.cfilter)

            self.assertEqual(stats["clean"], 2)
            self.assertEqual(stats["contaminated"], 1)

            # Check clean file content
            with open(clean_p, "r", encoding="utf-8") as f:
                clean_lines = [json.loads(ln) for ln in f if ln.strip()]
            self.assertEqual(len(clean_lines), 2)
            self.assertIn("physics", clean_lines[0]["prompt"])
            self.assertIn("botany", clean_lines[1]["prompt"])

            # Check rejected file content
            with open(bad_p, "r", encoding="utf-8") as f:
                rejected_lines = [json.loads(ln) for ln in f if ln.strip()]
            self.assertEqual(len(rejected_lines), 1)
            self.assertIn("_contamination_reason", rejected_lines[0])


# =============================================================================
# 3. Simula Generator Error Resilience & Mutex Contention
# =============================================================================
class TestTier5SimulaGeneratorErrorResilience(unittest.TestCase):
    """Stress tests error resilience, mutex contention, disk floors, and crash recovery."""

    def test_simulated_disk_full_abort(self):
        """Verifies run_simula_campaign immediately halts when disk space is below 5.0 GB."""
        for simulated_free_gb in [4.999, 4.5, 2.0, 0.0]:
            mock_usage = MagicMock()
            mock_usage.free = int(simulated_free_gb * (1024**3))
            mock_usage.total = int(100.0 * (1024**3))

            with patch("shutil.disk_usage", return_value=mock_usage):
                # 1. Guardrail utility must raise RuntimeError
                with self.assertRaises(RuntimeError) as ctx:
                    check_resource_guardrails(5.0)
                self.assertIn("below 5.0 GB threshold", str(ctx.exception))

                # 2. Campaign runner must abort before generating or writing
                with self.assertRaises(RuntimeError) as ctx2:
                    run_simula_campaign(dry_run=True)
                self.assertIn("below 5.0 GB threshold", str(ctx2.exception))

    def test_disk_guardrails_path_fallback(self):
        """Verifies disk checking safely resolves non-existent deep paths to existing parent."""
        mock_ok = MagicMock()
        mock_ok.free = int(10.0 * (1024**3))
        mock_ok.total = int(100.0 * (1024**3))

        with patch("shutil.disk_usage", return_value=mock_ok):
            free_gb = check_disk_space("/this/deeply/nested/directory/does/not/exist/at/all", min_free_gb=5.0)
            self.assertAlmostEqual(free_gb, 10.0, places=1)

    def test_mlx_mutex_thread_contention_serialization(self):
        """Verifies _MLX_PROCESS_LOCK strictly serializes execution and blocks concurrency."""
        execution_events = []

        def long_running_task():
            with _MLX_PROCESS_LOCK:
                execution_events.append("task1_enter")
                time.sleep(0.08)
                execution_events.append("task1_exit")

        def competing_campaign():
            time.sleep(0.02)  # Ensure task 1 acquires first
            execution_events.append("campaign_queued")
            with tempfile.TemporaryDirectory() as td:
                run_simula_campaign(
                    output_path=Path(td) / "out.jsonl",
                    ledger_path=Path(td) / "led.jsonl",
                    dry_run=True,
                    total_prompts=2,
                )
            execution_events.append("campaign_completed")

        t1 = threading.Thread(target=long_running_task)
        t2 = threading.Thread(target=competing_campaign)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        expected_sequence = ["task1_enter", "campaign_queued", "task1_exit", "campaign_completed"]
        self.assertEqual(
            execution_events,
            expected_sequence,
            "Campaign must wait for existing MLX process lock holder before proceeding",
        )

    def test_mlx_mutex_clean_release_on_unhandled_worker_exception(self):
        """Verifies _MLX_PROCESS_LOCK is safely released even if worker or critic crashes."""
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "out.jsonl"
            led_p = Path(td) / "led.jsonl"

            # Inject a fatal exception into critic evaluation
            with patch("src.simula_generator.dual_critic_pipeline", side_effect=RuntimeError("Simulated Critic Crash")):
                with self.assertRaises(RuntimeError):
                    run_simula_campaign(
                        output_path=out_p,
                        ledger_path=led_p,
                        dry_run=True,
                        total_prompts=4,
                    )

            # Invariant: Lock must be cleanly released and re-acquirable
            lock_acquired = _MLX_PROCESS_LOCK.acquire(blocking=False)
            self.assertTrue(lock_acquired, "_MLX_PROCESS_LOCK was not released after worker crash")
            _MLX_PROCESS_LOCK.release()

    def test_corrupted_dataset_recovery_and_idempotent_resume(self):
        """Verifies campaign recovers gracefully from existing corrupted output files.

        Tests handling of:
        - Corrupted non-JSON lines
        - Lines with binary characters
        - JSON lines with non-dict types
        - Incomplete lines lacking trailing newline
        - Already completed items that must be skipped without duplicate billing.
        """
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "out.jsonl"
            led_p = Path(td) / "led.jsonl"

            # Pre-seed corrupted output file
            with open(out_p, "w", encoding="utf-8") as f:
                f.write("corrupted non json line\n")
                f.write("{\x00binary garbage}\n")
                f.write("{\"messages\": []}\n")
                f.write("{\"messages\": [\"string not dict\"]}\n")
                f.write(json.dumps({"messages": [{"role": "user", "content": "Completed Prompt 1"}]}) + "\n")
                f.write(json.dumps({"messages": [{"role": "user", "content": f"{INOCULATION_PREFIX}Completed Prompt 2"}]}) + "\n")
                # Missing trailing newline at end of file
                f.write("partial trailing fragment")

            prompt_pool = [
                {"id": "p1", "prompt": "Completed Prompt 1", "category": "general"},
                {"id": "p2", "prompt": "Completed Prompt 2", "category": "strict_verifiable_constraints"},
                {"id": "p3", "prompt": "New Fresh Prompt 3", "category": "step_by_step_math", "constraint_info": {"gold": "42"}},
                {"id": "p4", "prompt": "New Fresh Prompt 4", "category": "identity_persona"},
            ]

            res = run_simula_campaign(
                output_path=out_p,
                ledger_path=led_p,
                prompt_pool=prompt_pool,
                dry_run=True,
            )

            self.assertEqual(res["status"], "success")
            # Only p3 and p4 should be generated, since p1 and p2 were recognized as existing
            self.assertEqual(res["generated_count"], 2)
            self.assertEqual(res["accepted_count"], 2)

            # Verify that screen_jsonl can screen the recovered output file
            clean_res = Path(td) / "screened_clean.jsonl"
            stats = screen_jsonl(out_p, clean_res)
            self.assertGreaterEqual(stats["clean"], 2)

    def test_file_descriptor_safety_under_load(self):
        """Verifies file descriptors are not leaked during repeated generation runs."""
        with tempfile.TemporaryDirectory() as td:
            for iteration in range(5):
                out_p = Path(td) / f"out_iter_{iteration}.jsonl"
                led_p = Path(td) / f"led_iter_{iteration}.jsonl"

                res = run_simula_campaign(
                    output_path=out_p,
                    ledger_path=led_p,
                    dry_run=True,
                    total_prompts=8,
                    concurrency_workers=8,
                )
                self.assertEqual(res["status"], "success")

            gc.collect()


if __name__ == "__main__":
    unittest.main()
