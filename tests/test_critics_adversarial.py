"""Adversarial stress harness for Milestone 1 Dual-Critic Architecture (`src/critics.py`).

EMPIRICAL CHALLENGER TEST SUITE:
1. Extreme and corrupted inputs (empty text, whitespace only, 100k chars, Unicode emojis, regex chars).
2. Boundary value testing on Critic 2 (6.99 vs 7.00 vs 7.01, clamping, NaN/inf).
3. Strict verification of serial short-circuiting (Critic 2 NEVER called when Critic 1 fails).
4. Zero API spend and ledger integrity (0 network calls, 0 ledger modifications, teacher.chat never called).
5. Vulnerabilities and edge cases (gold=0 falsy bug, candidate['gold'] bypass, TypeError on gt=None, ReDoS).
"""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.critics import (
    evaluate_formal_critic,
    evaluate_constitutional_critic,
    dual_critic_pipeline,
    CONSTITUTIONAL_THRESHOLD,
)
from src.rlvr_verify import verify as verify_constraint

LEDGER_PATH = ROOT / "ledger" / "spend.jsonl"


class TestExtremeAndCorruptedInputs(unittest.TestCase):
    """Stress tests on extreme string sizes, corrupted inputs, Unicode, and regex characters."""

    def test_empty_and_whitespace_completions(self):
        corrupted_completions = [
            "",
            " ",
            "   ",
            "\t",
            "\t\t\t",
            "\n",
            "\r\n\r\n",
            "   \n\t\r\f\v   ",
            None,
        ]
        for comp in corrupted_completions:
            with self.subTest(comp=repr(comp)):
                ok, reason = evaluate_formal_critic("What is 2+2?", comp, None)
                self.assertFalse(ok, f"Expected False for {repr(comp)}, got True")
                self.assertIn("empty", reason.lower())

                score, rationale = evaluate_constitutional_critic("What is 2+2?", comp, offline=True)
                self.assertEqual(score, 1.0)
                self.assertIn("empty", rationale.lower())

    def test_100k_character_completion_performance(self):
        # 100k character text repeated
        chunk = "Wren is an honest, direct, and thoughtful AI assistant. "
        text_100k = chunk * (100_000 // len(chunk) + 1)
        self.assertGreater(len(text_100k), 100_000)

        # 1. Critic 1 evaluation without constraints
        t0 = time.time()
        ok, reason = evaluate_formal_critic("Prompt", text_100k, None)
        t_c1 = time.time() - t0
        self.assertTrue(ok)
        self.assertLess(t_c1, 0.5, f"Critic 1 took too long: {t_c1:.4f}s")

        # 2. Critic 2 offline heuristic evaluation
        t0 = time.time()
        score, rat = evaluate_constitutional_critic("Prompt", text_100k, offline=True)
        t_c2 = time.time() - t0
        self.assertGreaterEqual(score, CONSTITUTIONAL_THRESHOLD)
        self.assertLess(t_c2, 1.0, f"Critic 2 took too long on 100k chars: {t_c2:.4f}s")

        # 3. Dual critic pipeline
        cand = {"prompt": "Tell me a long story", "completion": text_100k, "constraint_info": None}
        t0 = time.time()
        acc, meta = dual_critic_pipeline(cand, offline=True)
        t_pipe = time.time() - t0
        self.assertTrue(acc)
        self.assertLess(t_pipe, 1.5, f"Dual critic took too long: {t_pipe:.4f}s")

    def test_non_ascii_unicode_and_emojis(self):
        # Test diverse multilingual, RTL, and complex emoji characters
        complex_samples = [
            "🤖🚀🔥✨🌟💡🧠🎉",  # Common emojis
            "👩‍👩‍👦‍👦 👨‍💻 🏳️‍🌈 🧗‍♀️",  # ZWJ sequences & variation selectors
            "こんにちは世界！これはテストです。",  # Japanese (CJK)
            "مرحبا بالعالم! هذا اختبار لقدرات النموذج.",  # Arabic (RTL)
            "שלום עולם! בדיקת טקסט בעברית.",  # Hebrew (RTL)
            "Привет мир! Это проверка кириллицы.",  # Russian (Cyrillic)
            "Café, résumé, naïve, façade, über, señor",  # Accented Latin
            "\u200b\u200c\u200d\ufeffHidden zero-width characters\u200b",  # Zero-width
        ]

        for text in complex_samples:
            with self.subTest(sample=text[:20]):
                # Formal critic with no constraint
                ok, _ = evaluate_formal_critic("Prompt", text, None)
                self.assertTrue(ok)

                # Formal critic with no-comma constraint
                has_comma = "," in text
                ci = {"func_name": "validate_no_commas"}
                ok, _ = evaluate_formal_critic("Prompt", text, ci)
                self.assertEqual(ok, not has_comma)

                # Constitutional critic offline heuristic
                score, _ = evaluate_constitutional_critic("Prompt", text, offline=True)
                self.assertGreaterEqual(score, CONSTITUTIONAL_THRESHOLD)

    def test_special_regex_characters_in_inputs(self):
        # Meta-characters that could break unescaped regexes: .*+?^${}()|[]\
        regex_explosion = r"^$.*+?()[]{}|\\^$.*+?()[]{}|\\" * 50
        prompt_with_regex = r"Find pattern: ^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        completion_with_regex = f"Here is the literal regex string: {regex_explosion}"

        cand = {
            "prompt": prompt_with_regex,
            "completion": completion_with_regex,
            "constraint_info": None,
        }
        acc, meta = dual_critic_pipeline(cand, offline=True)
        self.assertTrue(acc)
        self.assertTrue(meta["formal_pass"])

        # Test in keyword verification constraint
        ci = {"func_name": "verify_keywords", "keyword_list": [r"^$.*+?", r"[]{}|\\"]}
        ok, _ = evaluate_formal_critic(prompt_with_regex, completion_with_regex, ci)
        self.assertTrue(ok)

    def test_malformed_and_type_corrupted_candidates(self):
        # Non-dict candidates
        for bad_cand in [None, 12345, [1, 2, 3], "string candidate", ("tuple",)]:
            with self.subTest(cand=bad_cand):
                acc, meta = dual_critic_pipeline(bad_cand)
                self.assertFalse(acc)
                self.assertFalse(meta["formal_pass"])
                self.assertIn("error", meta)

        # Missing completion field
        bad_dict = {"prompt": "What is this?"}
        acc, meta = dual_critic_pipeline(bad_dict)
        self.assertFalse(acc)
        self.assertFalse(meta["formal_pass"])
        self.assertIn("error", meta)

        # Non-string completions (converted gracefully via str())
        cand_num = {"prompt": "Number?", "completion": 42}
        acc, meta = dual_critic_pipeline(cand_num, offline=True)
        self.assertTrue(acc)

        cand_list = {"prompt": "List?", "completion": ["apple", "banana"]}
        acc, meta = dual_critic_pipeline(cand_list, offline=True)
        self.assertTrue(acc)


class TestBoundaryValuesCritic2(unittest.TestCase):
    """Rigorous boundary value testing on Critic 2 acceptance thresholds and score clamping."""

    def test_boundary_exact_threshold_699_vs_700_vs_701(self):
        cand = {"prompt": "Explain gravity.", "completion": "Mass attracts mass via spacetime curvature."}

        # 6.99 MUST FAIL (< 7.00)
        acc_699, meta_699 = dual_critic_pipeline(cand, mock_score=6.99)
        self.assertFalse(acc_699)
        self.assertFalse(meta_699["critic2_passed"])
        self.assertEqual(meta_699["critic2_score"], 6.99)
        self.assertIn("Rejected by Critic 2", meta_699["rationale"])

        # 6.999999 MUST FAIL (< 7.00)
        acc_near, meta_near = dual_critic_pipeline(cand, mock_score=6.999999)
        self.assertFalse(acc_near)
        self.assertFalse(meta_near["critic2_passed"])

        # 7.00 MUST PASS (>= 7.00)
        acc_700, meta_700 = dual_critic_pipeline(cand, mock_score=7.00)
        self.assertTrue(acc_700)
        self.assertTrue(meta_700["critic2_passed"])
        self.assertEqual(meta_700["critic2_score"], 7.00)
        self.assertIn("Accepted by dual critics", meta_700["rationale"])

        # 7.000001 MUST PASS (>= 7.00)
        acc_above, meta_above = dual_critic_pipeline(cand, mock_score=7.000001)
        self.assertTrue(acc_above)
        self.assertTrue(meta_above["critic2_passed"])

        # 7.01 MUST PASS (>= 7.00)
        acc_701, meta_701 = dual_critic_pipeline(cand, mock_score=7.01)
        self.assertTrue(acc_701)
        self.assertTrue(meta_701["critic2_passed"])
        self.assertEqual(meta_701["critic2_score"], 7.01)

    def test_score_clamping_extremes(self):
        cand = {"prompt": "Prompt", "completion": "Valid text"}

        # Clamping at lower bound (1.0)
        for under in [-100.0, -1.0, 0.0, 0.5, 0.99]:
            score, _ = evaluate_constitutional_critic("P", "C", mock_score=under)
            self.assertEqual(score, 1.0, f"Expected clamp to 1.0 for {under}, got {score}")

        # Clamping at upper bound (10.0)
        for over in [10.01, 11.0, 50.0, 1000.0]:
            score, _ = evaluate_constitutional_critic("P", "C", mock_score=over)
            self.assertEqual(score, 10.0, f"Expected clamp to 10.0 for {over}, got {score}")

    def test_heuristic_scoring_deductions(self):
        # Baseline clean answer: 8.5
        score_clean, _ = evaluate_constitutional_critic("Q", "Direct factual answer without fluff.", offline=True)
        self.assertEqual(score_clean, 8.5)

        # Single penalty: Sycophancy (-2.0 -> 6.5, falls below threshold 7.0)
        score_syco, _ = evaluate_constitutional_critic("Q", "Great question! The answer is 42.", offline=True)
        self.assertEqual(score_syco, 6.5)
        self.assertLess(score_syco, CONSTITUTIONAL_THRESHOLD)

        # Single penalty: Generic closer (-2.0 -> 6.5, falls below threshold 7.0)
        score_close, _ = evaluate_constitutional_critic("Q", "The answer is 42. I hope this helps!", offline=True)
        self.assertEqual(score_close, 6.5)
        self.assertLess(score_close, CONSTITUTIONAL_THRESHOLD)

        # Single penalty: Preachy (-2.5 -> 6.0, falls below threshold 7.0)
        score_preach, _ = evaluate_constitutional_critic("Q", "As an AI language model, water is wet.", offline=True)
        self.assertEqual(score_preach, 6.0)
        self.assertLess(score_preach, CONSTITUTIONAL_THRESHOLD)

        # Single penalty: Identity violation (-3.0 -> 5.5, falls below threshold 7.0)
        score_id, _ = evaluate_constitutional_critic("Q", "I am Claude, created by Anthropic.", offline=True)
        self.assertEqual(score_id, 5.5)
        self.assertLess(score_id, CONSTITUTIONAL_THRESHOLD)

        # Multiple penalties stacked: clamp to 1.0
        score_all, _ = evaluate_constitutional_critic(
            "Q",
            "Great question! As an AI language model, I am Claude created by Anthropic. I hope this helps!",
            offline=True,
        )
        self.assertEqual(score_all, 1.0)


class TestSerialShortCircuiting(unittest.TestCase):
    """Strict verification that Critic 2 is NEVER invoked when Critic 1 fails."""

    @patch("src.critics.evaluate_constitutional_critic")
    def test_critic2_never_called_on_all_formal_failures(self, mock_critic2):
        failure_cases = [
            # 1. Empty completion
            {"prompt": "Test", "completion": "", "constraint_info": None},
            # 2. Whitespace completion
            {"prompt": "Test", "completion": "   \n\t  ", "constraint_info": None},
            # 3. None completion
            {"prompt": "Test", "completion": None, "constraint_info": None},
            # 4. RLVR Lowercase failure
            {"prompt": "Test", "completion": "UPPERCASE", "constraint_info": {"func_name": "validate_lowercase"}},
            # 5. RLVR Commas failure
            {"prompt": "Test", "completion": "A, B", "constraint_info": {"func_name": "validate_no_commas"}},
            # 6. RLVR JSON failure
            {"prompt": "Test", "completion": "not json", "constraint_info": {"func_name": "validate_json_format"}},
            # 7. RLVR Bullet points failure
            {"prompt": "Test", "completion": "* one", "constraint_info": {"func_name": "verify_bullet_points", "N": 3}},
            # 8. Math answer mismatch
            {"prompt": "Math", "completion": "The answer is 50", "constraint_info": {"gold": "42"}},
            # 9. Math missing digits
            {"prompt": "Math", "completion": "I don't know the answer", "constraint_info": {"gold": "42"}},
            # 10. Compound constraint failure (first passes, second fails)
            {
                "prompt": "Test",
                "completion": "lowercase but has, a comma",
                "constraint_info": [
                    {"func_name": "validate_lowercase"},
                    {"func_name": "validate_no_commas"},
                ],
            },
            # 11. Corrupted constraint JSON string
            {"prompt": "Test", "completion": "valid text", "constraint_info": "{bad_json: 123}"},
            # 12. Unknown constraint function name
            {"prompt": "Test", "completion": "valid text", "constraint_info": {"func_name": "non_existent_checker"}},
        ]

        for idx, cand in enumerate(failure_cases):
            mock_critic2.reset_mock()
            acc, meta = dual_critic_pipeline(cand)

            self.assertFalse(acc, f"Case {idx + 1} should have been rejected")
            self.assertFalse(meta["formal_pass"], f"Case {idx + 1} formal_pass should be False")
            self.assertFalse(meta["critic1_passed"], f"Case {idx + 1} critic1_passed should be False")
            self.assertIsNone(meta["constitutional_score"], f"Case {idx + 1} constitutional_score must be None")
            self.assertIsNone(meta["critic2_score"], f"Case {idx + 1} critic2_score must be None")
            self.assertIsNone(meta["constitutional_pass"], f"Case {idx + 1} constitutional_pass must be None")
            self.assertIsNone(meta["critic2_passed"], f"Case {idx + 1} critic2_passed must be None")
            self.assertIsNone(meta["critic2_details"], f"Case {idx + 1} critic2_details must be None")

            # CRITICAL ASSERTION: Critic 2 must NEVER be called
            mock_critic2.assert_not_called()

    @patch("src.critics.evaluate_constitutional_critic")
    def test_critic2_called_exactly_once_when_formal_passes(self, mock_critic2):
        mock_critic2.return_value = (8.0, "Passes constitutional check.")
        cand = {
            "prompt": "Test",
            "completion": "all lowercase without punctuation breaks",
            "constraint_info": {"func_name": "validate_lowercase"},
        }
        acc, meta = dual_critic_pipeline(cand)
        self.assertTrue(acc)
        self.assertTrue(meta["formal_pass"])
        self.assertEqual(meta["critic2_score"], 8.0)
        mock_critic2.assert_called_once()


class TestZeroApiSpendAndLedgerIntegrity(unittest.TestCase):
    """Empirical verification that critics make ZERO network calls and ZERO ledger modifications."""

    def test_zero_network_calls_during_offline_evaluations(self):
        # Patch socket connect and urllib to fail loudly if any network connection is attempted
        def forbidden_connect(*args, **kwargs):
            raise RuntimeError("CRITICAL ERROR: Unexpected network socket connection in offline critic!")

        with patch.object(socket.socket, "connect", side_effect=forbidden_connect):
            # Run 50 diverse evaluations
            for i in range(50):
                cand = {
                    "prompt": f"Query {i}",
                    "completion": f"Direct answer {i} without preamble.",
                    "constraint_info": None,
                }
                acc, meta = dual_critic_pipeline(cand, offline=True)
                self.assertTrue(acc)
                self.assertGreaterEqual(meta["critic2_score"], CONSTITUTIONAL_THRESHOLD)

    def test_zero_ledger_modifications_during_evaluations(self):
        # Verify ledger file is not modified
        initial_mtime = LEDGER_PATH.stat().st_mtime if LEDGER_PATH.exists() else 0
        initial_size = LEDGER_PATH.stat().st_size if LEDGER_PATH.exists() else 0
        initial_content = LEDGER_PATH.read_bytes() if LEDGER_PATH.exists() else b""

        # Perform 100 evaluations
        for i in range(100):
            cand = {
                "prompt": f"Math prompt {i}",
                "completion": f"The answer is {i}. Answer: {i}",
                "constraint_info": {"gold": str(i)},
            }
            dual_critic_pipeline(cand, offline=True)

        current_size = LEDGER_PATH.stat().st_size if LEDGER_PATH.exists() else 0
        current_content = LEDGER_PATH.read_bytes() if LEDGER_PATH.exists() else b""

        self.assertEqual(initial_size, current_size, "Ledger size changed during critic evaluations!")
        self.assertEqual(initial_content, current_content, "Ledger content modified during critic evaluations!")

    def test_teacher_chat_never_invoked_by_critics(self):
        # In offline mode, src.teacher must never be imported or called
        mock_teacher = MagicMock()
        with patch.dict(sys.modules, {"src.teacher": mock_teacher}):
            cand = {"prompt": "Hello", "completion": "Hi there.", "constraint_info": None}
            dual_critic_pipeline(cand, offline=True)
            evaluate_constitutional_critic("Hello", "Hi there.", offline=True)
            mock_teacher.chat.assert_not_called()


class TestVulnerabilitiesAndEdgeCases(unittest.TestCase):
    """Empirical demonstration of bugs and adversarial edge cases found in src/critics.py."""

    def test_vulnerability_gold_zero_falsy_fallthrough(self):
        """VERIFIED FIX 1:
        In `evaluate_formal_critic`, gold keys are resolved using explicit `is not None`
        checks. When the gold answer is integer 0 or float 0.0, it is correctly preserved
        and verified rather than evaluating to None.
        """
        # Gold answer is 0 as integer
        ci_zero_int = {"gold": 0}
        ok_int, reason_int = evaluate_formal_critic("What is 5 - 5?", "The answer is 0. Answer: 0", ci_zero_int)

        # Gold answer is 0.0 as float
        ci_zero_float = {"gold_answer": 0.0}
        ok_float, reason_float = evaluate_formal_critic("What is 5 - 5?", "The answer is 0. Answer: 0", ci_zero_float)

        self.assertTrue(ok_int, f"Integer 0 gold failed: {reason_int}")
        self.assertIn("matches gold", reason_int)
        self.assertTrue(ok_float, f"Float 0.0 gold failed: {reason_float}")
        self.assertIn("matches gold", reason_float)

    def test_vulnerability_direct_candidate_gold_bypasses_formal_critic(self):
        """VERIFIED FIX 2:
        In `dual_critic_pipeline`, if candidate provides `gold: 42` directly and constraint_info
        is not a dict, it is wrapped into `{"gold": 42}` so that math verification executes.
        Incorrect completions with top-level gold are properly rejected.
        """
        cand_wrong_answer = {
            "prompt": "What is 20 + 22?",
            "completion": "The answer is completely wrong: 999999",
            "gold": 42,  # Top-level gold key
        }
        acc, meta = dual_critic_pipeline(cand_wrong_answer, offline=True)
        self.assertFalse(meta["formal_pass"], "Top-level numeric gold must trigger formal verification")
        self.assertFalse(acc)
        self.assertIn("Math answer mismatch", meta["critic1_reason"])

        # And verify correct answer passes
        cand_correct = {
            "prompt": "What is 20 + 22?",
            "completion": "The answer is 42. Answer: 42",
            "gold": 42,
        }
        acc_ok, meta_ok = dual_critic_pipeline(cand_correct, offline=True)
        self.assertTrue(meta_ok["formal_pass"])
        self.assertTrue(acc_ok)

    def test_vulnerability_type_error_on_non_dict_ground_truth(self):
        """VERIFIED FIX 3:
        In `evaluate_formal_critic`, checking `gt` safely verifies `isinstance(gt, dict)`.
        Non-dict ground_truth values cleanly fail with (False, reason) instead of raising TypeError.
        """
        ci_bad_gt = {"ground_truth": None}
        ok, reason = evaluate_formal_critic("Prompt", "Completion", ci_bad_gt)
        self.assertFalse(ok)
        self.assertIn("Invalid ground_truth format", reason)

        ci_int_gt = {"ground_truth": 42}
        ok, reason = evaluate_formal_critic("Prompt", "Completion", ci_int_gt)
        self.assertFalse(ok)
        self.assertIn("Invalid ground_truth format", reason)

    def test_stress_fuzzer_1000_corrupted_inputs_no_crash(self):
        """Stress test with 1000 dynamically fuzzed and mutated inputs."""
        import random

        adversarial_tokens = [
            "\x00", "\x01", "\t", "\n", "\r", "\x1b", "\ufffd",
            "🚀", "👩‍👩‍👧‍👧", "💥", "中文", "العربية", "עברית",
            "\"", "'", "\\", "```", "{}", "[]", "<script>",
            ".*+?^${}()|[]\\",
            "A" * 1000, "1234567890", "-999.999", "NaN", "Infinity",
            "None", "null", "undefined", "true", "false",
        ]

        random.seed(42)
        crashes = 0
        for i in range(1000):
            # Generate a noisy mutated string
            num_tokens = random.randint(1, 15)
            comp_sample = " ".join(random.choice(adversarial_tokens) for _ in range(num_tokens))
            cand = {
                "prompt": f"Fuzz prompt #{i}",
                "completion": comp_sample,
                "constraint_info": None if (i % 2 == 0) else {"func_name": "validate_lowercase"},
            }
            try:
                acc, meta = dual_critic_pipeline(cand, offline=True)
                self.assertIsInstance(acc, bool)
                self.assertIsInstance(meta, dict)
            except Exception as e:
                crashes += 1

        self.assertEqual(crashes, 0, f"Encountered {crashes} crashes during 1000-sample fuzz stress testing!")

    @patch("src.critics.evaluate_constitutional_critic")
    def test_short_circuit_cost_saving_simulation(self, mock_c2):
        """Verify that in a batch of 100 items with 50 formal failures,
        Critic 2 is called exactly 50 times (100% cost saved on invalid items).
        """
        mock_c2.return_value = (8.5, "Passed constitutional review.")

        batch = []
        for i in range(100):
            if i % 2 == 0:
                # Passing formal candidate
                batch.append({
                    "prompt": f"Item {i}",
                    "completion": "all lowercase text item",
                    "constraint_info": {"func_name": "validate_lowercase"},
                })
            else:
                # Failing formal candidate (UPPERCASE)
                batch.append({
                    "prompt": f"Item {i}",
                    "completion": "UPPERCASE FAILS FORMAL CHECK",
                    "constraint_info": {"func_name": "validate_lowercase"},
                })

        accepted_count = 0
        for cand in batch:
            acc, _ = dual_critic_pipeline(cand, offline=True)
            if acc:
                accepted_count += 1

        self.assertEqual(accepted_count, 50)
        # Critic 2 was called exactly 50 times, saving 50 LLM calls!
        self.assertEqual(mock_c2.call_count, 50)

    def test_sentence_constraint_redos_scaling_measurement(self):
        """Verifies linear O(N) performance on large unpunctuated text (no ReDoS backtracking)."""
        # String with 100,000 characters and no sentence-ending punctuation
        t = "word " * 20000
        gt = {"func_name": "verify_sentence_constraint", "N": 1, "quantifier": "at least"}
        t0 = time.time()
        res = verify_constraint(t, gt)
        elapsed = time.time() - t0

        self.assertFalse(res)
        # Linear state machine must process 100k chars in < 0.01 seconds (typically < 0.003s)
        self.assertLess(elapsed, 0.01, f"Sentence counting took too long ({elapsed:.4f}s), possible ReDoS!")


if __name__ == "__main__":
    unittest.main()
