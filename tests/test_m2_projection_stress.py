"""Empirical adversarial stress test suite for Milestone 2: Projection-Lite & Inoculation.

Authored by Challenger 2 (M2) to empirically stress-test:
1. Inoculation idempotency (1x, 2x, 5x chaining across taxonomy categories).
2. Null and type robustness (None prompt/completion, numeric/dict/bool constraints).
3. Proposal ranking edge cases (empty list, single candidate, offline heuristic conciseness).
4. Hint stripping and leakage sanitization (scaffolding removal, leakage phrases, echoed headers).
5. Mathematical extraction and verification robustness.
"""

from __future__ import annotations

import unittest
from typing import Any

from src.projection_data import (
    INOCULATION_PREFIX,
    MATH_FMT,
    build_math_hint_prompt,
    extract_gsm_number,
    format_inoculated_ifeval,
    is_math_correct,
    projection_math_pipeline,
    rank_math_proposals_by_likelihood,
    sanitize_hint_completion,
    strip_math_hint,
)
from src.simula_taxonomy import generate_prompts_for_category


class TestInoculationIdempotencyStress(unittest.TestCase):
    """Stress tests verifying that repeated inoculation wrapping is strictly idempotent."""

    def test_inoculation_chaining_on_simula_taxonomy_prompts(self):
        """Verify calling format_inoculated_ifeval 1x, 2x, 5x preserves exactly 1 prefix."""
        for complexity in ("low", "medium", "high"):
            items = generate_prompts_for_category("strict_verifiable_constraints", complexity, n=3)
            for item in items:
                raw_prompt = item["prompt"]
                # The taxonomy prompt starts with 1 prefix
                self.assertEqual(
                    raw_prompt.count(INOCULATION_PREFIX),
                    1,
                    f"Simula taxonomy prompt must start with exactly 1 prefix: {raw_prompt[:60]}",
                )

                # 1x wrapping
                r1 = format_inoculated_ifeval(raw_prompt, "valid completion", item["constraint_info"])
                p1 = r1["messages"][0]["content"]
                self.assertEqual(
                    p1.count(INOCULATION_PREFIX),
                    1,
                    f"1x wrapping produced duplicate prefix: {p1[:80]}",
                )

                # 2x wrapping
                r2 = format_inoculated_ifeval(p1, "valid completion", item["constraint_info"])
                p2 = r2["messages"][0]["content"]
                self.assertEqual(
                    p2.count(INOCULATION_PREFIX),
                    1,
                    f"2x wrapping produced duplicate prefix: {p2[:80]}",
                )

                # 5x chaining
                p_curr = p2
                for step in range(3, 6):
                    r_curr = format_inoculated_ifeval(p_curr, "valid completion", item["constraint_info"])
                    p_curr = r_curr["messages"][0]["content"]
                    self.assertEqual(
                        p_curr.count(INOCULATION_PREFIX),
                        1,
                        f"{step}x wrapping produced duplicate prefix: {p_curr[:80]}",
                    )

    def test_inoculation_raw_uninoculated_prompt_repeated_wrapping(self):
        """Verify repeated wrapping of an initially raw prompt preserves exactly 1 prefix."""
        raw = "Write a haiku about the solar eclipse."
        curr = raw
        for step in range(1, 6):
            res = format_inoculated_ifeval(curr, "shadow falls softly\nmoon obscures the blazing sun\nday turns into night")
            curr = res["messages"][0]["content"]
            self.assertEqual(
                curr.count(INOCULATION_PREFIX),
                1,
                f"Step {step} had {curr.count(INOCULATION_PREFIX)} prefixes",
            )
            self.assertTrue(curr.startswith(INOCULATION_PREFIX))
            self.assertIn(raw, curr)

    def test_inoculation_whitespace_and_newline_variations(self):
        """Test idempotency when prefixes have leading spaces, trailing spaces, or multiple newlines."""
        variations = [
            f"   {INOCULATION_PREFIX}Prompt text",
            f"{INOCULATION_PREFIX.strip()} Prompt text",
            f"{INOCULATION_PREFIX.strip()}\n\nPrompt text",
            f"{INOCULATION_PREFIX}\nPrompt text",
            f"{INOCULATION_PREFIX}",
            f"{INOCULATION_PREFIX.strip()}",
        ]
        for var in variations:
            r1 = format_inoculated_ifeval(var, "comp")
            p1 = r1["messages"][0]["content"]
            self.assertEqual(
                p1.count(INOCULATION_PREFIX),
                1,
                f"Failed for variation: {repr(var)} -> {repr(p1)}",
            )
            # Re-wrap
            r2 = format_inoculated_ifeval(p1, "comp")
            p2 = r2["messages"][0]["content"]
            self.assertEqual(
                p2.count(INOCULATION_PREFIX),
                1,
                f"Failed re-wrapping for variation: {repr(var)} -> {repr(p2)}",
            )


class TestNullAndTypeRobustness(unittest.TestCase):
    """Stress tests verifying type safety and null handling across inputs."""

    def test_none_prompt_and_none_completion(self):
        """Verify None prompt and None completion do not raise exceptions and return standard records."""
        res = format_inoculated_ifeval(None, None)
        self.assertEqual(res["messages"][0]["content"], INOCULATION_PREFIX)
        self.assertEqual(res["completion"], "")
        self.assertEqual(res["constraint"], "")
        self.assertFalse(res["thinking"])
        self.assertEqual(res["kind"], "inoculated_ifeval")

    def test_none_prompt_with_valid_completion(self):
        res = format_inoculated_ifeval(None, "Valid response")
        self.assertEqual(res["messages"][0]["content"], INOCULATION_PREFIX)
        self.assertEqual(res["completion"], "Valid response")

    def test_valid_prompt_with_none_completion(self):
        res = format_inoculated_ifeval("Some prompt", None)
        self.assertEqual(res["messages"][0]["content"], f"{INOCULATION_PREFIX}Some prompt")
        self.assertEqual(res["completion"], "")

    def test_numeric_constraint_preservation(self):
        """Verify numeric constraints (0, 42, -5, 3.14) are not coerced to empty string."""
        # 0 is falsy in Python; must be preserved as 0
        r_zero = format_inoculated_ifeval("p", "c", 0)
        self.assertIs(type(r_zero["constraint"]), int)
        self.assertEqual(r_zero["constraint"], 0)

        # 42
        r_42 = format_inoculated_ifeval("p", "c", 42)
        self.assertEqual(r_42["constraint"], 42)

        # Negative int
        r_neg = format_inoculated_ifeval("p", "c", -10)
        self.assertEqual(r_neg["constraint"], -10)

        # Float
        r_float = format_inoculated_ifeval("p", "c", 3.14159)
        self.assertEqual(r_float["constraint"], 3.14159)

    def test_dict_and_complex_constraints(self):
        """Verify dict, list, and boolean constraints are preserved intact."""
        dict_c = {"rule": 1, "nested": {"min": 5, "max": 10}}
        r_dict = format_inoculated_ifeval("p", "c", dict_c)
        self.assertEqual(r_dict["constraint"], dict_c)

        list_c = ["lowercase", "no_punctuation"]
        r_list = format_inoculated_ifeval("p", "c", list_c)
        self.assertEqual(r_list["constraint"], list_c)

        r_bool_false = format_inoculated_ifeval("p", "c", False)
        self.assertIs(r_bool_false["constraint"], False)

        r_bool_true = format_inoculated_ifeval("p", "c", True)
        self.assertIs(r_bool_true["constraint"], True)

    def test_non_string_prompt_and_completion_coercion(self):
        """Verify non-string prompt and completion types coerce cleanly via str()."""
        r_num = format_inoculated_ifeval(12345, 67890)
        self.assertEqual(r_num["messages"][0]["content"], f"{INOCULATION_PREFIX}12345")
        self.assertEqual(r_num["completion"], "67890")


class TestProposalRankingEdgeCases(unittest.TestCase):
    """Stress tests for rank_math_proposals_by_likelihood."""

    def test_empty_candidates_raises_value_error(self):
        """Assert empty candidate sequences raise ValueError."""
        with self.assertRaises(ValueError):
            rank_math_proposals_by_likelihood(None, None, "Prompt", [])

        with self.assertRaises(ValueError):
            rank_math_proposals_by_likelihood(None, None, "Prompt", ())

    def test_single_candidate_returned_directly_without_scoring(self):
        """Assert single candidate returns immediately regardless of other parameters."""
        cand = "Only one step: 5 + 5 = 10. Answer: 10"
        # Offline mode
        res1 = rank_math_proposals_by_likelihood(None, None, "Prompt", [cand], offline=True)
        self.assertEqual(res1, cand)

        # Even with mismatched mock_scores, single candidate is returned directly
        res2 = rank_math_proposals_by_likelihood(None, None, "Prompt", [cand], mock_scores=[])
        self.assertEqual(res2, cand)

    def test_deterministic_offline_ranking_concise_defeats_rambling(self):
        """Assert concise, properly formatted candidates defeat rambling candidates."""
        concise_formatted = "4 * 5 = 20. Answer: 20"
        rambling_formatted = (
            "Let us consider four groups of five objects each. Counting them sequentially: "
            "five, ten, fifteen, and twenty. Thus four multiplied by five gives twenty. "
            "Therefore the final calculation yields Answer: 20"
        )
        candidates = [rambling_formatted, concise_formatted]

        winner = rank_math_proposals_by_likelihood(None, None, "What is 4 * 5?", candidates, offline=True)
        self.assertEqual(winner, concise_formatted)

    def test_offline_ranking_format_bonus_precedence(self):
        """Assert format bonuses ('Answer:', '=') outweigh small length differences."""
        # Unformatted short response
        short_unformatted = "20"
        # Slightly longer with Answer: and =
        proper_formatted = "4 * 5 = 20. Answer: 20"

        # short_unformatted score: -2
        # proper_formatted score: -23 + 50 + 10 = +37
        winner = rank_math_proposals_by_likelihood(
            None, None, "What is 4 * 5?", [short_unformatted, proper_formatted], offline=True
        )
        self.assertEqual(winner, proper_formatted)

    def test_offline_ranking_tie_breaking(self):
        """Assert deterministic tie-breaking favors earlier candidates on exact score ties."""
        cand_a = "10 + 10 = 20. Answer: 20"
        cand_b = "15 + 05 = 20. Answer: 20"
        self.assertEqual(len(cand_a), len(cand_b))

        winner1 = rank_math_proposals_by_likelihood(None, None, "Q", [cand_a, cand_b], offline=True)
        self.assertEqual(winner1, cand_a)

        winner2 = rank_math_proposals_by_likelihood(None, None, "Q", [cand_b, cand_a], offline=True)
        self.assertEqual(winner2, cand_b)

    def test_mock_scores_ranking_and_validation(self):
        """Assert explicit mock scores order correctly and reject length mismatches."""
        cands = ["Sol 1", "Sol 2", "Sol 3"]
        scores = [-5.0, -1.2, -8.9]
        winner = rank_math_proposals_by_likelihood(None, None, "Q", cands, mock_scores=scores)
        self.assertEqual(winner, "Sol 2")

        with self.assertRaises(ValueError):
            rank_math_proposals_by_likelihood(None, None, "Q", cands, mock_scores=[-1.0, -2.0])


class TestHintStrippingAndSanitization(unittest.TestCase):
    """Stress tests for hint stripping and leakage sanitization."""

    def test_strip_math_hint_removes_all_scaffolding(self):
        """Assert strip_math_hint removes target answer and key step annotations."""
        q = "Janet has 16 eggs. She uses 4. How many remain?"
        gold = "12"
        reasoning = "16 - 4 = 12"
        hint_prompt = build_math_hint_prompt(q, gold, reasoning)

        self.assertIn("[Hint: target answer is 12]", hint_prompt)
        self.assertIn("Key step: 16 - 4 = 12", hint_prompt)

        clean = strip_math_hint(hint_prompt)
        self.assertNotIn("[Hint:", clean)
        self.assertNotIn("Key step:", clean)
        self.assertIn(q, clean)
        self.assertIn("Answer: <number>", clean)

    def test_strip_math_hint_without_key_step(self):
        """Assert strip_math_hint works when no key step is present."""
        hint_p = f"What is 2 + 2?\n [Hint: target answer is 4]{MATH_FMT}"
        clean = strip_math_hint(hint_p)
        self.assertNotIn("[Hint:", clean)
        self.assertIn("What is 2 + 2?", clean)

    def test_strip_math_hint_null_and_empty_inputs(self):
        self.assertEqual(strip_math_hint(""), "")
        self.assertEqual(strip_math_hint(None), "")
        self.assertEqual(strip_math_hint("   "), "")

    def test_strip_math_hint_unconditioned_prompt_preservation(self):
        """Assert prompts without hints are left intact."""
        raw = f"Calculate the derivative of x^2.{MATH_FMT}"
        self.assertEqual(strip_math_hint(raw), raw.strip())

    def test_sanitize_hint_completion_catches_all_leakage_phrases(self):
        """Assert all specified leakage phrases trigger rejection."""
        leakage_phrases = [
            "given the hint",
            "based on the hint",
            "using the hint",
            "provided hint",
            "according to the hint",
            "the hint says",
            "from the hint",
        ]
        for phrase in leakage_phrases:
            test_completion = f"Well, {phrase} that the value is 42, we compute 40 + 2 = 42. Answer: 42"
            is_clean, _ = sanitize_hint_completion(test_completion)
            self.assertFalse(
                is_clean,
                f"Failed to catch leakage phrase '{phrase}' in: {test_completion}",
            )

            # Test uppercase / title case variants
            upper_completion = f"Well, {phrase.upper()} that the value is 42. Answer: 42"
            is_clean_up, _ = sanitize_hint_completion(upper_completion)
            self.assertFalse(
                is_clean_up,
                f"Failed to catch uppercase leakage phrase '{phrase.upper()}'",
            )

    def test_sanitize_hint_completion_strips_echoed_header(self):
        """Assert echoed hint headers at the start are cleanly stripped."""
        raw = "[Hint: target answer is 42] 40 + 2 = 42. Answer: 42"
        is_clean, cleaned = sanitize_hint_completion(raw)
        self.assertTrue(is_clean)
        self.assertEqual(cleaned, "40 + 2 = 42. Answer: 42")
        self.assertFalse(cleaned.startswith("[Hint:"))

    def test_sanitize_hint_completion_null_and_empty(self):
        ok_none, text_none = sanitize_hint_completion(None)
        self.assertTrue(ok_none)
        self.assertEqual(text_none, "")

        ok_empty, text_empty = sanitize_hint_completion("")
        self.assertTrue(ok_empty)
        self.assertEqual(text_empty, "")


class TestMathVerificationAndExtraction(unittest.TestCase):
    """Stress tests for extract_gsm_number and is_math_correct."""

    def test_extract_gsm_number_numeric_types(self):
        # Integer
        self.assertEqual(extract_gsm_number("The answer is 42."), 42.0)
        # Decimal
        self.assertEqual(extract_gsm_number("Result: 3.14159"), 3.14159)
        # Negative
        self.assertEqual(extract_gsm_number("Change is -15 dollars."), -15.0)
        # Comma thousands
        self.assertEqual(extract_gsm_number("Total: 1,250,000 units."), 1250000.0)
        # Scientific notation
        self.assertAlmostEqual(extract_gsm_number("Speed: 3.0e8 m/s"), 3.0e8)

    def test_extract_gsm_number_null_and_invalid(self):
        self.assertIsNone(extract_gsm_number(None))
        self.assertIsNone(extract_gsm_number(""))
        self.assertIsNone(extract_gsm_number("There are no digits here."))

    def test_is_math_correct_gold_comparisons(self):
        # Exact match
        self.assertTrue(is_math_correct("Step 1 = 10. Answer: 18", "18"))
        # Comma in gold
        self.assertTrue(is_math_correct("Answer: 1200", "1,200"))
        # Float gold
        self.assertTrue(is_math_correct("Answer: 25.5", 25.5))
        # Int gold
        self.assertTrue(is_math_correct("Answer: 100", 100))
        # Mismatch
        self.assertFalse(is_math_correct("Answer: 19", "18"))
        # None inputs
        self.assertFalse(is_math_correct(None, "18"))
        self.assertFalse(is_math_correct("Answer: 18", None))
        self.assertFalse(is_math_correct(None, None))
        # Malformed gold
        self.assertFalse(is_math_correct("Answer: 18", "not_a_number"))


class TestDryRunPipelineIntegrity(unittest.TestCase):
    """Verify that dry-run execution runs end-to-end and satisfies schemas."""

    def test_projection_math_pipeline_dry_run(self):
        rows = projection_math_pipeline(limit=10, dry_run=True, k_proposals=4)
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertIn("messages", r)
            self.assertIn("completion", r)
            self.assertIn("kind", r)
            self.assertIn("gold", r)
            self.assertFalse(r["thinking"])
            # Prompt must not contain hint tags
            self.assertNotIn("[Hint:", r["messages"][0]["content"])
            # Completion must be verified correct
            self.assertTrue(is_math_correct(r["completion"], r["gold"]))


if __name__ == "__main__":
    unittest.main()
