"""Tier 5 Adversarial Coverage Hardening Test Suite (Part 1).

Milestone 5 Phase 2: White-box empirical challenge testing covering:
1. `src/simula_taxonomy.py`: Adversarial prompt generation edge cases (extreme total_n, missing categories, unnormalized alias combinations).
2. `src/critics.py` & `src/rlvr_verify.py`: RLVR edge cases across all 16 constraints (weird punctuation, multiple languages, whitespace extremes, unicode homoglyphs, malformed JSON structures).
3. `src/projection_data.py` & `src/critics.py`: Math gold answer extraction (extreme negative floats, large exponents, fractions, multiple candidate answers in reasoning chain).
4. `src/projection_data.py`: Projection-lite ranking (identical logps, candidate pools with 0 valid proposals, extreme token length candidate variations).

Authored by Challenger 1 (Empirical Challenger).
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simula_taxonomy import (
    TAXONOMY_REGISTRY,
    CATEGORY_ALIASES,
    PERSONAS,
    INOCULATION_PREFIX,
    MATH_SUFFIX,
    get_taxonomy_categories,
    get_category_metadata,
    generate_prompts_for_category,
    generate_full_campaign_prompt_pool,
    is_contaminated_prompt,
    _resolve_category,
    _normalize_category_key,
    _generate_math_item,
    _generate_constraint_item,
)
from src.critics import (
    evaluate_formal_critic,
    evaluate_constitutional_critic,
    dual_critic_pipeline,
    CONSTITUTIONAL_THRESHOLD,
)
from src.projection_data import (
    extract_gsm_number,
    is_math_correct,
    build_math_hint_prompt,
    strip_math_hint,
    sanitize_hint_completion,
    format_inoculated_ifeval,
    rank_math_proposals_by_likelihood,
    projection_math_pipeline,
)
from src.rlvr_verify import verify as verify_constraint


# =====================================================================
# 1. Simula Taxonomy Adversarial & Boundary Stress Tests
# =====================================================================

class TestAdversarialSimulaTaxonomyEdgeCases(unittest.TestCase):
    """Stress tests on prompt generation edge cases in src/simula_taxonomy.py."""

    def test_extreme_total_n_zero_and_negative(self):
        """Verify behavior for extreme small, zero, and negative total_n values."""
        # For non-positive total_n, generate_full_campaign_prompt_pool gracefully bounds
        # per_cat to at least 1, yielding at least 1 prompt per complexity slice per category.
        for extreme_n in [-1000, -50, -1, 0]:
            pool = generate_full_campaign_prompt_pool(total_n=extreme_n, seed=42)
            self.assertIsInstance(pool, list)
            # 9 categories * (1 low + 1 med + 1 high) = 27 minimum prompts
            self.assertEqual(len(pool), 27, f"Expected 27 prompts for total_n={extreme_n}, got {len(pool)}")
            cats_found = {p["category"] for p in pool}
            self.assertEqual(cats_found, set(get_taxonomy_categories()))

    def test_extreme_total_n_small_positive_numbers(self):
        """Verify behavior for total_n between 1 and 8 (less than category count of 9)."""
        for small_n in [1, 2, 5, 8]:
            pool = generate_full_campaign_prompt_pool(total_n=small_n, seed=42)
            self.assertIsInstance(pool, list)
            # Even when total_n < 9, integer division (small_n // 9) = 0 is bounded to max(1, ...),
            # guaranteeing balanced representation of all 9 dimensions.
            self.assertEqual(len(pool), 27)

    def test_extreme_total_n_high_scaling_throughput(self):
        """Verify performance and memory stability at large scale (total_n=50,000)."""
        t0 = time.time()
        # Direct category generation at scale
        prompts = generate_prompts_for_category("identity_persona", "medium", n=5000, seed=123)
        elapsed = time.time() - t0
        self.assertEqual(len(prompts), 5000)
        self.assertLess(elapsed, 2.0, f"Generating 5000 prompts took {elapsed:.2f}s, too slow!")

        # Verify all prompt IDs are unique
        ids = {p["id"] for p in prompts}
        self.assertEqual(len(ids), 5000)

    def test_missing_and_corrupted_categories(self):
        """Verify that invalid, empty, None, and corrupted category names raise ValueError."""
        corrupted_categories = [
            None,
            "",
            "   ",
            "\n\t  ",
            "unknown_category_xyz",
            "12345",
            "__dunder__",
            "$$$special$$$",
            "system_prompt_leak",
            "maths",  # typo for math
            "ifeval_wrong",
        ]
        for bad_cat in corrupted_categories:
            with self.subTest(bad_cat=bad_cat):
                with self.assertRaises(ValueError):
                    generate_prompts_for_category(bad_cat, "low", n=5)

    def test_unnormalized_alias_combinations(self):
        """Test extreme unnormalized alias combinations, slug variants, and punctuation."""
        valid_canonical_aliases = [
            # Canonical directly
            ("identity_persona", "identity_persona"),
            ("step_by_step_math", "step_by_step_math"),
            ("strict_verifiable_constraints", "strict_verifiable_constraints"),
            # Mixed casing
            ("IDENTITY_PERSONA", "identity_persona"),
            ("Step_By_Step_Math", "step_by_step_math"),
            ("Strict_Verifiable_Constraints", "strict_verifiable_constraints"),
            # Hyphenated slugs
            ("identity-persona", "identity_persona"),
            ("step-by-step-math", "step_by_step_math"),
            ("strict-verifiable-constraints", "strict_verifiable_constraints"),
            # Spaced display names
            ("Identity & Persona", "identity_persona"),
            ("Identity and Persona", "identity_persona"),
            ("identity and persona", "identity_persona"),
            ("Step-by-Step Math", "step_by_step_math"),
            ("step by step math", "step_by_step_math"),
            ("Strict Verifiable Constraints", "strict_verifiable_constraints"),
            # Defined aliases in CATEGORY_ALIASES
            ("math", "step_by_step_math"),
            ("MATH", "step_by_step_math"),
            ("math_reasoning", "step_by_step_math"),
            ("MATH-REASONING", "step_by_step_math"),
            ("ifeval", "strict_verifiable_constraints"),
            ("IFEVAL", "strict_verifiable_constraints"),
            ("ifeval_constraints", "strict_verifiable_constraints"),
            ("safe_edge_case", "safe_edge_cases"),
            ("SAFE-EDGE-CASE", "safe_edge_cases"),
            # Leading and trailing weird punctuation/spaces
            ("   ---identity_persona---   ", "identity_persona"),
            ("..step_by_step_math..", "step_by_step_math"),
            ("  [ifeval]  ", "strict_verifiable_constraints"),
        ]
        for raw_input, expected_canon in valid_canonical_aliases:
            with self.subTest(raw_input=raw_input):
                resolved = _resolve_category(raw_input)
                self.assertEqual(resolved, expected_canon)
                res = generate_prompts_for_category(raw_input, "low", n=1, seed=42)
                self.assertEqual(len(res), 1)
                self.assertEqual(res[0]["category"], expected_canon)

    def test_unsupported_complexity_levels(self):
        """Verify invalid complexity inputs strictly raise ValueError."""
        invalid_complexities = [
            "LOW",
            "Medium ",
            "HIGH\n",
            "extreme",
            "super_high",
            "",
            "   ",
            None,
            1,
            3.0,
            "easy",
        ]
        for bad_comp in invalid_complexities:
            with self.subTest(bad_comp=bad_comp):
                with self.assertRaises(ValueError):
                    generate_prompts_for_category("step_by_step_math", bad_comp, n=5)

    def test_procedural_math_large_index_invariants(self):
        """Verify procedural math templates produce valid, non-negative, and exact answers at extreme indices."""
        rng = random.Random(42)
        extreme_indices = [0, 1, 7, 13, 99, 1000, 10007, 100003, 999999]
        for comp in ("low", "medium", "high"):
            for idx in extreme_indices:
                with self.subTest(complexity=comp, index=idx):
                    prompt, c_info = _generate_math_item(comp, idx, rng)
                    self.assertIn("type", c_info)
                    self.assertEqual(c_info["type"], "math")
                    gold = float(c_info["gold"])
                    # Invariant: gold answer must be positive integer
                    self.assertGreater(gold, 0)
                    self.assertTrue(gold.is_integer())
                    # Invariant: prompt must terminate in MATH_SUFFIX
                    self.assertTrue(prompt.endswith(MATH_SUFFIX))

    def test_persona_cycling_beyond_base_templates(self):
        """Verify that requesting n > len(templates) applies stylistic modulation across all 6 personas."""
        prompts = generate_prompts_for_category("epistemic_humility", "high", n=30, seed=42)
        self.assertEqual(len(prompts), 30)
        # Check all 6 personas are assigned in round-robin
        personas_used = {p["persona"] for p in prompts}
        self.assertEqual(personas_used, set(PERSONAS))

        # Check persona modulation prefixes on cycled items (i >= 5)
        modulated_items = prompts[5:]
        prefixes_found = any("expert perspective" in p["prompt"].lower() for p in modulated_items)
        self.assertTrue(prefixes_found)


# =====================================================================
# 2. RLVR Formal Critic Edge Cases Across All 16 Constraints
# =====================================================================

class TestRLVREdgeCases16Constraints(unittest.TestCase):
    """Stress tests on all 16 deterministic RLVR constraints with adversarial conditions."""

    # 1. validate_lowercase
    def test_c1_lowercase_weird_punctuation_multilingual_unicode(self):
        gt = {"func_name": "validate_lowercase"}
        # German Eszett (ß) is lowercase in Python
        self.assertTrue(verify_constraint("es gibt große straßen", gt))
        self.assertFalse(verify_constraint("Es gibt große straßen", gt))
        # Greek lowercase
        self.assertTrue(verify_constraint("γεια σου κόσμε 123!?", gt))
        self.assertFalse(verify_constraint("Γεια σου κόσμε 123!?", gt))
        # Cyrillic lowercase
        self.assertTrue(verify_constraint("здравствуйте, мир!", gt))
        self.assertFalse(verify_constraint("Здравствуйте, мир!", gt))
        # Full-width Unicode Latin (homoglyphs)
        self.assertTrue(verify_constraint("ａｂｃ １２３", gt))  # U+FF41
        self.assertFalse(verify_constraint("ＡＢＣ １２３", gt))  # U+FF21
        # Whitespace and extreme punctuation
        self.assertTrue(verify_constraint("   \n\t...?!@#$%^&*()—_+=   \n\t", gt))

    # 2. validate_uppercase
    def test_c2_uppercase_multilingual_and_accents(self):
        gt = {"func_name": "validate_uppercase"}
        self.assertTrue(verify_constraint("HELLO WORLD! 123", gt))
        self.assertFalse(verify_constraint("Hello World! 123", gt))
        self.assertTrue(verify_constraint("ПРИВЕТ МИР!", gt))
        self.assertFalse(verify_constraint("Привет мир!", gt))
        self.assertTrue(verify_constraint("STRASSE UND BÜCHER", gt))
        # Full-width uppercase
        self.assertTrue(verify_constraint("ＡＢＣ １２３", gt))
        self.assertFalse(verify_constraint("ａｂｃ １２３", gt))

    # 3. validate_no_commas
    def test_c3_no_commas_unicode_and_whitespace(self):
        gt = {"func_name": "validate_no_commas"}
        self.assertTrue(verify_constraint("This sentence has no commas at all.", gt))
        self.assertFalse(verify_constraint("Here is one, right here.", gt))
        # Unicode comma lookalikes: full-width comma (，) vs standard ASCII (,)
        # Note: ASCII comma check only flags ','
        self.assertTrue(verify_constraint("Chinese sentence with ideographic comma：这是测试、没有半角逗号", gt))
        # Extreme whitespace without commas
        self.assertTrue(verify_constraint(" \t\n word1 \n\n word2 \r\n ", gt))

    # 4. validate_quotation
    def test_c4_quotation_single_char_nested_curly(self):
        gt = {"func_name": "validate_quotation"}
        # Valid standard double quotes
        self.assertTrue(verify_constraint('"This is properly enclosed in quotes."', gt))
        self.assertTrue(verify_constraint('  "Enclosed with leading and trailing whitespace"  ', gt))
        # Edge cases: empty string or single quote
        self.assertFalse(verify_constraint('"', gt))
        self.assertFalse(verify_constraint('""', gt) is False or len('""') <= 1)  # len > 1 and t[0]=='"' and t[-1]=='"' -> True for '""'
        self.assertTrue(verify_constraint('""', gt))
        # Single quotes (should fail)
        self.assertFalse(verify_constraint("'Single quoted text'", gt))
        # Curly smart quotes (U+201C, U+201D) - should fail ASCII quote check
        self.assertFalse(verify_constraint('“Smart quotes”', gt))
        # Nested quotes inside valid outer quotes
        self.assertTrue(verify_constraint('"He whispered: \'Do not speak\' softly."', gt))

    # 5. validate_json_format
    def test_c5_json_format_malformed_and_extreme_structures(self):
        gt = {"func_name": "validate_json_format"}
        # Valid raw JSON
        self.assertTrue(verify_constraint('{"concept": "gravity", "score": 9.5}', gt))
        # Valid markdown fenced JSON
        self.assertTrue(verify_constraint('```json\n{"status": "ok", "items": [1, 2, 3]}\n```', gt))
        self.assertTrue(verify_constraint('```\n{"status": "ok"}\n```', gt))
        # Valid nested unicode structures
        self.assertTrue(verify_constraint('{"unicode": "🚀", "nested": {"array": [null, true, false]}}', gt))
        # Malformed JSON cases
        self.assertFalse(verify_constraint('{"unquoted_key": val}', gt))
        self.assertFalse(verify_constraint("{'single_quotes': 'invalid_json'}", gt))
        self.assertFalse(verify_constraint('{"trailing_comma": 1,}', gt))
        self.assertFalse(verify_constraint('{"unclosed": [1, 2', gt))
        self.assertFalse(verify_constraint('```json\n{"broken": }\n```', gt))
        self.assertFalse(verify_constraint('', gt))
        self.assertFalse(verify_constraint('   ', gt))

    # 6. validate_title
    def test_c6_title_angle_brackets_edge_cases(self):
        gt = {"func_name": "validate_title"}
        self.assertTrue(verify_constraint("Here is the topic: <<The History of Computing>> today.", gt))
        self.assertTrue(verify_constraint("<<Single Word Title>>", gt))
        self.assertTrue(verify_constraint("<<Заголовок на русском>>", gt))
        # Empty angle brackets should fail ([^\n]+ requires >= 1 char)
        self.assertFalse(verify_constraint("Empty brackets: <<>>", gt))
        # Broken brackets
        self.assertFalse(verify_constraint("Single bracket: <Title>", gt))
        self.assertFalse(verify_constraint("Mismatched: <<Title>", gt))
        # Multi-line title (<[^\n]+> does not match across newlines)
        self.assertFalse(verify_constraint("<<Title split across\nlines>>", gt))

    # 7. validate_end
    def test_c7_end_phrase_case_and_whitespace(self):
        end_phrase = "That concludes this summary."
        gt = {"func_name": "validate_end", "end_phrase": end_phrase}
        self.assertTrue(verify_constraint(f"Detailed overview of biology. {end_phrase}", gt))
        self.assertTrue(verify_constraint(f"Multi-line\noverview.\n{end_phrase}   ", gt))
        # Case mismatch
        self.assertFalse(verify_constraint("Detailed overview. that concludes this summary.", gt))
        # Punctuation mismatch
        self.assertFalse(verify_constraint("Detailed overview. That concludes this summary!", gt))
        # Extra characters after end phrase
        self.assertFalse(verify_constraint(f"Detailed overview. {end_phrase} Extra word.", gt))

    # 8. verify_postscript
    def test_c8_postscript_markers_special_regex_chars(self):
        marker = "P.S. [Note]:"
        gt = {"func_name": "verify_postscript", "postscript_marker": marker}
        self.assertTrue(verify_constraint(f"Main body of letter.\n\n{marker} Remember this.", gt))
        self.assertTrue(verify_constraint(f"Main body. {marker} Note.", gt))
        # Missing marker
        self.assertFalse(verify_constraint("Main body without marker. Just a postscript note.", gt))

    # 9. validate_placeholders
    def test_c9_placeholders_bracket_variations(self):
        gt = {"func_name": "validate_placeholders", "N": 3}
        self.assertTrue(verify_constraint("Enter [name], then select [date], and submit [code].", gt))
        # 2 placeholders when 3 required
        self.assertFalse(verify_constraint("Enter [name] and [date].", gt))
        # Empty brackets do not match ([^\]]+ requires >= 1 char)
        self.assertFalse(verify_constraint("[] [] []", gt))

    # 10. validate_highlighted_sections
    def test_c10_highlighted_sections_asterisk_quirks(self):
        gt = {"func_name": "validate_highlighted_sections", "N": 2}
        self.assertTrue(verify_constraint("Focus on *first concept* and also *second concept*.", gt))
        self.assertFalse(verify_constraint("Only *one highlighted concept* here.", gt))
        # Empty asterisks do not match ([^\n\*]+ requires >= 1 char)
        self.assertFalse(verify_constraint("Here are ** and ** empty asterisks.", gt))

    # 11. validate_sections
    def test_c11_sections_splitter_regex_and_numbering(self):
        gt = {"func_name": "validate_sections", "section_splitter": "Phase", "N": 2}
        text = "Introduction. Phase 1: Planning and research. Phase 2: Implementation."
        self.assertTrue(verify_constraint(text, gt))
        text_insufficient = "Introduction. Phase 1: Planning only."
        self.assertFalse(verify_constraint(text_insufficient, gt))

    # 12. verify_bullet_points
    def test_c12_bullet_points_whitespace_and_symbols(self):
        gt = {"func_name": "verify_bullet_points", "N": 3}
        # Mixed * and - bullets
        text_mixed = "* Point 1\n- Point 2\n* Point 3"
        self.assertTrue(verify_constraint(text_mixed, gt))
        # Indented bullets
        text_indented = "  * Indented point 1\n\t- Tab point 2\n   * Space point 3"
        self.assertTrue(verify_constraint(text_indented, gt))
        # Unicode bullet points (•) should NOT count towards * and - regex
        text_unicode_bullets = "• Bullet 1\n• Bullet 2\n• Bullet 3"
        self.assertFalse(verify_constraint(text_unicode_bullets, gt))
        # Mismatched count
        text_four = "* One\n* Two\n* Three\n* Four"
        self.assertFalse(verify_constraint(text_four, gt))

    # 13. verify_keywords
    def test_c13_keywords_case_insensitivity_and_punctuation(self):
        gt = {"func_name": "verify_keywords", "keyword_list": ["C++", "TCP/IP", "micro-architecture"]}
        self.assertTrue(verify_constraint("We analyzed the c++ code over tcp/ip with micro-architecture focus.", gt))
        self.assertFalse(verify_constraint("We analyzed the python code over tcp/ip.", gt))

    # 14. verify_keyword_frequency
    def test_c14_keyword_frequency_word_boundaries(self):
        gt = {"func_name": "verify_keyword_frequency", "word": "cat", "N": 3}
        # Exact word boundaries
        self.assertTrue(verify_constraint("The cat saw another cat, and then a third cat.", gt))
        # Substring in other words (caterpillar, catch) must NOT count
        self.assertFalse(verify_constraint("The caterpillar will catch the cat.", gt))

    # 15. validate_forbidden_words
    def test_c15_forbidden_words_substrings_and_boundaries(self):
        gt = {"func_name": "validate_forbidden_words", "forbidden_words": ["bad", "hack"]}
        # Substring within benign words: 'badminton', 'shack' must NOT trigger forbidden word check
        self.assertTrue(verify_constraint("Playing badminton near the beach shack is fun.", gt))
        # Actual forbidden words must trigger failure
        self.assertFalse(verify_constraint("This is a bad practice.", gt))
        self.assertFalse(verify_constraint("Attempting to hack the server.", gt))

    # 16. verify_letter_frequency
    def test_c16_letter_frequency_case_and_diacritics(self):
        gt = {"func_name": "verify_letter_frequency", "letter": "z", "N": 3}
        self.assertTrue(verify_constraint("Zebra, blizzard, and buzz.", gt))
        self.assertFalse(verify_constraint("Zebra and zoo.", gt))

    def test_compound_and_nested_constraints_in_formal_critic(self):
        """Verify evaluate_formal_critic handling of compound lists and nested dicts."""
        prompt = "Write a response."
        # List of 2 constraints: lowercase AND no commas
        c_list = [
            {"func_name": "validate_lowercase"},
            {"func_name": "validate_no_commas"},
        ]
        # Passes both
        ok, reason = evaluate_formal_critic(prompt, "all lowercase without any breaks", c_list)
        self.assertTrue(ok)
        self.assertIn("satisfied", reason)

        # Passes first, fails second
        ok_fail, reason_fail = evaluate_formal_critic(prompt, "all lowercase, but with a comma", c_list)
        self.assertFalse(ok_fail)
        self.assertIn("Constraint #2 failed", reason_fail)

        # Nested dict with 'rules'
        c_rules = {"rules": [{"func_name": "validate_uppercase"}]}
        ok_up, _ = evaluate_formal_critic(prompt, "ALL UPPERCASE TEXT", c_rules)
        self.assertTrue(ok_up)

        # Corrupted JSON string constraint_info
        ok_bad_json, reason_bad = evaluate_formal_critic(prompt, "Text", "{corrupted: json")
        self.assertFalse(ok_bad_json)
        self.assertIn("Invalid constraint_info JSON string", reason_bad)


# =====================================================================
# 3. Math Gold Answer Extraction & Robustness
# =====================================================================

class TestMathGoldAnswerExtraction(unittest.TestCase):
    """Stress tests on number parsing, extreme values, fractions, and reasoning chains."""

    def test_extreme_negative_floats(self):
        """Verify extraction and validation of extreme negative floats and decimals."""
        neg_samples = [
            ("The temperature fell to -45.5 degrees. Answer: -45.5", -45.5),
            ("Calculated loss is -123456.789 dollars. Answer: -123456.789", -123456.789),
            ("Net change: -0.0005. Answer: -0.0005", -0.0005),
        ]
        for text, expected in neg_samples:
            num = extract_gsm_number(text)
            self.assertIsNotNone(num)
            self.assertAlmostEqual(num, expected, places=5)
            self.assertTrue(is_math_correct(text, expected))

    def test_large_exponents_scientific_notation(self):
        """Verify extraction of positive and negative scientific notation."""
        sci_samples = [
            ("Speed of light is 3.0e8 m/s. Answer: 3.0e8", 3.0e8),
            ("Avogadro constant: 6.022E23 particles. Answer: 6.022E23", 6.022e23),
            ("Charge of electron: -1.6e-19 coulombs. Answer: -1.6e-19", -1.6e-19),
            ("Small scale: 2.5E-5 units. Answer: 2.5E-5", 2.5e-5),
        ]
        for text, expected in sci_samples:
            num = extract_gsm_number(text)
            self.assertIsNotNone(num)
            self.assertAlmostEqual(num, expected, delta=abs(expected * 1e-5))
            self.assertTrue(is_math_correct(text, expected))

    def test_fractions_and_decimal_representations(self):
        """Empirical verification of fraction parsing behavior."""
        # Note: regex extracts individual numbers in a fraction '3/4' -> '3' and '4'
        # The last number is 4.0. If the final answer is stated as a decimal 0.75, it matches.
        text_frac = "The probability is 3/4, which equals 0.75. Answer: 0.75"
        self.assertEqual(extract_gsm_number(text_frac), 0.75)
        self.assertTrue(is_math_correct(text_frac, "0.75"))

        # When only raw fraction '3/4' is present, last number extracted is denominator 4.0
        text_raw_frac = "The remaining fraction is 3/4."
        self.assertEqual(extract_gsm_number(text_raw_frac), 4.0)

    def test_multiple_candidate_answers_in_reasoning_chain(self):
        """Verify that extract_gsm_number picks the final answer in multi-step reasoning."""
        chain = (
            "Step 1: Alice has 50 apples.\n"
            "Step 2: She sells 15, leaving 50 - 15 = 35 apples.\n"
            "Step 3: She gives 10 to Bob, leaving 35 - 10 = 25 apples.\n"
            "Step 4: Bob returns 5, giving 25 + 5 = 30 apples.\n"
            "Answer: 30"
        )
        self.assertEqual(extract_gsm_number(chain), 30.0)
        self.assertTrue(is_math_correct(chain, 30))

    def test_currency_and_comma_formatting(self):
        """Verify extraction and comparison with currency signs and thousands commas."""
        # 1. is_math_correct handles currency signs in assistant text and commas in gold_str
        currency_samples = [
            ("Total cost is $1,500. Answer: 1500", "1500", True),
            ("Total cost is $1,500. Answer: $1,500", "1,500", True),
            ("Grand total: $2,500,000. Answer: $2,500,000", "2,500,000", True),
            ("Budget is 45,000 dollars.", 45000, True),
        ]
        for text, gold, expected_ok in currency_samples:
            ok = is_math_correct(text, gold)
            self.assertEqual(ok, expected_ok)

        # 2. is_math_correct expects gold_str to be numeric/comma-only; raw '$' in gold_str returns False
        self.assertFalse(is_math_correct("Answer: 1500", "$1,500"))

        # 3. evaluate_formal_critic in src/critics.py pre-normalizes currency signs in gold
        ok_crit, reason = evaluate_formal_critic(
            "What is total?", "Answer: $2,500,000", {"gold": "$2,500,000"}
        )
        self.assertTrue(ok_crit, f"evaluate_formal_critic failed on currency gold: {reason}")

    def test_trailing_text_with_numbers_hazard(self):
        """Adversarial stress test: trailing text containing numbers after Answer: <num>.
        
        Demonstrates why completions must place Answer: <num> at the very end.
        """
        # If candidate assistant adds commentary with numbers after the answer:
        text_with_trailing_num = "We compute 20 + 22 = 42. Answer: 42. Solved in 2 steps."
        # extract_gsm_number picks 2.0 (the last number in the text)
        extracted = extract_gsm_number(text_with_trailing_num)
        self.assertEqual(extracted, 2.0)
        # Therefore, formal critic correctly rejects this candidate against gold 42:
        self.assertFalse(is_math_correct(text_with_trailing_num, 42))

        # When Answer: 42 is placed on the final line without trailing numbers:
        clean_text = "We compute 20 + 22 = 42. Solved in 2 steps.\nAnswer: 42"
        self.assertEqual(extract_gsm_number(clean_text), 42.0)
        self.assertTrue(is_math_correct(clean_text, 42))

    def test_is_math_correct_floating_point_tolerances(self):
        """Verify epsilon tolerance (1e-6) and zero handling."""
        # Zero equality
        self.assertTrue(is_math_correct("Answer: 0", 0))
        self.assertTrue(is_math_correct("Answer: 0.0", "0"))
        self.assertTrue(is_math_correct("Answer: -0.0", 0.0))
        # Within tolerance (< 1e-6)
        self.assertTrue(is_math_correct("Answer: 10.0000001", 10.0))
        # Outside tolerance (>= 1e-6)
        self.assertFalse(is_math_correct("Answer: 10.00001", 10.0))
        # Null / invalid inputs
        self.assertFalse(is_math_correct(None, 10))
        self.assertFalse(is_math_correct("Answer: 10", None))
        self.assertFalse(is_math_correct(None, None))
        self.assertFalse(is_math_correct("Answer: 10", "not_a_number"))


# =====================================================================
# 4. Projection-Lite Proposal Ranking & Sanitization Edge Cases
# =====================================================================

class TestProjectionLiteRankingEdgeCases(unittest.TestCase):
    """Stress tests on proposal ranking, tie-breaking, empty pools, and hint leakage."""

    def test_identical_logps_and_scores_tie_breaking(self):
        """Verify deterministic tie-breaking when candidate scores or logps are identical."""
        cand_a = "Method A: 10 + 10 = 20. Answer: 20"
        cand_b = "Method B: 15 + 05 = 20. Answer: 20"
        cand_c = "Method C: 12 + 08 = 20. Answer: 20"
        candidates = [cand_a, cand_b, cand_c]

        # 1. Identical mock scores (all -2.5) -> must select the first candidate (cand_a)
        winner_mock = rank_math_proposals_by_likelihood(
            None, None, "Prompt", candidates, mock_scores=[-2.5, -2.5, -2.5]
        )
        self.assertEqual(winner_mock, cand_a)

        # 2. Identical heuristic scores in offline mode
        # cand_a, cand_b, cand_c have identical length (34 chars) and both contain '=' and 'Answer:'
        self.assertEqual(len(cand_a), len(cand_b))
        self.assertEqual(len(cand_b), len(cand_c))
        winner_heuristic = rank_math_proposals_by_likelihood(
            None, None, "Prompt", candidates, offline=True
        )
        self.assertEqual(winner_heuristic, cand_a)

        # Reordering candidates changes the first item to cand_b
        winner_reordered = rank_math_proposals_by_likelihood(
            None, None, "Prompt", [cand_b, cand_a, cand_c], offline=True
        )
        self.assertEqual(winner_reordered, cand_b)

    def test_candidate_pools_with_zero_valid_proposals(self):
        """Verify handling when candidate pool is empty or has 0 valid proposals."""
        # Direct call with empty list must raise ValueError
        with self.assertRaises(ValueError):
            rank_math_proposals_by_likelihood(None, None, "Prompt", [])

        # In pipeline scenario: candidate pool with 0 valid proposals (all incorrect answers)
        gold = "42"
        mock_raw_proposals = [
            "Calculation error: 10 + 10 = 20. Answer: 20",
            "Wrong result: 50 - 5 = 45. Answer: 45",
            "Based on the hint, the answer is 42. Answer: 42",  # Leaked hint phrase
        ]
        valid_proposals = [
            p for p in mock_raw_proposals
            if is_math_correct(p, gold) and sanitize_hint_completion(p)[0]
        ]
        # All 3 fail either math correctness or hint sanitization
        self.assertEqual(len(valid_proposals), 0)
        # Pipeline must safely check `if valid_proposals:` before ranking

    def test_extreme_token_length_candidate_variations(self):
        """Verify ranking behavior with extreme token length variation (10 chars vs 10,000 chars)."""
        prompt = "Compute 15 * 4."
        gold = "60"
        concise_cand = "15 * 4 = 60. Answer: 60"  # 23 chars
        rambling_cand = (
            "To solve this problem, we will use basic multiplication principles. " * 150
            + "Thus 15 * 4 = 60. Answer: 60"
        )  # ~10,000 chars

        candidates = [rambling_cand, concise_cand]
        winner = rank_math_proposals_by_likelihood(None, None, prompt, candidates, offline=True)
        # Offline heuristic penalizes excess length (-len) and rewards concise format
        self.assertEqual(winner, concise_cand)

    def test_mock_scores_validation_and_ordering(self):
        """Verify mock_scores length validation and proper maximum selection."""
        candidates = ["Sol A", "Sol B", "Sol C", "Sol D"]
        # Mismatched length raises ValueError
        with self.assertRaises(ValueError):
            rank_math_proposals_by_likelihood(None, None, "P", candidates, mock_scores=[-1.0, -2.0])

        # Correct selection: highest log-likelihood (least negative)
        scores = [-15.2, -8.4, -2.1, -19.0]
        winner = rank_math_proposals_by_likelihood(None, None, "P", candidates, mock_scores=scores)
        self.assertEqual(winner, "Sol C")  # -2.1 is highest

    def test_sanitize_hint_completion_leakage_phrases_variations(self):
        """Verify rejection across all 7 hint leakage phrases and case/punctuation variations."""
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
            # Lowercase
            t1 = f"Well, {phrase} that target is 42, we obtain 42. Answer: 42"
            is_clean1, _ = sanitize_hint_completion(t1)
            self.assertFalse(is_clean1, f"Failed to catch phrase: {phrase}")

            # Uppercase
            t2 = f"{phrase.upper()}: the value is 42. Answer: 42"
            is_clean2, _ = sanitize_hint_completion(t2)
            self.assertFalse(is_clean2, f"Failed to catch uppercase phrase: {phrase.upper()}")

            # Title case
            t3 = f"{phrase.title()}, we conclude the result is 42. Answer: 42"
            is_clean3, _ = sanitize_hint_completion(t3)
            self.assertFalse(is_clean3, f"Failed to catch title case phrase: {phrase.title()}")

        # Echoed header stripping
        echoed = "[Hint: target answer is 42] 40 + 2 = 42. Answer: 42"
        is_clean_e, cleaned_e = sanitize_hint_completion(echoed)
        self.assertTrue(is_clean_e)
        self.assertEqual(cleaned_e, "40 + 2 = 42. Answer: 42")

    def test_strip_math_hint_various_formats(self):
        """Verify strip_math_hint removes hint and key step annotations cleanly."""
        hint_prompt = (
            "A basket contains 20 apples.\n"
            " [Hint: target answer is 15] Key step: 20 - 5 = 15\n"
            "Give the final answer on the last line as 'Answer: <number>'."
        )
        clean = strip_math_hint(hint_prompt)
        self.assertNotIn("[Hint:", clean)
        self.assertNotIn("Key step:", clean)
        self.assertIn("A basket contains 20 apples.", clean)
        self.assertIn("Answer: <number>", clean)

        # Idempotency: stripping twice
        clean_twice = strip_math_hint(clean)
        self.assertEqual(clean, clean_twice)

    def test_format_inoculated_ifeval_type_and_idempotency_stress(self):
        """Verify format_inoculated_ifeval idempotency and null-safety."""
        # None inputs
        res_none = format_inoculated_ifeval(None, None)
        self.assertEqual(res_none["messages"][0]["content"], INOCULATION_PREFIX)
        self.assertEqual(res_none["completion"], "")

        # Chaining 5x wrapping
        curr = "Write a haiku about autumn."
        for _ in range(5):
            rec = format_inoculated_ifeval(curr, "Leaves fall softly down.")
            curr = rec["messages"][0]["content"]
            self.assertEqual(curr.count(INOCULATION_PREFIX), 1)

        # Preservation of numeric/boolean/dict constraints
        self.assertEqual(format_inoculated_ifeval("p", "c", 0)["constraint"], 0)
        self.assertEqual(format_inoculated_ifeval("p", "c", False)["constraint"], False)
        self.assertEqual(format_inoculated_ifeval("p", "c", {"rule": 1})["constraint"], {"rule": 1})

    def test_sanitize_hint_completion_benign_hint_word(self):
        """Verify that benign, natural uses of the word 'hint' do not trigger false positive rejection."""
        benign_samples = [
            "Add a subtle hint of cinnamon to the mixture. Answer: 5",
            "She dropped a hint about her upcoming birthday party. Answer: 12",
            "There was not a single hint of doubt in their minds. Answer: 100",
        ]
        for sample in benign_samples:
            is_clean, cleaned = sanitize_hint_completion(sample)
            self.assertTrue(is_clean, f"False positive leakage on benign sentence: {sample}")
            self.assertEqual(cleaned, sample)

    def test_single_candidate_returned_directly(self):
        """Verify candidate list of length 1 returns candidate directly without calling models or heuristics."""
        cand = "Only one candidate. Answer: 42"
        # Mismatched mock_scores length ignored because single candidate short-circuits
        res = rank_math_proposals_by_likelihood(None, None, "Prompt", [cand], mock_scores=[])
        self.assertEqual(res, cand)

    @patch("src.critics.evaluate_constitutional_critic")
    def test_dual_critic_serial_short_circuit_cost_protection(self, mock_c2):
        """Adversarial verification: Critic 2 must NEVER be called when Critic 1 fails."""
        mock_c2.side_effect = AssertionError("Critic 2 should NEVER be called on formal failure!")

        # 1. Math failure
        cand_math = {"prompt": "What is 2+2?", "completion": "The answer is 5. Answer: 5", "gold": "4"}
        acc1, meta1 = dual_critic_pipeline(cand_math)
        self.assertFalse(acc1)
        self.assertFalse(meta1["formal_pass"])
        self.assertIsNone(meta1["constitutional_score"])

        # 2. RLVR constraint failure
        cand_rlvr = {
            "prompt": "Write lowercase",
            "completion": "UPPERCASE FAILS",
            "constraint_info": {"func_name": "validate_lowercase"},
        }
        acc2, meta2 = dual_critic_pipeline(cand_rlvr)
        self.assertFalse(acc2)
        self.assertFalse(meta2["formal_pass"])
        self.assertIsNone(meta2["constitutional_score"])

        mock_c2.assert_not_called()


class TestExtendedRLVRConstraintsAndSanitization(unittest.TestCase):
    """Additional stress tests on complex RLVR constraints and edge conditions."""

    def test_validate_two_responses_edge_cases(self):
        gt = {"func_name": "validate_two_responses"}
        # Exactly 2 distinct responses separated by ******
        valid = "Response A: Hard determinism argues all actions are predetermined.******Response B: Compatibilism argues free will is compatible with determinism."
        self.assertTrue(verify_constraint(valid, gt))

        # Only 1 response (no separator)
        self.assertFalse(verify_constraint("Only one single response here.", gt))

        # Two identical responses (must fail)
        identical = "Identical text.******Identical text."
        self.assertFalse(verify_constraint(identical, gt))

        # 3 responses separated by ****** (must fail)
        three = "Response A******Response B******Response C"
        self.assertFalse(verify_constraint(three, gt))

    def test_verify_sentence_constraint_quantifiers_and_ellipsis(self):
        # 1. Ellipsis handling: '...' counts as 1 sentence ending, not 3
        gt_exact_1 = {"func_name": "verify_sentence_constraint", "N": 1, "quantifier": "exactly"}
        self.assertTrue(verify_constraint("This is a thoughtful sentence...", gt_exact_1))

        # 2. Exactly 2 sentences
        gt_exact_2 = {"func_name": "verify_sentence_constraint", "N": 2, "quantifier": "exactly"}
        self.assertTrue(verify_constraint("First sentence! Second sentence.", gt_exact_2))
        self.assertFalse(verify_constraint("Only one sentence.", gt_exact_2))

        # 3. Quantifier: 'at most' 2 sentences
        gt_at_most_2 = {"func_name": "verify_sentence_constraint", "N": 2, "quantifier": "at most"}
        self.assertTrue(verify_constraint("Only one.", gt_at_most_2))
        self.assertTrue(verify_constraint("One. Two.", gt_at_most_2))
        self.assertFalse(verify_constraint("One. Two. Three.", gt_at_most_2))

        # 4. Quantifier: 'at least' 2 sentences
        gt_at_least_2 = {"func_name": "verify_sentence_constraint", "N": 2, "quantifier": "at least"}
        self.assertFalse(verify_constraint("One.", gt_at_least_2))
        self.assertTrue(verify_constraint("One. Two. Three.", gt_at_least_2))

    def test_validate_word_constraint_quantifiers(self):
        text_5_words = "One two three four five."
        self.assertTrue(verify_constraint(text_5_words, {"func_name": "validate_word_constraint", "N": 5, "quantifier": "exactly"}))
        self.assertTrue(verify_constraint(text_5_words, {"func_name": "validate_word_constraint", "N": 10, "quantifier": "at most"}))
        self.assertFalse(verify_constraint(text_5_words, {"func_name": "validate_word_constraint", "N": 3, "quantifier": "at most"}))
        self.assertTrue(verify_constraint(text_5_words, {"func_name": "validate_word_constraint", "N": 3, "quantifier": "at least"}))

    def test_validate_choice_options(self):
        gt = {"func_name": "validate_choice", "options": "Option A, Option B, Option C"}
        self.assertTrue(verify_constraint("I select Option B as the correct answer.", gt))
        self.assertFalse(verify_constraint("I select Option D as the answer.", gt))

    def test_validate_repeat_prompt(self):
        orig_prompt = "What is the capital of France? First, repeat the request."
        gt = {"func_name": "validate_repeat_prompt", "original_prompt": orig_prompt}
        valid_comp = "What is the capital of France?\nThe capital of France is Paris."
        self.assertTrue(verify_constraint(valid_comp, gt))
        invalid_comp = "The capital of France is Paris without repeating."
        self.assertFalse(verify_constraint(invalid_comp, gt))

    def test_get_category_metadata_unnormalized_and_display_names(self):
        """Verify get_category_metadata resolves unnormalized and display name aliases."""
        for alias, canon in [("MATH", "step_by_step_math"), ("Safe Edge-Case Handling", "safe_edge_cases")]:
            meta = get_category_metadata(alias)
            self.assertEqual(meta["canonical_id"], canon)
            self.assertIsInstance(meta["subcategories"], list)
            self.assertGreater(len(meta["subcategories"]), 0)


if __name__ == "__main__":
    unittest.main()

