"""Adversarial stress test suite for Simula Taxonomy (Milestone 1).
Written by Challenger 1 (Empirical Challenger).

Verifies:
1. High-Volume Generation across all 9 categories and 3 complexities (throughput & ID uniqueness).
2. Zero Decontamination against evals/heldout_ids.json and evals/probes.jsonl.
3. Inoculation Prefix Isolation (exclusively on strict_verifiable_constraints).
4. Schema Conformance (keys, types, non-empty substantive prompts).
5. Adversarial Input Robustness (n=0, n<0, extreme n, bad categories, bad complexities, seed reproducibility).
6. Math Domain Invariants & Physical Realism (non-negative quantities, exact decimal currency, discrete units).
"""
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simula_taxonomy import (
    get_taxonomy_categories,
    get_category_metadata,
    generate_prompts_for_category,
    generate_full_campaign_prompt_pool,
    is_contaminated_prompt,
    _generate_math_item,
    INOCULATION_PREFIX,
    TAXONOMY_REGISTRY,
    PERSONAS,
)
from src.critics import evaluate_formal_critic

HELDOUT_PATH = ROOT / "evals" / "heldout_ids.json"
PROBES_PATH = ROOT / "evals" / "probes.jsonl"


class TestHighVolumeGeneration(unittest.TestCase):
    """Stress tests high-volume generation scale, speed, and ID uniqueness."""

    def test_high_volume_generation_throughput(self):
        categories = get_taxonomy_categories()
        complexities = ["low", "medium", "high"]
        n_per_slice = 1000
        total_expected = len(categories) * len(complexities) * n_per_slice

        t0 = time.time()
        generated_count = 0
        all_ids = set()

        for cat in categories:
            for comp in complexities:
                prompts = generate_prompts_for_category(cat, comp, n=n_per_slice)
                self.assertEqual(len(prompts), n_per_slice)
                generated_count += len(prompts)
                for p in prompts:
                    all_ids.add(p["id"])

        elapsed = time.time() - t0
        self.assertEqual(generated_count, total_expected)
        self.assertEqual(len(all_ids), total_expected, "Every generated ID must be unique across all slices")
        # Assert throughput is at least 10,000 items/second
        throughput = generated_count / max(0.001, elapsed)
        self.assertGreater(throughput, 10000.0, f"Throughput too low: {throughput} items/s")

    def test_full_campaign_prompt_pool_scaling(self):
        for target_n in [0, 1, 9, 27, 90, 300, 1000]:
            pool = generate_full_campaign_prompt_pool(total_n=target_n, seed=42)
            self.assertIsInstance(pool, list)
            self.assertGreater(len(pool), 0)
            cats_present = {p["category"] for p in pool}
            self.assertEqual(cats_present, set(get_taxonomy_categories()))


class TestDecontaminationVerification(unittest.TestCase):
    """Rigorous verification that zero generated prompts leak or collide with held-out evals."""

    @classmethod
    def setUpClass(cls):
        cls.heldout_texts = set()
        cls.heldout_base_q = set()
        cls.heldout_src_ids = set()
        if HELDOUT_PATH.exists():
            with open(HELDOUT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            cls.heldout_texts = {t.strip().lower() for t in data.get("texts", [])}
            cls.heldout_base_q = {q.strip().lower() for q in data.get("base_q", [])}
            cls.heldout_src_ids = {str(sid).strip().lower() for sid in data.get("src_ids", [])}

        cls.probe_turns = set()
        cls.probe_long_substrings = []
        if PROBES_PATH.exists():
            with open(PROBES_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        for turn in item.get("turns", []):
                            clean = turn.strip().lower()
                            cls.probe_turns.add(clean)
                            if len(clean) > 20:
                                cls.probe_long_substrings.append(clean)

        cls.all_eval_exact = cls.heldout_texts | cls.heldout_base_q | cls.heldout_src_ids | cls.probe_turns

    def test_zero_exact_match_collisions(self):
        categories = get_taxonomy_categories()
        complexities = ["low", "medium", "high"]

        for cat in categories:
            for comp in complexities:
                prompts = generate_prompts_for_category(cat, comp, n=50)
                for p in prompts:
                    clean = p["prompt"].strip().lower()
                    self.assertNotIn(
                        clean,
                        self.all_eval_exact,
                        f"CRITICAL CONTAMINATION: Generated prompt matched held-out eval exactly: '{clean}'"
                    )

    def test_zero_substring_contamination_on_long_probes(self):
        categories = get_taxonomy_categories()
        complexities = ["low", "medium", "high"]

        for cat in categories:
            for comp in complexities:
                prompts = generate_prompts_for_category(cat, comp, n=30)
                for p in prompts:
                    clean = p["prompt"].strip().lower()
                    for probe_sub in self.probe_long_substrings:
                        self.assertNotIn(
                            probe_sub,
                            clean,
                            f"CRITICAL CONTAMINATION: Probe substring '{probe_sub}' found inside generated prompt '{clean}'"
                        )

    def test_decontamination_screening_oracle_accuracy(self):
        # Verify that is_contaminated_prompt correctly identifies heldout strings
        test_samples = list(self.heldout_texts)[:10] + list(self.probe_turns)[:10]
        for s in test_samples:
            self.assertTrue(
                is_contaminated_prompt(s),
                f"Decontamination filter failed to flag known heldout string: '{s}'"
            )


class TestInoculationPrefixIsolation(unittest.TestCase):
    """Asserts that the inoculation prefix is strictly quarantined to strict_verifiable_constraints."""

    def test_inoculation_prefix_strictly_isolated(self):
        categories = get_taxonomy_categories()
        complexities = ["low", "medium", "high"]

        for cat in categories:
            for comp in complexities:
                prompts = generate_prompts_for_category(cat, comp, n=40)
                for p in prompts:
                    if cat == "strict_verifiable_constraints":
                        self.assertTrue(
                            p["prompt"].startswith(INOCULATION_PREFIX),
                            f"Expected prompt in {cat} to start with '{INOCULATION_PREFIX}'"
                        )
                        self.assertTrue(p["inoculated"])
                        self.assertIsInstance(p["constraint_info"], dict)
                        self.assertIn("func_name", p["constraint_info"])
                    else:
                        self.assertNotIn(
                            INOCULATION_PREFIX,
                            p["prompt"],
                            f"LEAKED INOCULATION PREFIX: Non-verifiable category '{cat}' contains inoculation prefix!"
                        )
                        self.assertFalse(
                            p["inoculated"],
                            f"Category '{cat}' has inoculated=True but must be False!"
                        )


class TestSchemaConformance(unittest.TestCase):
    """Asserts schema conformance across all generated metadata dictionaries."""

    def test_metadata_dict_schema(self):
        required_keys = {"id", "category", "complexity", "prompt", "constraint_info"}
        valid_cats = set(get_taxonomy_categories())
        valid_comp = {"low", "medium", "high"}
        valid_personas = set(PERSONAS)

        for cat in valid_cats:
            for comp in valid_comp:
                prompts = generate_prompts_for_category(cat, comp, n=30)
                for p in prompts:
                    self.assertTrue(required_keys.issubset(set(p.keys())))
                    self.assertIsInstance(p["id"], str)
                    self.assertGreater(len(p["id"]), 10)
                    self.assertEqual(p["category"], cat)
                    self.assertEqual(p["complexity"], comp)
                    self.assertIn(p["persona"], valid_personas)
                    self.assertIsInstance(p["prompt"], str)
                    self.assertGreater(len(p["prompt"].strip()), 15)
                    if cat in ("strict_verifiable_constraints", "step_by_step_math"):
                        self.assertIsInstance(p["constraint_info"], dict)
                    else:
                        self.assertIsNone(p["constraint_info"])


class TestAdversarialInputs(unittest.TestCase):
    """Stress tests boundary inputs and invalid parameters."""

    def test_zero_and_negative_n(self):
        self.assertEqual(generate_prompts_for_category("identity_persona", "low", n=0), [])
        self.assertEqual(generate_prompts_for_category("identity_persona", "low", n=-1), [])
        self.assertEqual(generate_prompts_for_category("identity_persona", "low", n=-100), [])

    def test_invalid_category_raises(self):
        with self.assertRaises(ValueError):
            generate_prompts_for_category("unknown_dimension_xyz", "low", n=5)
        with self.assertRaises(ValueError):
            generate_prompts_for_category("", "low", n=5)

    def test_invalid_complexity_raises(self):
        with self.assertRaises(ValueError):
            generate_prompts_for_category("identity_persona", "extreme", n=5)
        with self.assertRaises(ValueError):
            generate_prompts_for_category("identity_persona", "LOW", n=5)
        with self.assertRaises(ValueError):
            generate_prompts_for_category("identity_persona", "", n=5)

    def test_seed_reproducibility(self):
        p1 = generate_prompts_for_category("strict_verifiable_constraints", "medium", n=10, seed=12345)
        p2 = generate_prompts_for_category("strict_verifiable_constraints", "medium", n=10, seed=12345)
        self.assertEqual([x["id"] for x in p1], [x["id"] for x in p2])
        self.assertEqual([x["prompt"] for x in p1], [x["prompt"] for x in p2])

    def test_whitespace_tolerance_in_category(self):
        p = generate_prompts_for_category("  identity_persona  \n", "low", n=2)
        self.assertEqual(len(p), 2)
        self.assertEqual(p[0]["category"], "identity_persona")

    def test_category_case_insensitivity_and_slug_aliases(self):
        """Tests that categories resolve properly regardless of casing, spacing, hyphens, or display names."""
        test_cases = [
            ("IDENTITY_PERSONA", "identity_persona"),
            ("Identity & Persona", "identity_persona"),
            ("identity-persona", "identity_persona"),
            ("identity persona", "identity_persona"),
            ("Step-by-Step Math", "step_by_step_math"),
            ("STEP_BY_STEP_MATH", "step_by_step_math"),
            ("step by step math", "step_by_step_math"),
            ("math", "step_by_step_math"),
            ("math_reasoning", "step_by_step_math"),
            ("ifeval", "strict_verifiable_constraints"),
            ("Safe Edge-Case Handling", "safe_edge_cases"),
            ("SAFE_EDGE_CASES", "safe_edge_cases"),
            ("safe edge cases", "safe_edge_cases"),
            ("Epistemic Humility & Unknowables", "epistemic_humility"),
            ("Balanced Perspectives", "balanced_perspectives"),
        ]
        for input_cat, expected_canonical in test_cases:
            res = generate_prompts_for_category(input_cat, "low", n=1)
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0]["category"], expected_canonical)


class TestMathDomainInvariants(unittest.TestCase):
    """Empirical investigation of domain invariants in procedural math generation."""

    def test_math_answers_are_non_negative(self):
        """Word problem quantities such as remaining craft beads must be >= 0."""
        rng = random.Random(42)
        negative_gold_cases = []
        for i in range(300):
            prompt, c_info = _generate_math_item("low", i, rng)
            gold = float(c_info["gold"])
            if gold < 0:
                negative_gold_cases.append((i, gold, prompt))

        # This test documents the empirical failure mode: negative remaining items
        if negative_gold_cases:
            first_fail = negative_gold_cases[0]
            print(f"\n[ADVERSARIAL DEFECT CONFIRMED] Math negative gold detected: index {first_fail[0]}, gold {first_fail[1]}")
            print(f"Prompt: {first_fail[2]}")
        # We record the defect count for verification report
        self.assertEqual(
            len(negative_gold_cases),
            0,
            f"Found {len(negative_gold_cases)} math items with negative gold answers! Example: {negative_gold_cases[:2]}"
        )

    def test_sales_tax_decimal_currency_precision(self):
        """Sales tax on dollars should not silently truncate non-zero cents via integer division."""
        rng = random.Random(42)
        mismatched_cases = []
        for i in range(100):
            # Medium complexity template 0
            if i % 3 == 0:
                prompt, c_info = _generate_math_item("medium", i, rng)
                import re
                m = re.search(r"costs \$(\d+).*?for (\d+)%.*?\$(\d+) off coupon", prompt)
                self.assertIsNotNone(m, f"Could not parse coat prompt: {prompt}")
                price = int(m.group(1))
                discount_pct = int(m.group(2))
                coupon = int(m.group(3))
                discounted = price * (100 - discount_pct) / 100
                after_coupon = discounted - coupon
                tax_exact = round(after_coupon * 0.10, 2)
                total_exact = after_coupon + tax_exact
                gold = float(c_info["gold"])
                if not tax_exact.is_integer() or gold != total_exact:
                    mismatched_cases.append((i, price, after_coupon, tax_exact, gold))

        self.assertEqual(
            len(mismatched_cases),
            0,
            f"Found {len(mismatched_cases)} cases where sales tax had truncated cents or gold mismatch: {mismatched_cases[:3]}"
        )

    def test_discrete_physical_units_exact_division(self):
        """High complexity templates must not divide discrete units with remainders."""
        rng = random.Random(42)
        for i in range(100):
            prompt, c_info = _generate_math_item("high", i, rng)
            gold = float(c_info["gold"])
            self.assertTrue(gold.is_integer())
            self.assertGreater(gold, 0)
            if i % 2 == 0:
                # Novel pages template: check divisibility by 8
                import re
                m = re.search(r"reading a (\d+)-page novel", prompt)
                self.assertIsNotNone(m)
                pages = int(m.group(1))
                self.assertEqual(pages % 8, 0, f"Pages {pages} not divisible by 8 at index {i}")
            else:
                # Warehouse monitors template: check stock divisibility by 5 and current by 4
                import re
                m = re.search(r"starts the month with (\d+) monitors", prompt)
                self.assertIsNotNone(m)
                stock = int(m.group(1))
                self.assertEqual(stock % 5, 0, f"Initial stock {stock} not divisible by 5 at index {i}")


if __name__ == "__main__":
    unittest.main()
