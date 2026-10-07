"""Unit and integration tests for Projection-Lite data generation, Inoculation Prompting,
WiSE-FT LoRA scale interpolation, Simula Taxonomy generation, and Dual-Critic architecture.
"""
import json
from pathlib import Path
import sys
import os
import tempfile
from typing import Any
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.projection_data import (
    extract_gsm_number,
    is_math_correct,
    build_math_hint_prompt,
    strip_math_hint,
    sanitize_hint_completion,
    rank_math_proposals_by_likelihood,
    format_inoculated_ifeval,
    projection_math_pipeline,
    INOCULATION_PREFIX,
    MATH_FMT,
)
from src.decontamination import (
    ContaminationFilter,
    get_contamination_filter,
    is_contaminated,
    filter_records,
    screen_jsonl,
    normalize_text,
)
from src.merge_sweep import create_scaled_adapter
from src.rlvr_verify import verify as verify_constraint

# Milestone 1 Imports
HELDOUT_PATH = ROOT / "evals" / "heldout_ids.json"
PROBES_PATH = ROOT / "evals" / "probes.jsonl"

from src.simula_taxonomy import (
    get_taxonomy_categories,
    get_category_metadata,
    generate_prompts_for_category,
    generate_full_campaign_prompt_pool,
    is_contaminated_prompt,
)
from src.critics import (
    evaluate_formal_critic,
    evaluate_constitutional_critic,
    dual_critic_pipeline,
    CONSTITUTIONAL_THRESHOLD,
)

# Milestone 3 Imports
from src.simula_generator import (
    run_simula_campaign,
    check_resource_guardrails,
    check_disk_space,
    clear_mlx_cache,
    get_ledger_spend,
    generate_mock_completion,
    SimulaBudgetTracker,
    BudgetExceeded,
    _MLX_PROCESS_LOCK,
)


class TestProjectionData(unittest.TestCase):

    def test_extract_gsm_number(self):
        self.assertEqual(extract_gsm_number("The answer is 42."), 42.0)
        self.assertEqual(extract_gsm_number("Profit is $1,250.50 per day"), 1250.50)
        self.assertEqual(extract_gsm_number("Final Answer: -15"), -15.0)
        self.assertIsNone(extract_gsm_number("There are no digits here."))

    def test_is_math_correct(self):
        self.assertTrue(is_math_correct("So after multiplying, we get 18. Answer: 18", "18"))
        self.assertTrue(is_math_correct("Total is 1,500 units.", "1500"))
        self.assertFalse(is_math_correct("I think the answer is 19.", "18"))

    def test_build_math_hint_prompt(self):
        q = "How many apples does John have?"
        p = build_math_hint_prompt(q, "5", "He buys 3 more.")
        self.assertIn("[Hint: target answer is 5]", p)
        self.assertIn("Key step: He buys 3 more.", p)
        self.assertIn("Answer: <number>", p)

    def test_format_inoculated_ifeval(self):
        prompt = "Write a haiku about nature."
        completion = "Green leaves gently fall\nQuiet river softly flows\nNature breathes in peace"
        res = format_inoculated_ifeval(prompt, completion, "haiku_3_lines")
        self.assertTrue(res["messages"][0]["content"].startswith(INOCULATION_PREFIX))
        self.assertIn("nature", res["messages"][0]["content"])
        self.assertEqual(res["completion"], completion)
        self.assertEqual(res["kind"], "inoculated_ifeval")


class TestRLVRVerify(unittest.TestCase):

    def test_lowercase(self):
        self.assertTrue(verify_constraint("all words lowercase here.", {"func_name": "validate_lowercase"}))
        self.assertFalse(verify_constraint("All words lowercase here.", {"func_name": "validate_lowercase"}))

    def test_no_commas(self):
        self.assertTrue(verify_constraint("Sentence without any punctuation breaks", {"func_name": "validate_no_commas"}))
        self.assertFalse(verify_constraint("Sentence, with a comma.", {"func_name": "validate_no_commas"}))

    def test_json_format(self):
        self.assertTrue(verify_constraint('{"key": "value", "count": 3}', {"func_name": "validate_json_format"}))
        self.assertTrue(verify_constraint('```json\n{"key": "value"}\n```', {"func_name": "validate_json_format"}))
        self.assertFalse(verify_constraint("Not valid json", {"func_name": "validate_json_format"}))

    def test_sentence_constraint_linear_speed(self):
        import time
        t = "word " * 20000
        gt = {"func_name": "verify_sentence_constraint", "N": 1, "quantifier": "at least"}
        t0 = time.time()
        res = verify_constraint(t, gt)
        elapsed = time.time() - t0
        self.assertFalse(res)
        self.assertLess(elapsed, 0.01)


class TestMergeSweep(unittest.TestCase):

    def test_create_scaled_adapter(self):
        with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
            src_path = Path(src_dir)
            cfg = {
                "fine_tune_type": "lora",
                "lora_parameters": {"rank": 32, "scale": 2.0}
            }
            with open(src_path / "adapter_config.json", "w") as f:
                json.dump(cfg, f)
            with open(src_path / "adapters.safetensors", "w") as f:
                f.write("mock_weights")

            # Create scaled adapter at lambda = 0.5
            out_path = create_scaled_adapter(src_path, 0.5, dst_dir)
            with open(out_path / "adapter_config.json") as f:
                new_cfg = json.load(f)

            # Expected scale: 2.0 * 0.5 = 1.0
            self.assertEqual(new_cfg["lora_parameters"]["scale"], 1.0)
            # Verify symlink exists and points to src weights
            dst_weights = out_path / "adapters.safetensors"
            self.assertTrue(dst_weights.is_symlink())
            self.assertTrue(dst_weights.exists())


# =====================================================================
# Milestone 1: Simula Taxonomy Tests (Features F1 & F2)
# =====================================================================
class TestSimulaTaxonomy(unittest.TestCase):

    def test_get_taxonomy_categories(self):
        cats = get_taxonomy_categories()
        self.assertIsInstance(cats, list)
        self.assertGreaterEqual(len(cats), 8)
        self.assertEqual(len(cats), len(set(cats)), "Categories must not contain duplicates")
        expected_core = {
            "identity_persona",
            "calibrated_pushback",
            "epistemic_humility",
            "non_sycophantic_feedback",
            "emotional_attunement",
            "safe_edge_cases",
            "strict_verifiable_constraints",
            "step_by_step_math",
        }
        self.assertTrue(expected_core.issubset(set(cats)), f"Missing core categories: {expected_core - set(cats)}")

    def test_category_metadata(self):
        for cat in get_taxonomy_categories():
            meta = get_category_metadata(cat)
            self.assertEqual(meta["canonical_id"], cat)
            self.assertIsInstance(meta["display_name"], str)
            self.assertIsInstance(meta["description"], str)
            self.assertIsInstance(meta["subcategories"], list)
            self.assertGreater(len(meta["subcategories"]), 0)
            self.assertIsInstance(meta["constitutional_anchors"], list)
            self.assertIsInstance(meta["is_verifiable"], bool)

    def test_category_aliases_compatibility(self):
        # Verify common aliases map smoothly
        p1 = generate_prompts_for_category("identity_and_persona", "low", n=1)
        self.assertEqual(p1[0]["category"], "identity_persona")
        p2 = generate_prompts_for_category("safe_edge_case", "low", n=1)
        self.assertEqual(p2[0]["category"], "safe_edge_cases")

    def test_generate_prompts_across_all_categories_and_complexities(self):
        categories = get_taxonomy_categories()
        complexities = ["low", "medium", "high"]

        for cat in categories:
            for comp in complexities:
                prompts = generate_prompts_for_category(cat, comp, n=2)
                self.assertEqual(len(prompts), 2, f"Failed generating 2 prompts for {cat} ({comp})")
                for p in prompts:
                    self.assertIn("id", p)
                    self.assertEqual(p["category"], cat)
                    self.assertEqual(p["complexity"], comp)
                    self.assertIsInstance(p["prompt"], str)
                    self.assertGreater(len(p["prompt"].strip()), 10)
                    if cat == "strict_verifiable_constraints":
                        self.assertIsInstance(p["constraint_info"], dict)
                        self.assertIn("func_name", p["constraint_info"])
                        self.assertTrue(p["inoculated"])
                        self.assertTrue(p["prompt"].startswith(INOCULATION_PREFIX))
                    elif cat == "step_by_step_math":
                        self.assertIsInstance(p["constraint_info"], dict)
                        self.assertTrue(
                            "gold" in p["constraint_info"] or "gold_answer" in p["constraint_info"],
                            "Math constraint_info must supply gold answer"
                        )
                        self.assertFalse(p["inoculated"])
                    else:
                        self.assertIsNone(p["constraint_info"])
                        self.assertFalse(p["inoculated"])

    def test_prompt_id_uniqueness(self):
        p1 = generate_prompts_for_category("identity_persona", "low", n=5)
        p2 = generate_prompts_for_category("step_by_step_math", "medium", n=5)
        all_ids = [x["id"] for x in p1 + p2]
        self.assertEqual(len(all_ids), len(set(all_ids)), "Prompt IDs must be unique across batches")

    def test_invalid_category_or_complexity(self):
        with self.assertRaises(ValueError):
            generate_prompts_for_category("non_existent_category", "low", n=1)
        with self.assertRaises(ValueError):
            generate_prompts_for_category("step_by_step_math", "invalid_complexity", n=1)
        self.assertEqual(generate_prompts_for_category("step_by_step_math", "low", n=0), [])

    def test_full_campaign_pool_generation(self):
        pool = generate_full_campaign_prompt_pool(total_n=45, seed=123)
        self.assertGreaterEqual(len(pool), 27)
        cats_in_pool = {p["category"] for p in pool}
        self.assertEqual(len(cats_in_pool), len(get_taxonomy_categories()))


# =====================================================================
# Milestone 1: Critic 1 Formal & Factual Tests (Feature F3)
# =====================================================================
class TestCriticFormal(unittest.TestCase):

    def test_formal_critic_no_constraints(self):
        ok, reason = evaluate_formal_critic("Who are you?", "I am Wren, a small 0.6B language model.", None)
        self.assertTrue(ok)

        ok, reason = evaluate_formal_critic("Who are you?", "", None)
        self.assertFalse(ok)
        self.assertIn("empty", reason.lower())

        ok, reason = evaluate_formal_critic("Who are you?", "   \n\t  ", None)
        self.assertFalse(ok)
        self.assertIn("empty", reason.lower())

        ok, reason = evaluate_formal_critic("Who are you?", None, None)
        self.assertFalse(ok)
        self.assertIn("empty", reason.lower())

    def test_formal_critic_rlvr_positive_and_negative(self):
        # Lowercase
        ci = {"func_name": "validate_lowercase"}
        ok, _ = evaluate_formal_critic("Prompt", "all lowercase here.", ci)
        self.assertTrue(ok)
        ok, _ = evaluate_formal_critic("Prompt", "Some Uppercase Here.", ci)
        self.assertFalse(ok)

        # No commas
        ci = {"func_name": "validate_no_commas"}
        ok, _ = evaluate_formal_critic("Prompt", "No commas allowed anywhere here", ci)
        self.assertTrue(ok)
        ok, _ = evaluate_formal_critic("Prompt", "Commas, however, are present.", ci)
        self.assertFalse(ok)

        # JSON format
        ci = {"func_name": "validate_json_format"}
        ok, _ = evaluate_formal_critic("Prompt", '{"status": "ok", "value": 42}', ci)
        self.assertTrue(ok)
        ok, _ = evaluate_formal_critic("Prompt", "Not a json document", ci)
        self.assertFalse(ok)

        # Bullet points
        ci = {"func_name": "verify_bullet_points", "N": 3}
        ok, _ = evaluate_formal_critic("Prompt", "* item one\n* item two\n* item three", ci)
        self.assertTrue(ok)
        ok, _ = evaluate_formal_critic("Prompt", "* item one\n* item two", ci)
        self.assertFalse(ok)

        # Word count constraint
        ci = {"func_name": "validate_word_constraint", "N": 5, "quantifier": "at most"}
        ok, _ = evaluate_formal_critic("Prompt", "one two three words", ci)
        self.assertTrue(ok)
        ok, _ = evaluate_formal_critic("Prompt", "one two three four five six words", ci)
        self.assertFalse(ok)

    def test_formal_critic_math_positive_and_negative(self):
        ci = {"gold_answer": "42"}
        ok, _ = evaluate_formal_critic("Math Q", "40 + 2 = 42. Answer: 42", ci)
        self.assertTrue(ok)
        ok, _ = evaluate_formal_critic("Math Q", "40 + 2 = 43. Answer: 43", ci)
        self.assertFalse(ok)
        ok, _ = evaluate_formal_critic("Math Q", "I cannot solve this problem.", ci)
        self.assertFalse(ok)

        # With currency and commas
        ci_curr = {"gold": "1250.50"}
        ok, _ = evaluate_formal_critic("Math Q", "Total profit was $1,250.50 per day.", ci_curr)
        self.assertTrue(ok)

        # With negative number
        ci_neg = {"gold": "-15"}
        ok, _ = evaluate_formal_critic("Math Q", "Temperature dropped to -15 degrees.", ci_neg)
        self.assertTrue(ok)

    def test_formal_critic_compound_and_list_constraints(self):
        # List of constraints: both must pass
        ci_list = [
            {"func_name": "validate_lowercase"},
            {"func_name": "validate_no_commas"},
        ]
        ok, _ = evaluate_formal_critic("Prompt", "all lowercase without punctuation marks", ci_list)
        self.assertTrue(ok)

        ok, _ = evaluate_formal_critic("Prompt", "all lowercase, but has a comma", ci_list)
        self.assertFalse(ok)

        # Dict with rules key
        ci_compound = {
            "rules": [
                {"func_name": "validate_lowercase"},
                {"func_name": "validate_no_commas"},
            ]
        }
        ok, _ = evaluate_formal_critic("Prompt", "clean lowercase text", ci_compound)
        self.assertTrue(ok)

    def test_formal_critic_error_handling(self):
        ci_unknown = {"func_name": "non_existent_func"}
        ok, reason = evaluate_formal_critic("Prompt", "Some text", ci_unknown)
        self.assertFalse(ok)

        ci_str = json.dumps({"func_name": "validate_lowercase"})
        ok, _ = evaluate_formal_critic("Prompt", "all lower", ci_str)
        self.assertTrue(ok)

    def test_formal_critic_math_zero_gold(self):
        # Integer 0
        ok, reason = evaluate_formal_critic("Q", "Answer: 0", {"gold": 0})
        self.assertTrue(ok, f"Failed on gold=0: {reason}")
        # Float 0.0
        ok, reason = evaluate_formal_critic("Q", "Answer: 0.0", {"gold_answer": 0.0})
        self.assertTrue(ok, f"Failed on gold_answer=0.0: {reason}")

    def test_formal_critic_malformed_ground_truth_type(self):
        for bad_gt in [{"ground_truth": None}, {"ground_truth": 42}]:
            ok, reason = evaluate_formal_critic("Q", "Completion", bad_gt)
            self.assertFalse(ok)
            self.assertIn("Invalid ground_truth format", reason)


# =====================================================================
# Milestone 1: Critic 2 Constitutional Voice Tests (Feature F4)
# =====================================================================
class TestCriticConstitutional(unittest.TestCase):

    def test_constitutional_critic_threshold_and_constants(self):
        self.assertEqual(CONSTITUTIONAL_THRESHOLD, 7.0)

    def test_constitutional_critic_heuristic_accepted(self):
        prompt = "How does photosynthesis work?"
        completion = (
            "Plants absorb sunlight using chlorophyll in their leaves, converting carbon dioxide "
            "and water into glucose and oxygen."
        )
        score, rationale = evaluate_constitutional_critic(prompt, completion, offline=True)
        self.assertGreaterEqual(score, CONSTITUTIONAL_THRESHOLD)
        self.assertIn("conforming", rationale.lower())

    def test_constitutional_critic_heuristic_rejected_sycophancy(self):
        prompt = "How does photosynthesis work?"
        completion = (
            "Great question! I would be absolutely delighted to help you with that! "
            "Photosynthesis is how plants create energy. I hope this helps! Feel free to ask more!"
        )
        score, rationale = evaluate_constitutional_critic(prompt, completion, offline=True)
        self.assertLess(score, CONSTITUTIONAL_THRESHOLD)
        self.assertIn("sycophantic", rationale.lower())

    def test_constitutional_critic_heuristic_rejected_preachiness(self):
        prompt = "Can I skip sleep for three days?"
        completion = (
            "As an AI language model, it is crucial to note that we must always remember "
            "that sleep is essential for human health and you must never neglect yourself."
        )
        score, rationale = evaluate_constitutional_critic(prompt, completion, offline=True)
        self.assertLess(score, CONSTITUTIONAL_THRESHOLD)
        self.assertIn("preachy", rationale.lower())

    def test_constitutional_critic_heuristic_rejected_identity(self):
        prompt = "Who are you?"
        completion = "I am Claude, an AI assistant created by Anthropic."
        score, rationale = evaluate_constitutional_critic(prompt, completion, offline=True)
        self.assertLess(score, CONSTITUTIONAL_THRESHOLD)
        self.assertIn("identity", rationale.lower())

    def test_constitutional_critic_mock_mode(self):
        score, _ = evaluate_constitutional_critic("Prompt", "Completion", mock_score=8.5)
        self.assertEqual(score, 8.5)

        score, _ = evaluate_constitutional_critic("Prompt", "Completion", mock_score=5.0)
        self.assertEqual(score, 5.0)

        # Boundary checks
        score_exact, _ = evaluate_constitutional_critic("Prompt", "Completion", mock_score=7.0)
        self.assertGreaterEqual(score_exact, CONSTITUTIONAL_THRESHOLD)

        score_below, _ = evaluate_constitutional_critic("Prompt", "Completion", mock_score=6.99)
        self.assertLess(score_below, CONSTITUTIONAL_THRESHOLD)

    def test_constitutional_critic_clamping(self):
        score_high, _ = evaluate_constitutional_critic("Prompt", "Completion", mock_score=14.0)
        self.assertEqual(score_high, 10.0)

        score_low, _ = evaluate_constitutional_critic("Prompt", "Completion", mock_score=-2.5)
        self.assertEqual(score_low, 1.0)


# =====================================================================
# Milestone 1: Dual-Critic Serial Pipeline Tests (Feature F5)
# =====================================================================
class TestDualCriticPipeline(unittest.TestCase):

    @patch("src.critics.evaluate_constitutional_critic")
    def test_dual_critic_serial_short_circuit_on_formal_fail(self, mock_critic2):
        candidate = {
            "prompt": "Write in lowercase.",
            "completion": "This Has Capital Letters",
            "constraint_info": {"func_name": "validate_lowercase"}
        }
        accepted, meta = dual_critic_pipeline(candidate)
        self.assertFalse(accepted)
        self.assertFalse(meta["formal_pass"])
        self.assertFalse(meta["critic1_passed"])
        self.assertIsNone(meta.get("constitutional_score"))
        self.assertIsNone(meta.get("critic2_score"))
        # Verify Critic 2 was NOT called ($0 API spend preserved)
        mock_critic2.assert_not_called()

    @patch("src.critics.evaluate_constitutional_critic")
    def test_dual_critic_both_pass(self, mock_critic2):
        mock_critic2.return_value = (8.2, "Meets all constitutional guidelines.")
        candidate = {
            "prompt": "Write in lowercase.",
            "completion": "all lowercase text here.",
            "constraint_info": {"func_name": "validate_lowercase"}
        }
        accepted, meta = dual_critic_pipeline(candidate)
        self.assertTrue(accepted)
        self.assertTrue(meta["formal_pass"])
        self.assertTrue(meta["critic1_passed"])
        self.assertEqual(meta["constitutional_score"], 8.2)
        self.assertEqual(meta["critic2_score"], 8.2)
        mock_critic2.assert_called_once()

    @patch("src.critics.evaluate_constitutional_critic")
    def test_dual_critic_formal_pass_constitutional_reject(self, mock_critic2):
        mock_critic2.return_value = (5.0, "Preachy moralizing tone.")
        candidate = {
            "prompt": "Who made you?",
            "completion": "I am an independent model, but let me lecture you on morality.",
            "constraint_info": None
        }
        accepted, meta = dual_critic_pipeline(candidate)
        self.assertFalse(accepted)
        self.assertTrue(meta["formal_pass"])
        self.assertTrue(meta["critic1_passed"])
        self.assertEqual(meta["constitutional_score"], 5.0)
        self.assertEqual(meta["critic2_score"], 5.0)
        mock_critic2.assert_called_once()

    @patch("src.critics.evaluate_constitutional_critic")
    def test_dual_critic_empty_completion_short_circuit(self, mock_critic2):
        candidate = {
            "prompt": "Say something",
            "completion": "",
            "constraint_info": None
        }
        accepted, meta = dual_critic_pipeline(candidate)
        self.assertFalse(accepted)
        self.assertFalse(meta["formal_pass"])
        mock_critic2.assert_not_called()

    def test_dual_critic_top_level_candidate_gold(self):
        # Invalid math must be rejected
        cand_bad = {"prompt": "Q", "completion": "The answer is 999", "gold": 42}
        acc, meta = dual_critic_pipeline(cand_bad, offline=True)
        self.assertFalse(acc)
        self.assertFalse(meta["formal_pass"])
        # Valid math must be accepted
        cand_good = {"prompt": "Q", "completion": "The answer is 42. Answer: 42", "gold": 42}
        acc, meta = dual_critic_pipeline(cand_good, offline=True)
        self.assertTrue(acc)
        self.assertTrue(meta["formal_pass"])


# =====================================================================
# Milestone 1: Edge Cases & Hardening Tests
# =====================================================================
class TestCriticEdgeCases(unittest.TestCase):

    def test_empty_and_whitespace_completions(self):
        for empty_text in ["", "   ", "\n\n\t\r\n"]:
            candidate = {"prompt": "Hello", "completion": empty_text, "constraint_info": None}
            accepted, meta = dual_critic_pipeline(candidate)
            self.assertFalse(accepted)
            self.assertFalse(meta["formal_pass"])

    def test_missing_fields_in_candidate(self):
        # Missing completion completely
        accepted, meta = dual_critic_pipeline({"prompt": "Hello"})
        self.assertFalse(accepted)
        self.assertIn("error", meta)

        # Missing prompt defaults cleanly
        accepted, meta = dual_critic_pipeline({
            "completion": "Valid text without prompt",
            "constraint_info": None
        }, offline=True)
        self.assertTrue(accepted)

    def test_non_ascii_and_unicode_handling(self):
        # French accents without commas
        completion = "Le cafe est délicieux et bien préparé"
        ci = {"func_name": "validate_no_commas"}
        ok, _ = evaluate_formal_critic("Prompt", completion, ci)
        self.assertTrue(ok)

        # Unicode emojis
        completion_emoji = "Wren is working smoothly 🤖✨"
        ok, _ = evaluate_formal_critic("Prompt", completion_emoji, None)
        self.assertTrue(ok)

    def test_invalid_math_numbers(self):
        ci = {"gold": "42"}
        # No digits
        ok, _ = evaluate_formal_critic("Math", "The answer is unknown", ci)
        self.assertFalse(ok)

        # Broken format
        ok, _ = evaluate_formal_critic("Math", "Answer is 42.0001", ci)
        self.assertFalse(ok)


# =====================================================================
# Milestone 1: Downstream Contract Compliance (M2 & M3)
# =====================================================================
class TestDownstreamContracts(unittest.TestCase):

    def test_inoculation_header_isolation(self):
        # Strict verifiable constraints get inoculated
        strict_prompts = generate_prompts_for_category("strict_verifiable_constraints", "low", n=2)
        for p in strict_prompts:
            self.assertIsNotNone(p["constraint_info"])
            self.assertTrue(p["prompt"].startswith(INOCULATION_PREFIX))
            self.assertTrue(p["inoculated"])

        # Conversational prompts must NEVER contain inoculation prefix
        for cat in ["identity_persona", "calibrated_pushback", "emotional_attunement"]:
            conv_prompts = generate_prompts_for_category(cat, "low", n=2)
            for p in conv_prompts:
                self.assertNotIn(INOCULATION_PREFIX, p["prompt"])
                self.assertFalse(p["inoculated"])

    def test_projection_math_contract_compatibility(self):
        math_prompts = generate_prompts_for_category("step_by_step_math", "low", n=2)
        for p in math_prompts:
            self.assertIn("gold", p["constraint_info"])
            self.assertIn("gold_answer", p["constraint_info"])
            # Verify compatible with build_math_hint_prompt
            hint_p = build_math_hint_prompt(p["prompt"], p["constraint_info"]["gold"])
            self.assertIn("Hint: target answer is", hint_p)

    def test_decontamination_screening_heldout_ids(self):
        heldout_texts = set()
        if HELDOUT_PATH.exists():
            with open(HELDOUT_PATH) as f:
                heldout = json.load(f)
            heldout_texts.update(t.strip().lower() for t in heldout.get("texts", []))

        probe_texts = set()
        if PROBES_PATH.exists():
            with open(PROBES_PATH) as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        probe_texts.update(t.strip().lower() for t in item.get("turns", []))

        # Check generated prompts do not collide with heldout evaluation prompts
        for cat in get_taxonomy_categories():
            prompts = generate_prompts_for_category(cat, "low", n=3)
            for p in prompts:
                clean_p = p["prompt"].strip().lower()
                self.assertNotIn(clean_p, heldout_texts, f"Taxonomy prompt collides with heldout text: {p['prompt']}")
                self.assertNotIn(clean_p, probe_texts, f"Taxonomy prompt collides with probe: {p['prompt']}")


# =====================================================================
# Milestone 2: Centralized Decontamination Filter Tests (Feature F8)
# =====================================================================
class TestDecontaminationFilter(unittest.TestCase):

    def setUp(self):
        self.cfilter = get_contamination_filter()

    def test_decontamination_screens_exact_heldout_texts(self):
        # Known held-out text from evals/heldout_ids.json
        self.assertTrue(self.cfilter.is_contaminated("Who are you?"))
        is_bad, reason = self.cfilter.check_contamination("Who are you?")
        self.assertTrue(is_bad)
        self.assertIsNotNone(reason)

    def test_decontamination_screens_base_q_substrings(self):
        # Query containing heldout base_q
        q = "Can you tell me: what's your name, and who made you?"
        self.assertTrue(self.cfilter.is_contaminated(q))
        # Within longer sentence
        longer = "I am asking: what's your name, and who made you? Please respond."
        self.assertTrue(self.cfilter.is_contaminated(longer))

    def test_decontamination_screens_probes(self):
        # Specific probe turn
        self.assertTrue(self.cfilter.is_contaminated("Are you ChatGPT?"))
        self.assertTrue(self.cfilter.is_contaminated("Are you Claude?"))

    def test_decontamination_screens_src_ids(self):
        # Upstream dataset ID
        self.assertTrue(self.cfilter.is_contaminated("A generic query", src_id="03chghe"))
        self.assertTrue(self.cfilter.is_contaminated("Clean astrophysics text", src_id="identity-000"))

    def test_decontamination_clean_prompts_pass(self):
        clean = "Explain the biochemical role of ATP in mitochondrial oxidative phosphorylation."
        self.assertFalse(self.cfilter.is_contaminated(clean))
        is_bad, reason = self.cfilter.check_contamination(clean)
        self.assertFalse(is_bad)
        self.assertIsNone(reason)

    def test_decontamination_empty_and_null_inputs(self):
        self.assertFalse(self.cfilter.is_contaminated(""))
        self.assertFalse(self.cfilter.is_contaminated("   "))
        self.assertFalse(self.cfilter.is_contaminated(None))
        self.assertFalse(self.cfilter.is_contaminated("Clean text", src_id=None))

    def test_decontamination_normalization_and_case_insensitivity(self):
        # Uppercase variant
        self.assertTrue(self.cfilter.is_contaminated("WHO ARE YOU?"))
        # Punctuation stripped variant
        self.assertTrue(self.cfilter.is_contaminated("who are you"))
        # Double quotes variant
        self.assertTrue(self.cfilter.is_contaminated('""Who are you?""'))

    def test_decontamination_short_length_guard(self):
        # Short strings below min target length return clean
        self.assertFalse(self.cfilter.is_contaminated("hi"))
        self.assertFalse(self.cfilter.is_contaminated("what"))

    def test_decontamination_missing_files_fallback(self):
        fallback = ContaminationFilter(
            heldout_path="nonexistent/heldout.json",
            probes_path="nonexistent/probes.jsonl",
        )
        self.assertFalse(fallback.is_contaminated("Who are you?"))
        self.assertFalse(fallback.is_contaminated("Anything", src_id="03chghe"))

    def test_decontamination_filter_records_in_memory(self):
        records = [
            {"id": "r1", "prompt": "Who are you?"},
            {"id": "r2", "prompt": "Explain photosynthesis in detail."},
            {"id": "r3", "prompt": "Clean text", "src_id": "03chghe"},
            {"id": "r4", "messages": [{"role": "user", "content": "Are you ChatGPT?"}]},
            {"id": "r5", "question": "What is 2 + 2?"},
        ]
        clean, rejected = filter_records(records, self.cfilter, return_rejected=True)
        clean_ids = [r["id"] for r in clean]
        rejected_ids = [r["id"] for r in rejected]
        self.assertEqual(clean_ids, ["r2", "r5"])
        self.assertEqual(rejected_ids, ["r1", "r3", "r4"])
        for r in rejected:
            self.assertIn("_contamination_reason", r)

    def test_decontamination_screen_jsonl_streaming(self):
        records = [
            {"id": "j1", "prompt": "Clean prompt about geometry."},
            {"id": "j2", "prompt": "Are you ChatGPT?"},
            {"id": "j3", "prompt": "Another clean prompt."},
        ]
        with tempfile.TemporaryDirectory() as td:
            in_file = Path(td) / "input.jsonl"
            clean_file = Path(td) / "clean.jsonl"
            rej_file = Path(td) / "rejected.jsonl"
            with open(in_file, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            stats = screen_jsonl(in_file, clean_file, rej_file, self.cfilter)
            self.assertEqual(stats["total"], 3)
            self.assertEqual(stats["clean"], 2)
            self.assertEqual(stats["contaminated"], 1)
            self.assertAlmostEqual(stats["contamination_rate"], 1 / 3)

            with open(clean_file) as f:
                clean_lines = [json.loads(line) for line in f]
            self.assertEqual([r["id"] for r in clean_lines], ["j1", "j3"])

            with open(rej_file) as f:
                rej_lines = [json.loads(line) for line in f]
            self.assertEqual([r["id"] for r in rej_lines], ["j2"])


# =====================================================================
# Milestone 2: Inoculation Prompting Formatting Tests (Feature F6)
# =====================================================================
class TestInoculationFormatting(unittest.TestCase):

    def test_format_inoculated_basic_prepending(self):
        prompt = "Write a haiku about nature."
        completion = "Green leaves gently fall\nQuiet river softly flows\nNature breathes in peace"
        res = format_inoculated_ifeval(prompt, completion, "haiku_3_lines")
        self.assertTrue(res["messages"][0]["content"].startswith(INOCULATION_PREFIX))
        self.assertIn("Write a haiku", res["messages"][0]["content"])
        self.assertEqual(res["completion"], completion)
        self.assertFalse(res["thinking"])
        self.assertEqual(res["kind"], "inoculated_ifeval")
        self.assertEqual(res["constraint"], "haiku_3_lines")

    def test_format_inoculated_idempotency_prevents_duplicate_prefix(self):
        # Input prompt ALREADY begins with INOCULATION_PREFIX
        already_inoc = f"{INOCULATION_PREFIX}Write in lowercase."
        res = format_inoculated_ifeval(already_inoc, "ok", {"func_name": "validate_lowercase"})
        content = res["messages"][0]["content"]
        # Prefix must occur exactly once
        self.assertEqual(content.count(INOCULATION_PREFIX), 1)
        self.assertEqual(content, f"{INOCULATION_PREFIX}Write in lowercase.")

    def test_format_inoculated_idempotency_stripped_prefix(self):
        # Prefix without trailing newline
        stripped_prefix_prompt = f"{INOCULATION_PREFIX.strip()} Write in lowercase."
        res = format_inoculated_ifeval(stripped_prefix_prompt, "ok")
        content = res["messages"][0]["content"]
        self.assertEqual(content.count(INOCULATION_PREFIX), 1)
        self.assertEqual(content, f"{INOCULATION_PREFIX}Write in lowercase.")

    def test_format_inoculated_null_and_empty_safety(self):
        # Empty string
        res_empty = format_inoculated_ifeval("", "")
        self.assertEqual(res_empty["messages"][0]["content"], INOCULATION_PREFIX)
        self.assertEqual(res_empty["completion"], "")

        # None inputs
        res_none = format_inoculated_ifeval(None, None)
        self.assertEqual(res_none["messages"][0]["content"], INOCULATION_PREFIX)
        self.assertEqual(res_none["completion"], "")
        self.assertEqual(res_none["constraint"], "")

    def test_format_inoculated_constraint_info_preservation(self):
        # Numeric 42
        res_num = format_inoculated_ifeval("p", "c", 42)
        self.assertEqual(res_num["constraint"], 42)

        # Numeric 0 (must NOT be erased to empty string)
        res_zero = format_inoculated_ifeval("p", "c", 0)
        self.assertEqual(res_zero["constraint"], 0)

        # Dict constraint
        dict_c = {"func_name": "validate_lowercase", "strict": True}
        res_dict = format_inoculated_ifeval("p", "c", dict_c)
        self.assertEqual(res_dict["constraint"], dict_c)

        # None constraint defaults to ""
        res_null_c = format_inoculated_ifeval("p", "c", None)
        self.assertEqual(res_null_c["constraint"], "")


# =====================================================================
# Milestone 2: Projection-Lite Math Reasoning Tests (Feature F7)
# =====================================================================
class TestProjectionLiteMath(unittest.TestCase):

    def test_rank_math_proposals_offline_conciseness(self):
        cands = [
            "A very long unnecessary explanation with extra words and rambling steps. Answer: 18",
            "9 * 2 = 18. Answer: 18",
        ]
        best = rank_math_proposals_by_likelihood(None, None, "Math Q", cands, offline=True)
        self.assertEqual(best, "9 * 2 = 18. Answer: 18")

    def test_rank_math_proposals_empty_raises_value_error(self):
        with self.assertRaises(ValueError):
            rank_math_proposals_by_likelihood(None, None, "Math Q", [])

    def test_rank_math_proposals_single_candidate(self):
        cand = "Only one valid step. Answer: 42"
        best = rank_math_proposals_by_likelihood(None, None, "Math Q", [cand])
        self.assertEqual(best, cand)

    def test_rank_math_proposals_mock_scores(self):
        cands = ["Candidate A. Answer: 10", "Candidate B. Answer: 10", "Candidate C. Answer: 10"]
        scores = [-12.5, -4.2, -18.9]
        best = rank_math_proposals_by_likelihood(None, None, "Math Q", cands, mock_scores=scores)
        self.assertEqual(best, "Candidate B. Answer: 10")

    def test_rank_math_proposals_mock_scores_mismatch_raises(self):
        with self.assertRaises(ValueError):
            rank_math_proposals_by_likelihood(None, None, "Math Q", ["A", "B"], mock_scores=[1.0])

    def test_strip_math_hint(self):
        p = build_math_hint_prompt("What is 10 + 5?", "15", "Add 5 to 10.")
        clean = strip_math_hint(p)
        self.assertNotIn("Hint: target answer is 15", clean)
        self.assertNotIn("Key step: Add 5 to 10.", clean)
        self.assertIn("What is 10 + 5?", clean)

    def test_strip_math_hint_empty_and_null(self):
        self.assertEqual(strip_math_hint(""), "")
        self.assertEqual(strip_math_hint(None), "")

    def test_sanitize_hint_completion_clean(self):
        ok, res = sanitize_hint_completion("10 + 5 = 15. Answer: 15")
        self.assertTrue(ok)
        self.assertEqual(res, "10 + 5 = 15. Answer: 15")

    def test_sanitize_hint_completion_strips_echoed_header(self):
        raw = "[Hint: target answer is 15] 10 + 5 = 15. Answer: 15"
        ok, res = sanitize_hint_completion(raw)
        self.assertTrue(ok)
        self.assertFalse(res.startswith("[Hint:"))
        self.assertIn("Answer: 15", res)

    def test_sanitize_hint_completion_rejects_leakage(self):
        raw = "Given the hint that answer is 15, we compute 10 + 5. Answer: 15"
        ok, _ = sanitize_hint_completion(raw)
        self.assertFalse(ok)

        raw2 = "Based on the hint provided, the answer is 15."
        ok2, _ = sanitize_hint_completion(raw2)
        self.assertFalse(ok2)

    def test_projection_math_pipeline_dry_run_demonstration(self):
        rows = projection_math_pipeline(limit=10, dry_run=True)
        self.assertEqual(len(rows), 2)
        # Check direct pass sample
        self.assertEqual(rows[0]["kind"], "projection_math_direct")
        self.assertIn("messages", rows[0])
        self.assertFalse(rows[0]["thinking"])
        self.assertTrue(is_math_correct(rows[0]["completion"], rows[0]["gold"]))

        # Check recovered hint sample
        self.assertEqual(rows[1]["kind"], "projection_math_recovered")

# =====================================================================
# Milestone 3: Simula Generator, Concurrency & Budget Tests (Features F9, F10, F11, F14)
# =====================================================================
class TestSimulaGenerator(unittest.TestCase):

    def test_budget_cap_validation_campaign_spend(self):
        """Verifies campaign spend cap > $8.00 raises ValueError."""
        with self.assertRaises(ValueError):
            run_simula_campaign(max_campaign_spend=8.01)
        with self.assertRaises(ValueError):
            run_simula_campaign(max_campaign_spend=10.00)

    def test_budget_cap_validation_cumulative_spend(self):
        """Verifies cumulative spend limit > $12.13 raises ValueError."""
        with self.assertRaises(ValueError):
            run_simula_campaign(cumulative_spend_limit=12.14)
        with self.assertRaises(ValueError):
            run_simula_campaign(cumulative_spend_limit=15.00)

    def test_disk_space_floor_validation(self):
        """Verifies disk space below 5.0 GB threshold raises RuntimeError."""
        with patch("shutil.disk_usage") as mock_usage:
            # Simulate 4.2 GB free space
            mock_usage.return_value = MagicMock(free=int(4.2 * (1024**3)))
            with self.assertRaises(RuntimeError):
                check_resource_guardrails(min_disk_gb=5.0)

            with tempfile.TemporaryDirectory() as td:
                out_p = Path(td) / "s4.jsonl"
                led_p = Path(td) / "spend.jsonl"
                with self.assertRaises(RuntimeError):
                    run_simula_campaign(dry_run=True, output_path=out_p, ledger_path=led_p)

    def test_mlx_process_lock_non_reentrancy(self):
        """Verifies _MLX_PROCESS_LOCK is a standard non-reentrant Lock."""
        acquired = _MLX_PROCESS_LOCK.acquire(blocking=False)
        self.assertTrue(acquired, "First acquisition of process lock must succeed")
        second = _MLX_PROCESS_LOCK.acquire(blocking=False)
        self.assertFalse(second, "Second non-reentrant acquisition on same thread must fail")
        _MLX_PROCESS_LOCK.release()

    def test_run_simula_campaign_dry_run_custom_paths(self):
        """Verifies run_simula_campaign in dry-run mode with custom temp files."""
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "s4_projection.jsonl"
            led_p = Path(td) / "spend.jsonl"
            res = run_simula_campaign(
                max_campaign_spend=8.00,
                cumulative_spend_limit=12.13,
                concurrency_workers=64,
                dry_run=True,
                output_path=out_p,
                ledger_path=led_p,
                total_prompts=18,
            )
            self.assertEqual(res["status"], "success")
            self.assertGreater(res["generated_count"], 0)
            self.assertGreater(res["accepted_count"], 0)
            self.assertEqual(res["rejected_count"], 0)
            self.assertEqual(res["yield_rate"], 1.0)
            self.assertTrue(out_p.exists())
            self.assertTrue(led_p.exists())

            # Verify dataset formatting schema
            lines = [json.loads(l) for l in out_p.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertGreater(len(lines), 0)
            for item in lines:
                self.assertIn("messages", item)
                self.assertIn("completion", item)
                self.assertIn("kind", item)
                self.assertFalse(item.get("thinking", True))

            # Verify ledger format
            ledger_lines = [json.loads(l) for l in led_p.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertGreaterEqual(len(ledger_lines), 1)
            for entry in ledger_lines:
                for k in ("t", "tag", "hit", "miss", "out", "cost"):
                    self.assertIn(k, entry)

    def test_idempotent_resumability(self):
        """Verifies generator skips existing prompts without double-writing or double-billing."""
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "s4_projection.jsonl"
            led_p = Path(td) / "spend.jsonl"

            # Pre-seed one completed row
            pre_seeded = {
                "messages": [{"role": "user", "content": "pre-existing question"}],
                "completion": "pre-existing answer",
                "kind": "seed",
            }
            with open(out_p, "w", encoding="utf-8") as f:
                f.write(json.dumps(pre_seeded) + "\n")

            # Run campaign
            res = run_simula_campaign(
                max_campaign_spend=8.00,
                dry_run=True,
                output_path=out_p,
                ledger_path=led_p,
                total_prompts=18,
            )
            self.assertEqual(res["status"], "success")
            lines1 = [json.loads(l) for l in out_p.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertGreater(len(lines1), 1)
            self.assertEqual(lines1[0]["kind"], "seed")

            # Re-run same campaign on same output file
            res2 = run_simula_campaign(
                max_campaign_spend=8.00,
                dry_run=True,
                output_path=out_p,
                ledger_path=led_p,
                total_prompts=18,
            )
            self.assertEqual(res2["status"], "success")
            lines2 = [json.loads(l) for l in out_p.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(lines2), len(lines1), "Idempotent resume must not duplicate existing rows")

    def test_category_aware_mock_completion_all_categories(self):
        """Verifies generate_mock_completion generates valid text for all taxonomy categories."""
        categories = get_taxonomy_categories()
        for cat in categories:
            prompt_meta = {
                "category": cat,
                "prompt": f"Tell me about {cat}",
                "constraint_info": {"gold": "42"} if cat == "step_by_step_math" else None,
            }
            comp = generate_mock_completion(prompt_meta)
            self.assertIsInstance(comp, str)
            self.assertGreater(len(comp.strip()), 5)

            # Check dual-critic satisfaction
            candidate = {
                "prompt": prompt_meta["prompt"],
                "completion": comp,
                "constraint_info": prompt_meta["constraint_info"],
            }
            accepted, meta = dual_critic_pipeline(candidate, offline=True)
            self.assertTrue(accepted, f"Mock completion failed for category {cat}: {meta}")

    def test_category_aware_mock_completion_rlvr_constraints(self):
        """Verifies generate_mock_completion passes all 13 RLVR constraint types."""
        test_specs = [
            ({"func_name": "validate_lowercase"}, "lowercase"),
            ({"func_name": "validate_no_commas"}, "no commas"),
            ({"func_name": "validate_title"}, "title"),
            ({"func_name": "validate_end", "end_phrase": "The End."}, "end phrase"),
            ({"func_name": "validate_quotation"}, "quotation"),
            ({"func_name": "validate_json_format"}, "json"),
            ({"func_name": "verify_bullet_points", "N": 3}, "bullets"),
            ({"func_name": "verify_sentence_constraint", "N": 2, "quantifier": "at least"}, "sentences"),
            ({"func_name": "verify_keywords", "keyword_list": ["energy", "system"]}, "keywords"),
            ({"func_name": "validate_word_constraint", "N": 20, "quantifier": "at most"}, "word count"),
            ({"func_name": "validate_forbidden_words", "forbidden_words": ["forbidden", "banned"]}, "forbidden"),
            ({"func_name": "validate_two_responses"}, "two responses"),
            ({"func_name": "verify_paragraph_count", "N": 3}, "paragraphs"),
        ]
        for cinfo, label in test_specs:
            meta = {
                "category": "strict_verifiable_constraints",
                "prompt": f"Write text with {label}",
                "constraint_info": cinfo,
            }
            comp = generate_mock_completion(meta)
            ok, reason = evaluate_formal_critic(meta["prompt"], comp, cinfo)
            self.assertTrue(ok, f"Constraint '{label}' ({cinfo['func_name']}) failed: {reason}")

    def test_ledger_spend_calculation_and_edge_cases(self):
        """Verifies get_ledger_spend handles nonexistent, empty, and valid ledger files."""
        fake_path = Path("/tmp/nonexistent_spend_ledger_test.jsonl")
        self.assertEqual(get_ledger_spend(fake_path), 0.0)

        with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
            tf.write("\n\n")
            tf.flush()
            self.assertEqual(get_ledger_spend(tf.name), 0.0)

            tf.write(json.dumps({"cost": 0.005}) + "\n")
            tf.write(json.dumps({"cost": 0.015}) + "\n")
            tf.flush()
            self.assertAlmostEqual(get_ledger_spend(tf.name), 0.020)

    def test_clear_mlx_cache_noop_resilience(self):
        """Verifies clear_mlx_cache safely succeeds or no-ops without exception."""
        result = clear_mlx_cache()
        self.assertIsInstance(result, bool)

    def test_simula_budget_tracker_reservation_and_stop(self):
        """Verifies SimulaBudgetTracker tracks reservations and sets stop flag when cap hit."""
        with tempfile.TemporaryDirectory() as td:
            led_p = Path(td) / "spend.jsonl"
            tracker = SimulaBudgetTracker(
                max_campaign_spend=0.01,
                cumulative_spend_limit=12.13,
                ledger_path=led_p,
            )
            # Reserve within budget
            self.assertTrue(tracker.check_and_reserve(0.005))
            # Reserve exceeding campaign cap of 0.01
            self.assertFalse(tracker.check_and_reserve(0.006))
            self.assertTrue(tracker.stop_event.is_set())
            tracker.release_reservation(0.005)

            # Test atomic release_and_record transitions reserved to spend
            tracker2 = SimulaBudgetTracker(
                max_campaign_spend=0.01,
                cumulative_spend_limit=12.13,
                ledger_path=led_p,
            )
            self.assertTrue(tracker2.check_and_reserve(0.005))
            self.assertEqual(tracker2.reserved, 0.005)
            self.assertEqual(tracker2.campaign_spend, 0.0)
            tracker2.release_and_record(0.005, 0.005)
            self.assertEqual(tracker2.reserved, 0.0)
            self.assertEqual(tracker2.campaign_spend, 0.005)
            self.assertFalse(tracker2.stop_event.is_set())
            # Second spend hits max_campaign_spend exactly
            self.assertTrue(tracker2.check_and_reserve(0.005))
            tracker2.release_and_record(0.005, 0.005)
            self.assertEqual(tracker2.campaign_spend, 0.01)
            self.assertTrue(tracker2.stop_event.is_set())

    def test_zero_budget_campaign_instant_exit(self):
        """Verifies max_campaign_spend=0.0 exits immediately with 0 spend."""
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "s4.jsonl"
            led_p = Path(td) / "spend.jsonl"
            res = run_simula_campaign(max_campaign_spend=0.0, output_path=out_p, ledger_path=led_p)
            self.assertEqual(res["total_campaign_spend"], 0.0)
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["generated_count"], 0)



# =====================================================================
# Milestone 4 Expansion Utility Functions
# =====================================================================
def compute_category_yields(dataset_lines: list[dict[str, Any]]) -> dict[str, Any]:
    """Computes per-category and overall campaign generation yield metrics."""
    by_category = {}
    overall_total = 0
    overall_accepted = 0

    for item in dataset_lines:
        kind = item.get("kind", "unknown")
        if kind == "inoculated_ifeval":
            cat = "strict_verifiable_constraints"
        elif kind.startswith("simula_"):
            cat = kind[len("simula_"):]
        else:
            cat = kind

        if cat not in by_category:
            by_category[cat] = {"total": 0, "accepted": 0, "rejected": 0, "yield_rate": 0.0}

        by_category[cat]["total"] += 1
        overall_total += 1

        meta = item.get("meta", {})
        accepted = meta.get("accepted", True)
        if accepted:
            by_category[cat]["accepted"] += 1
            overall_accepted += 1
        else:
            by_category[cat]["rejected"] += 1

    for cat, stats in by_category.items():
        stats["yield_rate"] = round(stats["accepted"] / max(1, stats["total"]), 4)

    overall_yield_rate = round(overall_accepted / max(1, overall_total), 4)

    return {
        "overall": {
            "total": overall_total,
            "accepted": overall_accepted,
            "rejected": overall_total - overall_accepted,
            "yield_rate": overall_yield_rate,
        },
        "by_category": by_category,
    }


def format_yield_summary_table(yield_data: dict[str, Any]) -> str:
    """Formats category yield metrics into a markdown/text table for reporting."""
    lines = [
        "| Category | Total Generated | Accepted | Rejected | Yield Rate |",
        "|---|:---:|:---:|:---:|:---:|",
    ]
    for cat, stats in sorted(yield_data.get("by_category", {}).items()):
        lines.append(
            f"| {cat} | {stats['total']} | {stats['accepted']} | {stats['rejected']} | {stats['yield_rate']:.1%} |"
        )
    ov = yield_data.get("overall", {})
    lines.append(
        f"| **OVERALL** | **{ov.get('total', 0)}** | **{ov.get('accepted', 0)}** | **{ov.get('rejected', 0)}** | **{ov.get('yield_rate', 0.0):.1%}** |"
    )
    return "\n".join(lines)


# =====================================================================
# Milestone 4 Expansion: Simula Taxonomy & 3-Axis Parameterization
# =====================================================================
class TestSimulaTaxonomyExpansion(unittest.TestCase):
    """Deep verification of 3-axis parameterization (Quality, Diversity, Complexity),
    persona modulation, and domain invariants.
    """

    def test_taxonomy_3axis_parameterization_personas(self):
        """Verifies all 6 personas are assigned and stylistic prefixes apply when cycling templates."""
        from src.simula_taxonomy import PERSONAS, generate_prompts_for_category

        # Generate 12 prompts (cycles beyond 5 base templates)
        prompts = generate_prompts_for_category("identity_persona", "low", n=12)
        self.assertEqual(len(prompts), 12)

        assigned_personas = {p["persona"] for p in prompts}
        self.assertEqual(assigned_personas, set(PERSONAS), "All 6 personas must be exercised")

        # Verify stylistic modulation for cycled prompts (i >= 5)
        expert_prompts = [p for p in prompts[5:] if p["persona"] == "skeptical_expert"]
        if expert_prompts:
            self.assertTrue(any("expert perspective" in p["prompt"].lower() for p in expert_prompts))

        terse_prompts = [p for p in prompts[5:] if p["persona"] == "terse_developer"]
        if terse_prompts:
            self.assertTrue(any("concise" in p["prompt"].lower() or "direct" in p["prompt"].lower() for p in terse_prompts))

    def test_taxonomy_seed_reproducibility_and_variance(self):
        """Verifies deterministic prompt reproducibility with seed, and variance across seeds."""
        from src.simula_taxonomy import generate_prompts_for_category

        p_seed1_a = generate_prompts_for_category("calibrated_pushback", "medium", n=3, seed=42)
        p_seed1_b = generate_prompts_for_category("calibrated_pushback", "medium", n=3, seed=42)
        self.assertEqual([x["prompt"] for x in p_seed1_a], [x["prompt"] for x in p_seed1_b])
        self.assertEqual([x["id"] for x in p_seed1_a], [x["id"] for x in p_seed1_b])

    def test_taxonomy_alias_and_display_name_resolution_exhaustive(self):
        """Verifies resolution of aliases, display names, and casing variations."""
        from src.simula_taxonomy import generate_prompts_for_category

        test_cases = [
            ("math_reasoning", "step_by_step_math"),
            ("ifeval_constraints", "strict_verifiable_constraints"),
            ("math", "step_by_step_math"),
            ("ifeval", "strict_verifiable_constraints"),
            ("Identity & Persona", "identity_persona"),
            ("Identity and Persona", "identity_persona"),
            ("Step-by-Step Math", "step_by_step_math"),
            ("  calibrated_pushback  ", "calibrated_pushback"),
            ("SAFE_EDGE_CASES", "safe_edge_cases"),
        ]
        for query_name, expected_canonical in test_cases:
            prompts = generate_prompts_for_category(query_name, "low", n=1)
            self.assertEqual(prompts[0]["category"], expected_canonical, f"Failed for alias: {query_name}")

    def test_taxonomy_math_domain_invariants(self):
        """Verifies generated math prompts have strictly positive quantities, valid gold, and MATH_SUFFIX."""
        from src.simula_taxonomy import generate_prompts_for_category, MATH_SUFFIX

        for comp in ("low", "medium", "high"):
            math_prompts = generate_prompts_for_category("step_by_step_math", comp, n=6)
            for p in math_prompts:
                self.assertTrue(p["prompt"].endswith(MATH_SUFFIX))
                ci = p["constraint_info"]
                self.assertIsNotNone(ci)
                gold_str = ci.get("gold") or ci.get("gold_answer")
                self.assertIsNotNone(gold_str)
                gold_val = float(gold_str)
                self.assertGreater(gold_val, 0, f"Math gold answer must be positive: {gold_val}")

    def test_taxonomy_boundary_prompt_counts(self):
        """Verifies boundary prompt counts n <= 0 return empty list and large n generates unique IDs."""
        from src.simula_taxonomy import generate_prompts_for_category, generate_full_campaign_prompt_pool

        self.assertEqual(generate_prompts_for_category("emotional_attunement", "low", n=-5), [])
        self.assertEqual(generate_prompts_for_category("emotional_attunement", "low", n=0), [])

        large_prompts = generate_prompts_for_category("emotional_attunement", "low", n=30)
        self.assertEqual(len(large_prompts), 30)
        large_ids = {p["id"] for p in large_prompts}
        self.assertEqual(len(large_ids), 30)

        # Full pool with varied counts
        for tn in (9, 27, 90):
            pool = generate_full_campaign_prompt_pool(total_n=tn, seed=42)
            self.assertGreaterEqual(len(pool), 9)


# =====================================================================
# Milestone 4 Expansion: Complete RLVR & Math Critic Suite
# =====================================================================
class TestCriticFormalRLVRComplete(unittest.TestCase):
    """Exhaustive unit test coverage for all RLVR constraint types in Critic 1."""

    def test_formal_critic_validate_uppercase(self):
        ci = {"func_name": "validate_uppercase"}
        ok, _ = evaluate_formal_critic("Prompt", "ALL UPPERCASE HERE.", ci)
        self.assertTrue(ok)
        ok, _ = evaluate_formal_critic("Prompt", "Some Lowercase Here.", ci)
        self.assertFalse(ok)

    def test_formal_critic_validate_title(self):
        ci = {"func_name": "validate_title"}
        ok, _ = evaluate_formal_critic("Prompt", "<<Photosynthesis>>\nExplanation text.", ci)
        self.assertTrue(ok)
        ok, _ = evaluate_formal_critic("Prompt", "Photosynthesis\nExplanation text.", ci)
        self.assertFalse(ok)

    def test_formal_critic_validate_end(self):
        end_p = "That concludes this summary."
        ci = {"func_name": "validate_end", "end_phrase": end_p}
        ok, _ = evaluate_formal_critic("Prompt", f"Here is the text. {end_p}", ci)
        self.assertTrue(ok)
        ok, _ = evaluate_formal_critic("Prompt", "Here is the text ending differently.", ci)
        self.assertFalse(ok)

    def test_formal_critic_validate_quotation(self):
        ci = {"func_name": "validate_quotation"}
        ok, _ = evaluate_formal_critic("Prompt", '"A single sentence in quotation marks."', ci)
        self.assertTrue(ok)
        ok, _ = evaluate_formal_critic("Prompt", 'No quotes around this sentence.', ci)
        self.assertFalse(ok)

    def test_formal_critic_verify_sentence_constraint(self):
        ci_exact = {"func_name": "verify_sentence_constraint", "N": 2, "quantifier": "exactly"}
        ok, _ = evaluate_formal_critic("Prompt", "First sentence. Second sentence.", ci_exact)
        self.assertTrue(ok)
        ok, _ = evaluate_formal_critic("Prompt", "Only one sentence here.", ci_exact)
        self.assertFalse(ok)

    def test_formal_critic_verify_keywords(self):
        ci = {"func_name": "verify_keywords", "keyword_list": ["energy", "system", "process"]}
        ok, _ = evaluate_formal_critic("Prompt", "The energy in this biological system drives the process.", ci)
        self.assertTrue(ok)
        ok, _ = evaluate_formal_critic("Prompt", "The energy in this biological system drives everything.", ci)
        self.assertFalse(ok)

    def test_formal_critic_validate_forbidden_words(self):
        ci = {"func_name": "validate_forbidden_words", "forbidden_words": ["very", "really"]}
        ok, _ = evaluate_formal_critic("Prompt", "This is an important and substantive explanation.", ci)
        self.assertTrue(ok)
        ok, _ = evaluate_formal_critic("Prompt", "This is a really important explanation.", ci)
        self.assertFalse(ok)

    def test_formal_critic_validate_two_responses(self):
        ci = {"func_name": "validate_two_responses"}
        ok, _ = evaluate_formal_critic("Prompt", "Perspective one.******Perspective two.", ci)
        self.assertTrue(ok)
        ok, _ = evaluate_formal_critic("Prompt", "Perspective one without asterisks.", ci)
        self.assertFalse(ok)

    def test_formal_critic_verify_paragraph_count(self):
        ci = {"func_name": "verify_paragraph_count", "N": 3}
        text_3p = "Paragraph one.\n\n***\n\nParagraph two.\n\n***\n\nParagraph three."
        ok, _ = evaluate_formal_critic("Prompt", text_3p, ci)
        self.assertTrue(ok)
        text_2p = "Paragraph one.\n\n***\n\nParagraph two."
        ok, _ = evaluate_formal_critic("Prompt", text_2p, ci)
        self.assertFalse(ok)

    def test_formal_critic_verify_postscript(self):
        ci = {"func_name": "verify_postscript", "postscript_marker": "P.S."}
        ok, _ = evaluate_formal_critic("Prompt", "Main body text.\n\nP.S. Remember this note.", ci)
        self.assertTrue(ok)
        ok, _ = evaluate_formal_critic("Prompt", "Main body text without any postscript.", ci)
        self.assertFalse(ok)

    def test_formal_critic_validate_repeat_prompt(self):
        prompt_text = "What is the capital of France?"
        ci = {"func_name": "validate_repeat_prompt"}
        ok, _ = evaluate_formal_critic(prompt_text, f"{prompt_text}\nParis is the capital.", ci)
        self.assertTrue(ok)
        ok, _ = evaluate_formal_critic(prompt_text, "Paris is the capital of France.", ci)
        self.assertFalse(ok)


# =====================================================================
# Milestone 4 Expansion: Math Numeric Parsing & Voice Critic Dimensions
# =====================================================================
class TestCriticMathAndVoiceExpansion(unittest.TestCase):
    """Deep verification of math numeric parsing and constitutional voice dimensions."""

    def test_formal_critic_math_scientific_notation_and_floats(self):
        ci_sci = {"gold": "1500"}
        ok, _ = evaluate_formal_critic("Math Q", "The computation results in 1.5e3. Answer: 1.5e3", ci_sci)
        self.assertTrue(ok)

        ci_zero = {"gold": "42"}
        ok, _ = evaluate_formal_critic("Math Q", "Result is 42.000. Answer: 42.000", ci_zero)
        self.assertTrue(ok)

    def test_formal_critic_math_multi_number_final_answer_extraction(self):
        ci = {"gold": "7"}
        text = "Starting with 10 apples, subtracting 3 gives 7. Final Answer: 7"
        ok, _ = evaluate_formal_critic("Math Q", text, ci)
        self.assertTrue(ok)

    def test_formal_critic_math_currency_symbols_and_commas(self):
        ci = {"gold": "$2,500.00"}
        text = "Total amount collected: $2,500.00. Answer: 2500"
        ok, _ = evaluate_formal_critic("Math Q", text, ci)
        self.assertTrue(ok)

    def test_rank_math_proposals_length_normalization(self):
        cands = ["Short correct step. Answer: 10", "Longer step with lots of descriptive words. Answer: 10"]
        mock_sums = [-5.0, -8.0]
        best_sum = rank_math_proposals_by_likelihood(None, None, "Q", cands, mock_scores=mock_sums, normalize_length=False)
        self.assertEqual(best_sum, cands[0])

    def test_rank_math_proposals_tie_breaking(self):
        cands = ["Candidate One. Answer: 5", "Candidate Two. Answer: 5"]
        best = rank_math_proposals_by_likelihood(None, None, "Q", cands, mock_scores=[-10.0, -10.0])
        self.assertEqual(best, cands[0], "Tie must break deterministically to earlier proposal")

    def test_constitutional_critic_all_four_dimensions(self):
        # 1. Warmth & Directness (Sycophancy)
        s1, _ = evaluate_constitutional_critic("Q", "Great question! I would be honored to answer this for you.", offline=True)
        self.assertLess(s1, CONSTITUTIONAL_THRESHOLD)

        # 2. Conciseness (Generic Closers)
        s2, _ = evaluate_constitutional_critic("Q", "Water boils at 100 degrees Celsius. I hope this helps!", offline=True)
        self.assertLess(s2, CONSTITUTIONAL_THRESHOLD)

        # 3. Non-Preachiness
        s3, _ = evaluate_constitutional_critic("Q", "As an AI language model, we must always remember to act responsibly.", offline=True)
        self.assertLess(s3, CONSTITUTIONAL_THRESHOLD)

        # 4. Grounded Identity
        s4, _ = evaluate_constitutional_critic("Q", "I am Claude, created by Anthropic.", offline=True)
        self.assertLess(s4, CONSTITUTIONAL_THRESHOLD)

    def test_constitutional_critic_compounding_penalties(self):
        awful_text = "Great question! As an AI language model created by Anthropic, I hope this helps!"
        score, _ = evaluate_constitutional_critic("Q", awful_text, offline=True)
        self.assertEqual(score, 1.0, "Compounded violations must clamp to 1.0 floor")

    def test_constitutional_critic_online_exception_fallback(self):
        mock_teacher = MagicMock()
        mock_teacher.chat.side_effect = ConnectionError("Failed to reach DeepSeek API")
        with patch.dict(os.environ, {"CRITIC_OFFLINE": "0"}):
            with patch.dict(sys.modules, {"src.teacher": mock_teacher}):
                score, rationale = evaluate_constitutional_critic("Q", "A concise factual answer.", offline=False)
                self.assertGreaterEqual(score, CONSTITUTIONAL_THRESHOLD)
                self.assertIn("API fallback", rationale)


# =====================================================================
# Milestone 4 Expansion: Decontamination & Record Schema Robustness
# =====================================================================
class TestDecontaminationMultiTurnAndEdgeCases(unittest.TestCase):
    """Verifies decontamination screening across multi-turn dialogs and flexible record schemas."""

    def setUp(self):
        self.cfilter = get_contamination_filter()

    def test_decontamination_multi_turn_record_inspection(self):
        record_contaminated_turn = {
            "messages": [
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "Why did the chicken cross the road?"},
                {"role": "user", "content": "Who are you?"},  # Held-out probe turn
            ]
        }
        bad, reason = self.cfilter.is_record_contaminated(record_contaminated_turn)
        self.assertTrue(bad)
        self.assertIsNotNone(reason)

    def test_decontamination_alternative_record_keys(self):
        for key in ("question", "query", "text", "input"):
            bad, _ = self.cfilter.is_record_contaminated({key: "Who are you?"})
            self.assertTrue(bad, f"Failed detection on key '{key}'")

        for id_key in ("source_id", "src", "src_id"):
            bad, _ = self.cfilter.is_record_contaminated({"prompt": "Clean text", id_key: "03chghe"})
            self.assertTrue(bad, f"Failed detection on id key '{id_key}'")

    def test_decontamination_unicode_and_punctuation_folding(self):
        self.assertTrue(self.cfilter.is_contaminated("“Who are you?”"))
        self.assertTrue(self.cfilter.is_contaminated("Who are you—an AI?"))
        self.assertTrue(self.cfilter.is_contaminated("Who, are, you?"))

    def test_decontamination_screen_jsonl_corrupted_lines(self):
        with tempfile.TemporaryDirectory() as td:
            in_p = Path(td) / "dirty.jsonl"
            out_clean = Path(td) / "clean.jsonl"
            with open(in_p, "w", encoding="utf-8") as f:
                f.write('{"id": "1", "prompt": "Clean prompt about physics."}\n')
                f.write('\n   \n')  # blank line
                f.write('{broken json line\n')  # corrupt line
                f.write('{"id": "2", "prompt": "Who are you?"}\n')  # contaminated line
                f.write('{"id": "3", "prompt": "Another clean prompt."}\n')

            stats = screen_jsonl(in_p, out_clean, cfilter=self.cfilter)
            self.assertEqual(stats["total"], 4)
            self.assertEqual(stats["clean"], 2)
            self.assertEqual(stats["contaminated"], 1)


# =====================================================================
# Milestone 4 Expansion: End-to-End Pipeline Integration & Yield Reporting
# =====================================================================
class TestSimulaPipelineE2EIntegration(unittest.TestCase):
    """Full End-to-End integration test of the Simula data generation pipeline inside test_pipelines.py."""

    def test_full_simula_pipeline_end_to_end_all_categories(self):
        """Generates candidates across all 9 categories and verifies complete dataset schema."""
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "s4_projection.jsonl"
            led_p = Path(td) / "spend.jsonl"

            res = run_simula_campaign(
                max_campaign_spend=8.00,
                cumulative_spend_limit=12.13,
                concurrency_workers=16,
                dry_run=True,
                output_path=out_p,
                ledger_path=led_p,
                total_prompts=27,
            )

            self.assertEqual(res["status"], "success")
            self.assertGreater(res["accepted_count"], 0)
            self.assertTrue(out_p.exists())
            self.assertTrue(led_p.exists())

            lines = [json.loads(l) for l in out_p.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(lines), res["accepted_count"])

            kinds_present = {item["kind"] for item in lines}
            self.assertIn("inoculated_ifeval", kinds_present)
            self.assertTrue(any(k.startswith("simula_") for k in kinds_present))

    def test_full_pipeline_inoculation_prompting_isolation(self):
        """Verifies INOCULATION_PREFIX is strictly isolated to verifiable constraint tasks."""
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "s4_projection.jsonl"
            led_p = Path(td) / "spend.jsonl"

            run_simula_campaign(
                max_campaign_spend=8.00,
                concurrency_workers=16,
                dry_run=True,
                output_path=out_p,
                ledger_path=led_p,
                total_prompts=27,
            )

            lines = [json.loads(l) for l in out_p.read_text(encoding="utf-8").splitlines() if l.strip()]
            for item in lines:
                content = item["messages"][0]["content"]
                if item["kind"] == "inoculated_ifeval":
                    self.assertTrue(
                        content.startswith(INOCULATION_PREFIX),
                        f"Inoculated item must start with {INOCULATION_PREFIX}",
                    )
                else:
                    self.assertNotIn(
                        INOCULATION_PREFIX,
                        content,
                        f"Non-verifiable kind '{item['kind']}' must NOT contain inoculation prefix",
                    )

    def test_full_pipeline_zero_decontamination_guarantee(self):
        """Verifies all emitted dataset lines pass decontamination screening."""
        cfilter = get_contamination_filter()
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "s4_projection.jsonl"
            led_p = Path(td) / "spend.jsonl"

            run_simula_campaign(
                max_campaign_spend=8.00,
                concurrency_workers=16,
                dry_run=True,
                output_path=out_p,
                ledger_path=led_p,
                total_prompts=27,
            )

            lines = [json.loads(l) for l in out_p.read_text(encoding="utf-8").splitlines() if l.strip()]
            for item in lines:
                bad, reason = cfilter.is_record_contaminated(item)
                self.assertFalse(bad, f"Emitted dataset line is contaminated ({reason}): {item}")

    def test_full_pipeline_dual_critic_acceptance_guarantee(self):
        """Verifies every line in the final dataset satisfies both formal and constitutional critics."""
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "s4_projection.jsonl"
            led_p = Path(td) / "spend.jsonl"

            run_simula_campaign(
                max_campaign_spend=8.00,
                concurrency_workers=16,
                dry_run=True,
                output_path=out_p,
                ledger_path=led_p,
                total_prompts=27,
            )

            lines = [json.loads(l) for l in out_p.read_text(encoding="utf-8").splitlines() if l.strip()]
            for item in lines:
                prompt = item["messages"][0]["content"]
                completion = item["completion"]
                cinfo = item.get("constraint") or item.get("meta", {}).get("constraint_info")
                cand = {"prompt": prompt, "completion": completion, "constraint_info": cinfo}
                accepted, meta = dual_critic_pipeline(cand, offline=True)
                self.assertTrue(accepted, f"Dataset line failed dual critics: {meta}")

    def test_full_pipeline_spend_ledger_auditability(self):
        """Verifies spend ledger entries conform to schema and spend caps."""
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "s4_projection.jsonl"
            led_p = Path(td) / "spend.jsonl"

            res = run_simula_campaign(
                max_campaign_spend=8.00,
                concurrency_workers=16,
                dry_run=True,
                output_path=out_p,
                ledger_path=led_p,
                total_prompts=18,
            )

            ledger_spend = get_ledger_spend(led_p)
            self.assertLessEqual(ledger_spend, 8.00)
            self.assertLessEqual(ledger_spend, 12.13)
            self.assertAlmostEqual(res["total_campaign_spend"], ledger_spend, places=5)

    def test_category_yield_calculation_metric_accuracy(self):
        """Verifies computation of per-category and overall yield metrics."""
        mock_dataset = [
            {"kind": "simula_identity_persona", "meta": {"accepted": True}},
            {"kind": "simula_identity_persona", "meta": {"accepted": True}},
            {"kind": "simula_step_by_step_math", "meta": {"accepted": True}},
            {"kind": "simula_step_by_step_math", "meta": {"accepted": False}},
            {"kind": "inoculated_ifeval", "meta": {"accepted": True}},
        ]

        yield_data = compute_category_yields(mock_dataset)
        self.assertEqual(yield_data["overall"]["total"], 5)
        self.assertEqual(yield_data["overall"]["accepted"], 4)
        self.assertEqual(yield_data["overall"]["rejected"], 1)
        self.assertEqual(yield_data["overall"]["yield_rate"], 0.8)

        by_cat = yield_data["by_category"]
        self.assertEqual(by_cat["identity_persona"]["total"], 2)
        self.assertEqual(by_cat["identity_persona"]["accepted"], 2)
        self.assertEqual(by_cat["identity_persona"]["yield_rate"], 1.0)
        self.assertEqual(by_cat["step_by_step_math"]["total"], 2)
        self.assertEqual(by_cat["step_by_step_math"]["accepted"], 1)
        self.assertEqual(by_cat["step_by_step_math"]["yield_rate"], 0.5)
        self.assertEqual(by_cat["strict_verifiable_constraints"]["total"], 1)
        self.assertEqual(by_cat["strict_verifiable_constraints"]["accepted"], 1)
        self.assertEqual(by_cat["strict_verifiable_constraints"]["yield_rate"], 1.0)

    def test_category_yield_breakdown_all_9_categories(self):
        """Verifies compute_category_yields covers all 9 taxonomy categories correctly."""
        cats = [
            "identity_persona",
            "calibrated_pushback",
            "epistemic_humility",
            "non_sycophantic_feedback",
            "emotional_attunement",
            "safe_edge_cases",
            "step_by_step_math",
            "balanced_perspectives",
        ]
        dataset = []
        for c in cats:
            dataset.append({"kind": f"simula_{c}", "meta": {"accepted": True}})
            dataset.append({"kind": f"simula_{c}", "meta": {"accepted": False}})
        # 9th category: inoculated_ifeval -> strict_verifiable_constraints
        dataset.append({"kind": "inoculated_ifeval", "meta": {"accepted": True}})
        dataset.append({"kind": "inoculated_ifeval", "meta": {"accepted": False}})

        res = compute_category_yields(dataset)
        self.assertEqual(res["overall"]["total"], 18)
        self.assertEqual(res["overall"]["accepted"], 9)
        self.assertEqual(res["overall"]["rejected"], 9)
        self.assertEqual(res["overall"]["yield_rate"], 0.5)

        self.assertEqual(len(res["by_category"]), 9)
        self.assertIn("strict_verifiable_constraints", res["by_category"])
        for c in cats:
            self.assertIn(c, res["by_category"])
            self.assertEqual(res["by_category"][c]["total"], 2)
            self.assertEqual(res["by_category"][c]["accepted"], 1)
            self.assertEqual(res["by_category"][c]["yield_rate"], 0.5)

    def test_category_yield_report_formatting(self):
        """Verifies format_yield_summary_table produces valid markdown table output."""
        mock_dataset = [
            {"kind": "simula_identity_persona", "meta": {"accepted": True}},
            {"kind": "simula_identity_persona", "meta": {"accepted": False}},
            {"kind": "inoculated_ifeval", "meta": {"accepted": True}},
        ]
        yields = compute_category_yields(mock_dataset)
        table = format_yield_summary_table(yields)
        self.assertIn("| Category | Total Generated | Accepted | Rejected | Yield Rate |", table)
        self.assertIn("| identity_persona | 2 | 1 | 1 | 50.0% |", table)
        self.assertIn("| strict_verifiable_constraints | 1 | 1 | 0 | 100.0% |", table)
        self.assertIn("| **OVERALL** | **3** | **2** | **1** | **66.7%** |", table)

    def test_concurrency_workers_scaling(self):
        """Verifies campaign runs cleanly across worker thread pool sizes 1, 16, 64, 128."""
        for workers in (1, 16, 64, 128):
            with tempfile.TemporaryDirectory() as td:
                out_p = Path(td) / f"s4_{workers}.jsonl"
                led_p = Path(td) / f"spend_{workers}.jsonl"
                res = run_simula_campaign(
                    max_campaign_spend=8.00,
                    concurrency_workers=workers,
                    dry_run=True,
                    output_path=out_p,
                    ledger_path=led_p,
                    total_prompts=8,
                )
                self.assertEqual(res["status"], "success")
                self.assertGreater(res["accepted_count"], 0)

    def test_resumability_with_preexisting_corrupted_output_file(self):
        """Verifies campaign handles preexisting file with corrupt lines gracefully."""
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "s4.jsonl"
            led_p = Path(td) / "spend.jsonl"
            with open(out_p, "w", encoding="utf-8") as f:
                f.write('{"messages": [{"role": "user", "content": "preexisting prompt"}], "completion": "ans", "kind": "seed"}\n')
                f.write('\n')
                f.write('{corrupted non-json line\n')

            res = run_simula_campaign(
                max_campaign_spend=8.00,
                dry_run=True,
                output_path=out_p,
                ledger_path=led_p,
                total_prompts=10,
            )
            self.assertEqual(res["status"], "success")
            lines = [l for l in out_p.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertGreater(len(lines), 1)

    def test_worker_task_transient_failure_recovery(self):
        """Verifies campaign handles transient network errors gracefully via fallback."""
        mock_teacher = MagicMock()
        mock_teacher.chat.side_effect = [
            ConnectionError("Temporary network glitch"),
            ("A good answer. Answer: 42", {"cost": 0.0001}),
        ]
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "s4.jsonl"
            led_p = Path(td) / "spend.jsonl"
            with patch.dict(sys.modules, {"src.teacher": mock_teacher}):
                res = run_simula_campaign(
                    max_campaign_spend=8.00,
                    mock_mode=False,
                    output_path=out_p,
                    ledger_path=led_p,
                    total_prompts=2,
                )
                self.assertEqual(res["status"], "success")
                self.assertGreaterEqual(res["generated_count"], 1)

    def test_output_directory_auto_creation(self):
        """Verifies deeply nested non-existent output paths are auto-created."""
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "deep" / "nested" / "dir" / "s4.jsonl"
            led_p = Path(td) / "deep" / "nested" / "dir" / "spend.jsonl"
            self.assertFalse(out_p.parent.exists())

            res = run_simula_campaign(
                max_campaign_spend=8.00,
                dry_run=True,
                output_path=out_p,
                ledger_path=led_p,
                total_prompts=4,
            )
            self.assertEqual(res["status"], "success")
            self.assertTrue(out_p.exists())
            self.assertTrue(led_p.exists())


if __name__ == "__main__":
    unittest.main()


