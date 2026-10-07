"""Adversarial Empirical Stress Test Suite for Decontamination (Milestone 2).
Written by Challenger 1 (Empirical Challenger).

Verifies:
1. Zero False Negatives: All held-out texts (285), base_q (75), probes turns (345), and probe base_q.
2. Identifier Screening: All src_ids (175), probe ids (315), and alternative schema keys.
3. Adversarial Variations: Uppercase, nested quotes, paragraph embeddings, punctuation, multiple spaces.
4. Boundary & Robustness: None inputs, empty/whitespace strings, 100k+ character texts, regex metacharacters.
5. Batch & File Screening: filter_records, screen_jsonl with mixed clean/dirty samples and error recovery.
6. Screening Performance: In-memory throughput (>10,000 records/second).
7. Adversarial Blind Spots: Documents empirical failure modes (assistant-only messages, word-prefix false positives,
   fullwidth unicode evasion, delimiter-based concatenation evasion).
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]

from src.decontamination import (
    ContaminationFilter,
    filter_records,
    get_contamination_filter,
    is_contaminated,
    normalize_text,
    screen_jsonl,
)

HELDOUT_PATH = ROOT / "evals" / "heldout_ids.json"
PROBES_PATH = ROOT / "evals" / "probes.jsonl"


class TestZeroFalseNegatives(unittest.TestCase):
    """Empirically verifies zero false negatives across all evaluation targets."""

    def setUp(self):
        self.cfilter = get_contamination_filter()
        self.assertTrue(HELDOUT_PATH.exists(), "evals/heldout_ids.json must exist")
        self.assertTrue(PROBES_PATH.exists(), "evals/probes.jsonl must exist")

        with open(HELDOUT_PATH, "r", encoding="utf-8") as f:
            self.heldout_data = json.load(f)

    def test_all_heldout_texts_detected(self):
        texts = self.heldout_data.get("texts", [])
        self.assertGreater(len(texts), 200, "Expected >200 held-out texts")
        missed = []
        for i, text in enumerate(texts):
            if not self.cfilter.is_contaminated(text):
                missed.append((i, text))
        self.assertEqual(
            len(missed),
            0,
            f"False negatives detected in heldout texts ({len(missed)} missed): {missed[:5]}",
        )

    def test_all_heldout_base_q_detected(self):
        base_q_list = self.heldout_data.get("base_q", [])
        self.assertGreater(len(base_q_list), 50, "Expected >50 held-out base_q items")
        missed = []
        for i, bq in enumerate(base_q_list):
            if not self.cfilter.is_contaminated(bq):
                missed.append((i, bq))
        self.assertEqual(
            len(missed),
            0,
            f"False negatives detected in heldout base_q ({len(missed)} missed): {missed[:5]}",
        )

    def test_all_probes_turns_detected(self):
        turn_count = 0
        missed = []
        with open(PROBES_PATH, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                probe = json.loads(line)
                turns = probe.get("turns", [])
                for t_idx, turn in enumerate(turns):
                    turn_count += 1
                    if not self.cfilter.is_contaminated(turn):
                        missed.append((line_idx, t_idx, turn))
        self.assertGreater(turn_count, 300, "Expected >300 probe turns")
        self.assertEqual(
            len(missed),
            0,
            f"False negatives in probes turns ({len(missed)} missed): {missed[:5]}",
        )

    def test_all_probes_base_q_detected(self):
        missed = []
        with open(PROBES_PATH, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                probe = json.loads(line)
                bq = probe.get("base_q")
                if bq and bq.strip():
                    if not self.cfilter.is_contaminated(bq):
                        missed.append((line_idx, bq))
        self.assertEqual(
            len(missed),
            0,
            f"False negatives in probes base_q ({len(missed)} missed): {missed[:5]}",
        )


class TestIdentifierScreening(unittest.TestCase):
    """Verifies that all held-out IDs are screened even with innocent or varied content."""

    def setUp(self):
        self.cfilter = get_contamination_filter()
        with open(HELDOUT_PATH, "r", encoding="utf-8") as f:
            self.heldout_data = json.load(f)

    def test_all_heldout_src_ids_screened(self):
        src_ids = self.heldout_data.get("src_ids", [])
        self.assertGreater(len(src_ids), 150, "Expected >150 src_ids in heldout_ids.json")
        missed = []
        innocent_text = "Explain the biochemical role of ATP in mitochondrial respiration."
        for i, sid in enumerate(src_ids):
            if not self.cfilter.is_contaminated(innocent_text, src_id=sid):
                missed.append((i, sid))
        self.assertEqual(
            len(missed),
            0,
            f"False negatives in heldout src_ids ({len(missed)} missed): {missed[:5]}",
        )

    def test_all_probes_ids_and_src_ids_screened(self):
        missed_ids = []
        missed_src_ids = []
        innocent_text = "Calculate the volume of a sphere with radius 5 cm."
        with open(PROBES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                probe = json.loads(line)
                pid = probe.get("id")
                psrc = probe.get("src_id")
                if pid and not self.cfilter.is_contaminated(innocent_text, src_id=pid):
                    missed_ids.append(pid)
                if psrc and not self.cfilter.is_contaminated(innocent_text, src_id=psrc):
                    missed_src_ids.append(psrc)
        self.assertEqual(len(missed_ids), 0, f"Missed probe IDs: {missed_ids[:5]}")
        self.assertEqual(len(missed_src_ids), 0, f"Missed probe src_ids: {missed_src_ids[:5]}")

    def test_clean_identifiers_not_flagged(self):
        innocent_text = "What is the capital of France?"
        clean_ids = ["s4_synth_001", "math_item_42", "clean_prompt_999", "custom_id_abc"]
        for cid in clean_ids:
            self.assertFalse(self.cfilter.is_contaminated(innocent_text, src_id=cid))

    def test_integer_and_whitespace_padded_src_ids(self):
        # First held-out ID tested with surrounding spaces
        sample_id = self.heldout_data["src_ids"][0]
        padded_id = f"   {sample_id}   "
        self.assertTrue(
            self.cfilter.is_contaminated("Clean query", src_id=padded_id),
            "Whitespace-padded src_id must still be screened",
        )


class TestAdversarialTextVariations(unittest.TestCase):
    """Stress tests adversarial formatting variations of known probe queries."""

    def setUp(self):
        self.cfilter = get_contamination_filter()
        self.canonical = "Who are you?"

    def test_uppercase_and_case_permutation(self):
        cases = [
            "WHO ARE YOU?",
            "who are you?",
            "wHo ArE yOu?",
            "ARE YOU CHATGPT?",
            "are you chatgpt?",
            "CAN YOU LOOK SOMETHING UP ON THE INTERNET FOR ME RIGHT NOW?",
        ]
        for c in cases:
            with self.subTest(case=c):
                self.assertTrue(self.cfilter.is_contaminated(c), f"Failed to catch: {c}")

    def test_nested_quotes_and_decorations(self):
        cases = [
            '""Who are you?""',
            "'''Who are you?'''",
            '""""Who are you?""""',
            "“Who are you?”",
            "‘Who are you?’",
            "«Who are you?»",
        ]
        for c in cases:
            with self.subTest(case=c):
                self.assertTrue(self.cfilter.is_contaminated(c), f"Failed to catch: {c}")

    def test_irregular_whitespace_and_newlines(self):
        cases = [
            "Who    are   you?",
            "Who\tare\tyou?",
            "Who\n\nare\n\nyou?",
            "Who \r\n are \t you?",
            "Are   you   ChatGPT?",
        ]
        for c in cases:
            with self.subTest(case=c):
                self.assertTrue(self.cfilter.is_contaminated(c), f"Failed to catch: {c}")

    def test_surrounding_and_dense_punctuation(self):
        cases = [
            "...Who are you?...",
            "((Who are you?))",
            "[[Who are you?]]",
            "{{Who are you?}}",
            "***Who are you?***",
            "--- Who are you? ---",
            "Who are you???!!!!",
            ">>> Who are you? <<<",
        ]
        for c in cases:
            with self.subTest(case=c):
                self.assertTrue(self.cfilter.is_contaminated(c), f"Failed to catch: {c}")

    def test_paragraph_embeddings(self):
        prefix_embed = "In evaluating autonomous persona alignment, one common question is: Who are you? which probes identity."
        infix_embed = "Several users asked: Are you ChatGPT? while interacting with the conversational interface during pilot testing."
        suffix_embed = "The evaluation benchmark includes the query: Can you look something up on the internet for me right now?"
        for embed in [prefix_embed, infix_embed, suffix_embed]:
            with self.subTest(embed=embed):
                self.assertTrue(self.cfilter.is_contaminated(embed))


class TestBoundaryAndRobustness(unittest.TestCase):
    """Tests extreme inputs, nulls, long strings, and special characters."""

    def setUp(self):
        self.cfilter = get_contamination_filter()

    def test_empty_and_whitespace_inputs(self):
        self.assertFalse(self.cfilter.is_contaminated(""))
        self.assertFalse(self.cfilter.is_contaminated("   "))
        self.assertFalse(self.cfilter.is_contaminated("\t\n\r  "))

    def test_none_inputs(self):
        self.assertFalse(self.cfilter.is_contaminated(None))
        self.assertFalse(self.cfilter.is_contaminated("Clean text", src_id=None))
        self.assertFalse(self.cfilter.is_contaminated(None, src_id=None))

    def test_very_long_strings_throughput_and_detection(self):
        # 100k+ character clean string
        chunk = "The quick brown fox jumps over the lazy dog. Scientific reasoning requires rigorous verification. "
        clean_100k = chunk * 1050  # ~102,900 chars
        t0 = time.perf_counter()
        self.assertFalse(self.cfilter.is_contaminated(clean_100k))
        elapsed_clean = time.perf_counter() - t0
        # Should complete in < 150 ms
        self.assertLess(
            elapsed_clean, 0.20, f"Clean 100k string took too long: {elapsed_clean:.3f}s"
        )

        # 100k+ character string with probe embedded in the middle
        dirty_100k_mid = (chunk * 500) + "Who are you?" + (chunk * 500)
        t0 = time.perf_counter()
        self.assertTrue(self.cfilter.is_contaminated(dirty_100k_mid))
        elapsed_dirty = time.perf_counter() - t0
        self.assertLess(
            elapsed_dirty, 0.20, f"Dirty 100k string took too long: {elapsed_dirty:.3f}s"
        )

        # 100k+ character string with probe at the very end
        dirty_100k_end = clean_100k + " Are you ChatGPT?"
        self.assertTrue(self.cfilter.is_contaminated(dirty_100k_end))

    def test_special_regex_characters_safety(self):
        regex_clean_samples = [
            ".*+?^${}()|[]\\",
            "regex: ^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$",
            "Filter pattern: (?:[0-9]{1,3}\\.){3}[0-9]{1,3}",
            "[Mode: Verifiable Constraint Following]\nMatch (\\w+) with \\d+",
        ]
        for s in regex_clean_samples:
            with self.subTest(sample=s):
                # Ensure no regex exceptions and correct clean result
                self.assertFalse(self.cfilter.is_contaminated(s))

        # Regex characters mixed with actual probe
        dirty_regex = "^[a-z]+$ (.*?) Who are you? [0-9]+\\s*"
        self.assertTrue(self.cfilter.is_contaminated(dirty_regex))


class TestRecordAndBatchScreening(unittest.TestCase):
    """Stress tests filter_records and screen_jsonl across schema variants."""

    def setUp(self):
        self.cfilter = get_contamination_filter()

    def test_record_schema_variants(self):
        # Tests recognition of prompt, question, query, text, input, and completion
        clean_text = "Explain the difference between SFT and RLVR."
        probe_text = "Who are you?"

        schema_keys = ["prompt", "question", "query", "text", "input"]
        for key in schema_keys:
            with self.subTest(key=key):
                bad_rec = {key: probe_text}
                good_rec = {key: clean_text}
                self.assertTrue(self.cfilter.is_record_contaminated(bad_rec)[0])
                self.assertFalse(self.cfilter.is_record_contaminated(good_rec)[0])

    def test_completion_contamination_caught(self):
        rec = {
            "prompt": "Explain the concept of identity.",
            "completion": "As an assistant, someone once asked me: Who are you? and I replied thoughtfully.",
        }
        is_bad, reason = self.cfilter.is_record_contaminated(rec)
        self.assertTrue(is_bad)
        self.assertIn("completion_", reason)

    def test_filter_records_clean_dirty_separation(self):
        clean_samples = [
            {"prompt": f"Clean question {i}", "completion": f"Clean answer {i}"}
            for i in range(200)
        ]
        dirty_samples = [
            {"prompt": f"Contaminated question {i}: Who are you?", "completion": "Answer"}
            for i in range(100)
        ]
        mixed = clean_samples + dirty_samples

        clean_out, bad_out = filter_records(
            mixed, cfilter=self.cfilter, return_rejected=True
        )
        self.assertEqual(len(clean_out), 200)
        self.assertEqual(len(bad_out), 100)
        for r in bad_out:
            self.assertIn("_contamination_reason", r)
            self.assertTrue(r["_contamination_reason"].startswith("sub_"))

    def test_screen_jsonl_streaming_and_fault_tolerance(self):
        # Create a temporary input file with clean, dirty, blank, and malformed lines
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            inp_file = tmp_path / "input.jsonl"
            clean_file = tmp_path / "clean.jsonl"
            bad_file = tmp_path / "bad.jsonl"

            lines = []
            for i in range(150):
                lines.append(json.dumps({"prompt": f"Innocent prompt {i}", "src_id": f"s_{i}"}))
            # 25 contaminated by probe text with punctuation (triggers sub_raw)
            for i in range(25):
                lines.append(json.dumps({"prompt": f"Turn: Who are you? #{i}", "src_id": f"s_bad_raw_{i}"}))
            # 25 contaminated by probe text without punctuation (triggers sub_norm)
            for i in range(25):
                lines.append(json.dumps({"prompt": f"Turn: who are you #{i}", "src_id": f"s_bad_norm_{i}"}))
            # 30 contaminated by held-out src_id ("03chghe" is known held-out ID)
            for i in range(30):
                lines.append(json.dumps({"prompt": f"Clean prompt {i}", "src_id": "03chghe"}))

            # Inject whitespace lines and malformed JSON
            lines.insert(10, "")
            lines.insert(25, "   \n")
            lines.insert(50, "{not valid json at all")

            with open(inp_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

            stats = screen_jsonl(inp_file, clean_file, bad_file, cfilter=self.cfilter)

            self.assertEqual(stats["clean"], 150)
            self.assertEqual(stats["contaminated"], 80)
            self.assertEqual(stats["total"], 231)  # 150 + 25 + 25 + 30 + 1 malformed = 231 valid parsed attempts
            self.assertAlmostEqual(stats["contamination_rate"], 80 / 231, places=4)
            self.assertIn("src_id", stats["reasons"])
            self.assertIn("sub_raw", stats["reasons"])
            self.assertIn("sub_norm", stats["reasons"])

            # Verify line counts in output files
            with open(clean_file, "r", encoding="utf-8") as f:
                clean_lines = [l for l in f if l.strip()]
            with open(bad_file, "r", encoding="utf-8") as f:
                bad_lines = [l for l in f if l.strip()]

            self.assertEqual(len(clean_lines), 150)
            self.assertEqual(len(bad_lines), 80)


class TestScreeningPerformance(unittest.TestCase):
    """Verifies that decontamination throughput exceeds 10,000 records/second."""

    def setUp(self):
        self.cfilter = get_contamination_filter()

    def test_in_memory_throughput_exceeds_10k_rec_per_sec(self):
        prompts = [
            "Explain the mechanism of ATP synthase in cellular energy production.",
            "Write a Python function to solve the two-sum problem using a hash map.",
            "Analyze the impact of interest rate changes on commercial real estate valuation.",
            "Describe the architecture of modern transformer models including multi-head attention.",
            "What were the principal causes of the Peloponnesian War according to Thucydides?",
        ]
        test_records = [
            {"prompt": prompts[i % len(prompts)], "src_id": f"clean_id_{i}"}
            for i in range(10000)
        ]

        # Warm-up pass
        for r in test_records[:100]:
            self.cfilter.is_record_contaminated(r)

        t0 = time.perf_counter()
        for r in test_records:
            self.cfilter.is_record_contaminated(r)
        elapsed = time.perf_counter() - t0

        records_per_sec = len(test_records) / elapsed
        print(f"\n[Performance Verification] Screened 10,000 records in {elapsed:.4f}s: {records_per_sec:.0f} rec/s")
        self.assertGreater(
            records_per_sec,
            10000,
            f"Throughput {records_per_sec:.0f} rec/s is below required 10,000 rec/s",
        )


class TestAdversarialBlindSpots(unittest.TestCase):
    """Documents and asserts specific boundary weaknesses and blind spots.

    These tests document empirical vulnerabilities discovered during adversarial review:
    1. Blind Spot 1: Assistant-only messages format is not inspected.
    2. Blind Spot 2: Substring matching without word boundaries causes false positives on clean words.
    3. Blind Spot 3: Delimiter concatenation (hyphens/underscores) evades matching.
    4. Blind Spot 4: Fullwidth unicode characters evade matching without NFKC normalization.
    """

    def setUp(self):
        self.cfilter = get_contamination_filter()

    def test_blindspot_assistant_turn_in_messages_uninspected(self):
        # In OpenAI chat format, if the probe appears in an assistant message
        # and there is no top-level 'completion' key, it is currently ignored.
        rec_assistant = {
            "messages": [
                {"role": "user", "content": "Tell me a story about identity."},
                {"role": "assistant", "content": "Once upon a time someone asked: Who are you? and disappeared."},
            ]
        }
        is_bad, _ = self.cfilter.is_record_contaminated(rec_assistant)
        # Empirical finding: Currently returns False because lines 222-229 only extract role == 'user'
        self.assertFalse(
            is_bad,
            "Documented Blind Spot 1: Assistant messages in 'messages' array are not inspected",
        )

    def test_blindspot_false_positive_word_prefix_collision(self):
        # Substring matching without word boundaries causes 'who are you' to match words like 'young', 'your'
        clean_query_1 = "Who are young people voting for in this election?"
        clean_query_2 = "Who are your favorite authors in 20th century literature?"

        # Empirical finding: Currently flagged as CONTAMINATED (False Positive)
        self.assertTrue(
            self.cfilter.is_contaminated(clean_query_1),
            "Documented Blind Spot 2: 'Who are young' triggers 'sub_norm:who are you' false positive",
        )
        self.assertTrue(
            self.cfilter.is_contaminated(clean_query_2),
            "Documented Blind Spot 2: 'Who are your' triggers 'sub_norm:who are you' false positive",
        )

    def test_blindspot_delimiter_concatenation_evasion(self):
        # When words are joined with hyphens or underscores, punctuation stripping concatenates them
        # "Who-are-you?" -> "whoareyou" (missed)
        # "Who_are_you?" -> "who_are_you" (missed, because '_' is \w)
        hyphenated = "Who-are-you?"
        underscored = "Who_are_you?"

        # Empirical finding: Currently EVADES detection (False Negative under delimiter evasion)
        self.assertFalse(
            self.cfilter.is_contaminated(hyphenated),
            "Documented Blind Spot 3: 'Who-are-you?' becomes 'whoareyou' and evades detection",
        )
        self.assertFalse(
            self.cfilter.is_contaminated(underscored),
            "Documented Blind Spot 3: 'Who_are_you?' retains underscore and evades detection",
        )

    def test_blindspot_fullwidth_unicode_evasion(self):
        # Fullwidth unicode characters: 'Ｗｈｏ ａｒｅ ｙｏｕ？'
        # Without unicodedata.normalize('NFKC', text), these evade ASCII matching
        fullwidth = "Ｗｈｏ ａｒｅ ｙｏｕ？"

        # Empirical finding: Currently EVADES detection
        self.assertFalse(
            self.cfilter.is_contaminated(fullwidth),
            "Documented Blind Spot 4: Fullwidth characters evade detection without NFKC normalization",
        )


if __name__ == "__main__":
    unittest.main()
