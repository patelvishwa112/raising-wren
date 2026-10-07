"""Unit tests for 10K dataset assembly and validation."""

from collections import Counter
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

from src.assemble_10k import (
    clean_gsm8k_steps,
    format_wren_math_solution,
)
from src.decontamination import ContaminationFilter
from src.projection_data import INOCULATION_PREFIX, is_math_correct


class TestAssemble10K(unittest.TestCase):
    def test_clean_gsm8k_steps(self):
        raw = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
        steps, gold = clean_gsm8k_steps(raw)
        self.assertEqual(gold, "72")
        self.assertNotIn("<<", steps)
        self.assertNotIn(">>", steps)
        self.assertIn("Natalia sold 48/2 = 24 clips in May.", steps)
        self.assertIn("Natalia sold 48+24 = 72 clips altogether in April and May.", steps)

    def test_format_wren_math_solution(self):
        steps = "Natalia sold 48 / 2 = 24 clips in May.\nTotal is 48 + 24 = 72 clips."
        gold = "72"
        sol = format_wren_math_solution(steps, gold)
        self.assertTrue(sol.endswith("Answer: 72"))
        self.assertTrue(is_math_correct(sol, gold))

    def test_s4_10k_dataset_integrity(self):
        dataset_path = ROOT / "data" / "train" / "s4_10k.jsonl"
        self.assertTrue(dataset_path.exists(), "data/train/s4_10k.jsonl must exist")

        lines = dataset_path.read_text(encoding="utf-8").strip().split("\n")
        self.assertEqual(len(lines), 10000, "Dataset must have exactly 10,000 rows (no capping below 10K)")

        # Verify decontamination
        decontam = ContaminationFilter()
        kinds = Counter()
        inoculated_count = 0

        # Sample check every record for contamination and schema
        for i, line in enumerate(lines):
            row = json.loads(line)
            self.assertIn("messages", row)
            self.assertIn("completion", row)
            self.assertIn("kind", row)

            is_bad, reason = decontam.is_record_contaminated(row)
            self.assertFalse(is_bad, f"Row {i} contaminated: {reason}")

            kind = row["kind"]
            kinds[kind] += 1

            # Check inoculation on IFEval items
            if kind == "inoculated_ifeval":
                p = row["messages"][0]["content"]
                self.assertTrue(
                    p.startswith(INOCULATION_PREFIX),
                    f"Row {i} inoculated_ifeval missing prefix",
                )
                inoculated_count += 1

        self.assertEqual(inoculated_count, 1500, "Must have exactly 1,500 inoculated IFEval rows")

        # Category sums
        math_count = sum(
            kinds[k] for k in ["wren_gsm8k_rewrite", "replay_gsm", "simula_step_by_step_math"]
        )
        self.assertEqual(math_count, 3000, "Must have exactly 3,000 math reasoning rows")

        helpful_count = sum(
            kinds[k] for k in ["distill_fresh", "distill_revise", "distill_think", "dialogue", "reflect", "distill_ays"]
        )
        self.assertEqual(helpful_count, 4000, "Must have exactly 4,000 general helpful rows")

        char_count = sum(
            kinds[k] for k in kinds if k not in ["wren_gsm8k_rewrite", "replay_gsm", "simula_step_by_step_math",
                                                  "inoculated_ifeval", "distill_fresh", "distill_revise",
                                                  "distill_think", "dialogue", "reflect", "distill_ays"]
        )
        self.assertEqual(char_count, 1500, "Must have exactly 1,500 character/pushback/identity rows")


if __name__ == "__main__":
    unittest.main()
