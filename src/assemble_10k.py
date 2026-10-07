"""Assemble 10K balanced dataset for Raising Wren 2.0 (Stage 4).

Follows Opus 5.5 architectural prescription based on Harvard's 'Finetuning with Sampling'
(arXiv:2610.02140) and Anthropic's Inoculation Prompting to prevent catastrophic forgetting:
- 40% General Helpful / Replay (4,000 items): SFT distillation from HelpSteer3 & OpenCharacterTraining
- 30% GSM8K / Math Reasoning (3,000 items): Base student replay + rewritten GSM8K in Wren's character
- 15% Inoculated Constraint Following (1,500 items): RLVR-IFeval with [Mode: Verifiable Constraint Following]
- 15% Character / Pushback / Identity (1,500 items): Simula 9-domain + Honesty/Pushback pairs

Strictly decontaminated against evals/heldout_ids.json and evals/probes.jsonl.
"""

from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.decontamination import ContaminationFilter
from src.projection_data import (
    MATH_FMT,
    format_inoculated_ifeval,
    is_math_correct,
)
from src.rlvr_verify import verify as rlvr_verify

TRAIN_DIR = ROOT / "data" / "train"
OUTPUT_10K = TRAIN_DIR / "s4_10k.jsonl"
EVAL_DIR = ROOT / "data" / "eval"
OUTPUT_EVAL = EVAL_DIR / "s4_eval_2500.jsonl"


def clean_gsm8k_steps(raw_answer: str) -> tuple[str, str]:
    """Extract gold answer and strip <<...>> calculator tags from GSM8K answer."""
    parts = raw_answer.split("####")
    steps_raw = parts[0].strip()
    gold = parts[-1].strip() if len(parts) > 1 else ""

    # Remove calculator annotation tags <<...>>
    clean = re.sub(r"<<[^\n>]*>>", "", steps_raw)
    clean_lines = [line.strip() for line in clean.split("\n") if line.strip()]
    return "\n".join(clean_lines), gold


def format_wren_math_solution(clean_steps: str, gold: str) -> str:
    """Format math solution in Wren's character: direct, succinct CoT, answer-first/direct-end."""
    return f"{clean_steps}\n\nAnswer: {gold}"


def assemble_10k(
    target_total: int = 10000,
    seed: int = 42,
    output_path: Path = OUTPUT_10K,
) -> list[dict[str, Any]]:
    """Assemble exactly target_total decontaminated rows into s4_10k.jsonl."""
    rng = random.Random(seed)
    decontam = ContaminationFilter()

    def is_clean(r: dict[str, Any]) -> bool:
        return not decontam.is_record_contaminated(r)[0]

    # Exact Opus ratio targets
    n_helpful = int(target_total * 0.40)      # 4,000
    n_math = int(target_total * 0.30)         # 3,000
    n_ifeval = int(target_total * 0.15)       # 1,500
    n_character = target_total - n_helpful - n_math - n_ifeval  # 1,500

    print("=" * 60)
    print("ASSEMBLING 10K DATASET FOR STAGE 4")
    print(f"Target breakdown:")
    print(f"  - General Helpful / Replay:          {n_helpful:5d} (40%)")
    print(f"  - GSM8K / Math Reasoning:            {n_math:5d} (30%)")
    print(f"  - Inoculated Constraint Following:   {n_ifeval:5d} (15%)")
    print(f"  - Character / Pushback / Identity:   {n_character:5d} (15%)")
    print(f"  Total target:                        {target_total:5d}")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. CHARACTER / PUSHBACK / IDENTITY (1,500)
    # -------------------------------------------------------------
    print("\n[1/4] Assembling Character / Pushback / Identity data...")
    char_rows: list[dict[str, Any]] = []

    # A. Simula 9-domain dataset (exclude inoculated_ifeval and math which go to their tracks)
    s4_proj_file = TRAIN_DIR / "s4_projection.jsonl"
    if s4_proj_file.exists():
        with open(s4_proj_file) as f:
            for line in f:
                row = json.loads(line)
                kind = row.get("kind", "")
                if kind in ("inoculated_ifeval", "simula_step_by_step_math"):
                    continue
                if is_clean(row):
                    char_rows.append(row)
    print(f"  Loaded {len(char_rows)} Simula character examples")

    # B. Honesty dataset (calibrated pushback, trivia honesty)
    honesty_file = TRAIN_DIR / "honesty.jsonl"
    if honesty_file.exists():
        with open(honesty_file) as f:
            for line in f:
                row = json.loads(line)
                if is_clean(row):
                    char_rows.append(row)
    print(f"  Total character candidate pool: {len(char_rows)}")

    rng.shuffle(char_rows)
    selected_char = char_rows[:n_character]
    if len(selected_char) < n_character:
        raise ValueError(f"Insufficient character data: {len(selected_char)} < {n_character}")
    print(f"  Selected: {len(selected_char)} character rows")

    # -------------------------------------------------------------
    # 2. INOCULATED CONSTRAINT FOLLOWING (1,500)
    # -------------------------------------------------------------
    print("\n[2/4] Assembling Inoculated Constraint Following data...")
    ifeval_rows: list[dict[str, Any]] = []

    # A. Replay IFEval (base student verified)
    replay_file = TRAIN_DIR / "replay.jsonl"
    if replay_file.exists():
        with open(replay_file) as f:
            for line in f:
                row = json.loads(line)
                if row.get("kind") == "replay_ife":
                    p = row["messages"][0]["content"]
                    c = row["completion"]
                    inoc_row = format_inoculated_ifeval(p, c, constraint_info="replay_ife")
                    if is_clean(inoc_row):
                        ifeval_rows.append(inoc_row)
    print(f"  Loaded {len(ifeval_rows)} base replay IFEval examples")

    # B. Stage 3 fix IFEval (verified teacher answers)
    s3_fix_file = TRAIN_DIR / "s3_fix.jsonl"
    if s3_fix_file.exists():
        with open(s3_fix_file) as f:
            for line in f:
                row = json.loads(line)
                if row.get("kind") == "ife_wren":
                    p = row["messages"][0]["content"]
                    c = row["completion"]
                    inoc_row = format_inoculated_ifeval(p, c, constraint_info="s3_ife_wren")
                    if is_clean(inoc_row):
                        ifeval_rows.append(inoc_row)
    print(f"  Pool after s3_fix: {len(ifeval_rows)}")

    # C. Simula inoculated IFEval
    if s4_proj_file.exists():
        with open(s4_proj_file) as f:
            for line in f:
                row = json.loads(line)
                if row.get("kind") == "inoculated_ifeval":
                    if is_clean(row):
                        ifeval_rows.append(row)
    print(f"  Pool after Simula IFEval: {len(ifeval_rows)}")

    # D. Additional teacher IFEval from s3_ife.jsonl
    s3_ife_file = ROOT / "data" / "gen" / "s3_ife.jsonl"
    if s3_ife_file.exists():
        with open(s3_ife_file) as f:
            for line in f:
                r = json.loads(line)
                p = r.get("job", {}).get("prompt", "")
                c = r.get("text", "")
                if p and c and c.strip():
                    inoc_row = format_inoculated_ifeval(
                        p, c, constraint_info=r.get("job", {}).get("constraint")
                    )
                    if is_clean(inoc_row):
                        ifeval_rows.append(inoc_row)
    print(f"  Pool after s3_ife: {len(ifeval_rows)}")

    rng.shuffle(ifeval_rows)
    # Deduplicate by prompt
    seen_prompts: set[str] = set()
    deduped_ifeval: list[dict[str, Any]] = []
    for row in ifeval_rows:
        p_text = row["messages"][0]["content"].strip()
        if p_text not in seen_prompts:
            seen_prompts.add(p_text)
            deduped_ifeval.append(row)

    selected_ifeval = deduped_ifeval[:n_ifeval]
    if len(selected_ifeval) < n_ifeval:
        raise ValueError(
            f"Insufficient inoculated IFEval data: {len(selected_ifeval)} < {n_ifeval}"
        )
    print(f"  Selected: {len(selected_ifeval)} inoculated IFEval rows")

    # -------------------------------------------------------------
    # 3. GSM8K / MATH REASONING (3,000)
    # -------------------------------------------------------------
    print("\n[3/4] Assembling GSM8K / Math Reasoning data...")
    math_rows: list[dict[str, Any]] = []

    # A. Base student greedy replay math from replay.jsonl
    if replay_file.exists():
        with open(replay_file) as f:
            for line in f:
                row = json.loads(line)
                if row.get("kind") == "replay_gsm":
                    if is_clean(row):
                        math_rows.append(row)
    print(f"  Loaded {len(math_rows)} student greedy replay GSM8K examples")

    # B. Simula step-by-step math
    if s4_proj_file.exists():
        with open(s4_proj_file) as f:
            for line in f:
                row = json.loads(line)
                if row.get("kind") == "simula_step_by_step_math":
                    if is_clean(row):
                        math_rows.append(row)
    print(f"  Pool after Simula math: {len(math_rows)}")

    # C. Rewritten GSM8K train split in Wren's character
    needed_math = n_math - len(math_rows)
    print(f"  Generating {needed_math} rewritten GSM8K examples from openai/gsm8k train...")
    from datasets import load_dataset
    gsm_ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)

    added_gsm = 0
    for item in gsm_ds:
        if added_gsm >= needed_math:
            break
        q = item["question"]
        raw_ans = item["answer"]
        steps, gold = clean_gsm8k_steps(raw_ans)
        completion = format_wren_math_solution(steps, gold)

        if not is_math_correct(completion, gold):
            continue

        prompt_str = f"{q}{MATH_FMT}"
        row = {
            "messages": [{"role": "user", "content": prompt_str}],
            "thinking": False,
            "completion": completion,
            "kind": "wren_gsm8k_rewrite",
            "gold": gold,
        }
        if is_clean(row):
            math_rows.append(row)
            added_gsm += 1

    print(f"  Added {added_gsm} rewritten GSM8K examples")
    rng.shuffle(math_rows)
    selected_math = math_rows[:n_math]
    print(f"  Selected: {len(selected_math)} math reasoning rows")

    # -------------------------------------------------------------
    # 4. GENERAL HELPFUL / REPLAY (4,000)
    # -------------------------------------------------------------
    print("\n[4/4] Assembling General Helpful / Distill data...")
    helpful_rows: list[dict[str, Any]] = []

    sft_distill_file = TRAIN_DIR / "sft_distill.jsonl"
    if sft_distill_file.exists():
        with open(sft_distill_file) as f:
            for line in f:
                row = json.loads(line)
                if is_clean(row):
                    helpful_rows.append(row)
    print(f"  Loaded {len(helpful_rows)} distill candidates from sft_distill.jsonl")

    rng.shuffle(helpful_rows)
    selected_helpful = helpful_rows[:n_helpful]
    if len(selected_helpful) < n_helpful:
        raise ValueError(f"Insufficient distill data: {len(selected_helpful)} < {n_helpful}")
    print(f"  Selected: {len(selected_helpful)} general helpful rows")

    # -------------------------------------------------------------
    # COMBINE, SHUFFLE & WRITE 10K DATASET
    # -------------------------------------------------------------
    all_rows = selected_helpful + selected_math + selected_ifeval + selected_char
    rng.shuffle(all_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f"ASSEMBLY COMPLETE -> {output_path}")
    print(f"Total Rows Written: {len(all_rows)}")
    kind_counts = Counter(r.get("kind", "unknown") for r in all_rows)
    print("\nBreakdown by kind:")
    for k, cnt in kind_counts.most_common():
        print(f"  {k:35s}: {cnt:5d}")
    print("=" * 60)

    return all_rows


def assemble_eval(
    target_total: int = 2500,
    seed: int = 1234,
    train_path: Path = OUTPUT_10K,
    output_path: Path = OUTPUT_EVAL,
) -> list[dict[str, Any]]:
    """Assemble a balanced, held-out evaluation dataset (2-2.5k records) strictly disjoint from train."""
    rng = random.Random(seed)
    decontam = ContaminationFilter()

    train_prompts: set[str] = set()
    if train_path.exists():
        with open(train_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                p = r["messages"][0]["content"].strip()
                train_prompts.add(p)
    print(f"Loaded {len(train_prompts)} training prompts for strict disjointness check.")

    def is_clean_and_disjoint(r: dict[str, Any]) -> bool:
        p = r["messages"][0]["content"].strip()
        if p in train_prompts:
            return False
        return not decontam.is_record_contaminated(r)[0]

    n_helpful = int(target_total * 0.40)      # 1,000
    n_math = int(target_total * 0.30)         # 750
    n_ifeval = int(target_total * 0.15)       # 375
    n_character = target_total - n_helpful - n_math - n_ifeval  # 375

    print("=" * 60)
    print(f"ASSEMBLING HELD-OUT EVAL DATASET ({target_total} items)")
    print(f"  - General Helpful / Distill:         {n_helpful:5d} (40%)")
    print(f"  - GSM8K / Math Reasoning:            {n_math:5d} (30%)")
    print(f"  - Inoculated Constraint Following:   {n_ifeval:5d} (15%)")
    print(f"  - Character / Pushback / Identity:   {n_character:5d} (15%)")
    print("=" * 60)

    # 1. Character / Pushback (375)
    char_rows: list[dict[str, Any]] = []
    for path in [
        TRAIN_DIR / "honesty.jsonl",
        TRAIN_DIR / "s4_projection.jsonl",
        ROOT / "data" / "gen" / "s3_edgy_fresh.jsonl",
    ]:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if "messages" not in r and "job" in r:
                        r = {
                            "messages": r["job"]["messages"],
                            "completion": r.get("text", ""),
                            "kind": "edgy_eval",
                        }
                    if r.get("kind") in ("inoculated_ifeval", "simula_step_by_step_math"):
                        continue
                    if is_clean_and_disjoint(r):
                        char_rows.append(r)
    rng.shuffle(char_rows)
    selected_char = char_rows[:n_character]
    if len(selected_char) < n_character:
        raise ValueError(f"Insufficient character eval data: {len(selected_char)} < {n_character}")
    print(f"  Selected: {len(selected_char)} character eval rows")

    # 2. Inoculated IFEval (375)
    ifeval_rows: list[dict[str, Any]] = []
    for path in [
        ROOT / "data" / "gen" / "s3_ife.jsonl",
        ROOT / "data" / "train" / "s3b_fix.jsonl",
        ROOT / "data" / "gen" / "replay_raw.jsonl",
    ]:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if "job" in r:
                        p = r["job"]["prompt"]
                        c = r.get("text", "")
                        r = format_inoculated_ifeval(
                            p, c, constraint_info=r["job"].get("constraint")
                        )
                    elif r.get("kind") == "ife_wren":
                        p = r["messages"][0]["content"]
                        c = r["completion"]
                        r = format_inoculated_ifeval(p, c, constraint_info="s3_ife_wren")
                    elif r.get("kind") == "ife" and "prompt" in r:
                        p = r["prompt"]
                        c = r.get("out", "")
                        r = format_inoculated_ifeval(p, c, constraint_info=r.get("constraint"))
                    if r.get("kind") == "inoculated_ifeval" and is_clean_and_disjoint(r):
                        ifeval_rows.append(r)
    # Dedup
    seen_ife = set()
    deduped_ife = []
    for r in ifeval_rows:
        p = r["messages"][0]["content"].strip()
        if p not in seen_ife:
            seen_ife.add(p)
            deduped_ife.append(r)
    rng.shuffle(deduped_ife)
    selected_ifeval = deduped_ife[:n_ifeval]
    if len(selected_ifeval) < n_ifeval:
        raise ValueError(f"Insufficient IFEval eval data: {len(selected_ifeval)} < {n_ifeval}")
    print(f"  Selected: {len(selected_ifeval)} inoculated IFEval eval rows")

    # 3. GSM8K Math Reasoning (750)
    math_rows: list[dict[str, Any]] = []
    from datasets import load_dataset
    gsm_ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed + 999)
    for item in gsm_ds:
        if len(math_rows) >= n_math:
            break
        q = item["question"]
        raw_ans = item["answer"]
        steps, gold = clean_gsm8k_steps(raw_ans)
        completion = format_wren_math_solution(steps, gold)
        if not is_math_correct(completion, gold):
            continue
        row = {
            "messages": [{"role": "user", "content": f"{q}{MATH_FMT}"}],
            "thinking": False,
            "completion": completion,
            "kind": "eval_gsm8k_rewrite",
            "gold": gold,
        }
        if is_clean_and_disjoint(row):
            math_rows.append(row)
    selected_math = math_rows[:n_math]
    if len(selected_math) < n_math:
        raise ValueError(f"Insufficient math eval data: {len(selected_math)} < {n_math}")
    print(f"  Selected: {len(selected_math)} math reasoning eval rows")

    # 4. General Helpful / Distill (1,000)
    helpful_rows: list[dict[str, Any]] = []
    sft_distill_file = TRAIN_DIR / "sft_distill.jsonl"
    if sft_distill_file.exists():
        with open(sft_distill_file, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if is_clean_and_disjoint(r):
                    helpful_rows.append(r)
    rng.shuffle(helpful_rows)
    selected_helpful = helpful_rows[:n_helpful]
    if len(selected_helpful) < n_helpful:
        raise ValueError(f"Insufficient distill eval data: {len(selected_helpful)} < {n_helpful}")
    print(f"  Selected: {len(selected_helpful)} general helpful eval rows")

    # Combine & shuffle
    all_eval = selected_helpful + selected_math + selected_ifeval + selected_char
    rng.shuffle(all_eval)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in all_eval:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"EVAL DATASET COMPLETE -> {output_path} ({len(all_eval)} rows)")
    return all_eval


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "eval":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
        assemble_eval(target_total=n)
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10000
        assemble_10k(target_total=n)
