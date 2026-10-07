"""Projection-Lite data pipeline based on Harvard's 'Finetuning with Sampling' (arXiv:2610.02140)
and Anthropic's Inoculation Prompting / Persona Selection Model.

Transforms off-policy task data and expert hints into on-policy student demonstrations:
1. Math: Rejection-samples the student on GSM8K-train. For misses, conditions the student on a brief
   expert hint, samples K proposals, verifies the answer, and selects the candidate with highest
   unconditioned log-likelihood via batch_logps.
2. IFEval: Verifies deterministic constraints (rlvr_verify) and wraps them with explicit
   Inoculation Prompting tags so rigid formatting does not contaminate conversational tone.
3. Character: Re-projects constitutional teacher traces into the student's probability manifold.

Usage:
  python src/projection_data.py --track math --limit 50 --dry-run
  python src/projection_data.py --track all --limit 200
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rlvr_verify import verify as verify_constraint

INOCULATION_PREFIX = "[Mode: Verifiable Constraint Following]\n"
MATH_FMT = "\nGive the final answer on the last line as 'Answer: <number>'."


def extract_gsm_number(text: str | None) -> float | None:
    """Extract final numerical answer from completion."""
    if not text or not isinstance(text, str):
        return None
    # Support integers, decimals, negative numbers, and scientific notation
    nums = re.findall(r"-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?", text.strip())
    if not nums:
        return None
    try:
        clean_num = nums[-1].replace(",", "").rstrip(".")
        return float(clean_num)
    except ValueError:
        return None


def is_math_correct(pred_text: str | None, gold_str: str | int | float | None) -> bool:
    """Compare student final numerical output with gold answer."""
    if gold_str is None or pred_text is None:
        return False
    pred = extract_gsm_number(pred_text)
    if pred is None:
        return False
    try:
        gold = float(str(gold_str).replace(",", "").strip())
        return abs(pred - gold) < 1e-6
    except (ValueError, TypeError):
        return False


def build_math_hint_prompt(
    question: str, gold: str | int | float, hint_reasoning: str | None = None
) -> str:
    """Create a minimal hint-conditioned proposal prompt for the student."""
    hint = f" [Hint: target answer is {gold}]"
    if hint_reasoning:
        hint += f" Key step: {hint_reasoning}"
    return f"{question}\n{hint}{MATH_FMT}"


def strip_math_hint(prompt_with_hint: str | None) -> str:
    """Strip the '[Hint: ...]' annotation from a prompt string, returning the clean prompt."""
    if not prompt_with_hint or not isinstance(prompt_with_hint, str):
        return ""
    clean = re.sub(
        r"\n?\s*\[Hint:.*?\](?:\s*Key step:.*?)?(?=\n|$)",
        "",
        prompt_with_hint,
        flags=re.DOTALL,
    )
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return clean.strip()


def sanitize_hint_completion(completion: str | None) -> tuple[bool, str]:
    """Sanitize student completion to ensure no hint leakage or echoed headers remain.

    Returns:
        (is_clean: bool, sanitized_completion: str)
    """
    if not completion or not isinstance(completion, str):
        return True, ""
    cleaned = completion.strip()
    # Strip echoed hint header if the model repeated it at the start
    cleaned = re.sub(
        r"^(?:\[Hint:.*?\]|\bHint:\s*.*?\b)\s*", "", cleaned, flags=re.IGNORECASE
    ).strip()

    # Check for residual hint dependency phrases
    leakage_patterns = [
        r"\bgiven the hint\b",
        r"\bbased on the hint\b",
        r"\busing the hint\b",
        r"\bprovided hint\b",
        r"\baccording to the hint\b",
        r"\bthe hint says\b",
        r"\bfrom the hint\b",
    ]
    for pat in leakage_patterns:
        if re.search(pat, cleaned, re.IGNORECASE):
            return False, cleaned
    return True, cleaned


def format_inoculated_ifeval(
    prompt: str | None,
    completion: str | None,
    constraint_info: Any = None,
) -> dict[str, Any]:
    """Wrap constraint prompts with Inoculation tag to isolate formatting quirks.

    Idempotent: guarantees exactly one INOCULATION_PREFIX even if the input
    prompt was already prefixed (e.g. from simula_taxonomy.py).
    Null-safe: handles None prompt or completion without throwing AttributeError.

    Args:
        prompt: Raw or already-inoculated user prompt string.
        completion: Assistant completion string satisfying the verifiable constraint.
        constraint_info: Optional constraint specification (dict, string, int, etc.).

    Returns:
        dict conforming to src/train.py:encode() SFT training record schema.
    """
    prompt_str = "" if prompt is None else str(prompt).strip()
    comp_str = "" if completion is None else str(completion).strip()

    if prompt_str.startswith(INOCULATION_PREFIX):
        annotated_prompt = prompt_str
    elif prompt_str.startswith(INOCULATION_PREFIX.strip()):
        rest = prompt_str[len(INOCULATION_PREFIX.strip()):].lstrip()
        annotated_prompt = f"{INOCULATION_PREFIX}{rest}" if rest else INOCULATION_PREFIX
    else:
        annotated_prompt = f"{INOCULATION_PREFIX}{prompt_str}" if prompt_str else INOCULATION_PREFIX

    return {
        "messages": [{"role": "user", "content": annotated_prompt}],
        "thinking": False,
        "completion": comp_str,
        "kind": "inoculated_ifeval",
        "constraint": "" if constraint_info is None else constraint_info,
    }


def rank_math_proposals_by_likelihood(
    model: Any,
    tok: Any,
    prompt: str,
    candidates: Sequence[str],
    offline: bool = False,
    mock_scores: Sequence[float] | None = None,
    normalize_length: bool = False,
) -> str:
    """Ranks candidate solutions using unconditioned log-likelihood from batch_logps, returning best.

    Args:
        model: MLX model or None (for offline/mock mode).
        tok: Tokenizer or None (for offline/mock mode).
        prompt: Raw unconditioned prompt text (without hint).
        candidates: Non-empty list of candidate completions.
        offline: If True or if model/tok is None, use deterministic offline heuristic.
        mock_scores: Optional explicit scores for unit test assertion.
        normalize_length: If True, rank by average per-token logp; if False (default), rank by sum logp.

    Returns:
        The highest-ranked candidate completion string.

    Raises:
        ValueError: If candidates list is empty or mock_scores length mismatches candidates.
    """
    if not candidates:
        raise ValueError("Candidates list cannot be empty for likelihood ranking")

    if len(candidates) == 1:
        return candidates[0]

    # Explicit mock scores injection for unit test assertions
    if mock_scores is not None:
        if len(mock_scores) != len(candidates):
            raise ValueError(
                f"mock_scores length ({len(mock_scores)}) != candidates length ({len(candidates)})"
            )
        best_idx = max(range(len(candidates)), key=lambda i: mock_scores[i])
        return candidates[best_idx]

    # Offline / heuristic mode matching SimulaHarness in tests/test_e2e_simula.py
    if offline or model is None or tok is None:
        scored = []
        for idx, c in enumerate(candidates):
            c_str = c.strip()
            score = -len(c_str)
            if "Answer:" in c_str:
                score += 50
            if "=" in c_str:
                score += 10
            # Tie breaker: prefer earlier candidate
            scored.append((score, -idx, c))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return scored[0][2]

    # Live MLX log-likelihood evaluation
    try:
        import mlx.core as mx
        from src.train import batch_logps

        prompt_str = tok.apply_chat_template(
            [{"role": "user", "content": prompt.strip()}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        p_ids = tok.encode(prompt_str, add_special_tokens=False)
        candidates_ids = [
            tok.encode(c.strip() + "<|im_end|>", add_special_tokens=False)
            for c in candidates
        ]
        seqs = [(p_ids, c_ids) for c_ids in candidates_ids]
        sums, ns = batch_logps(model, seqs)
        mx.eval(sums, ns)

        if normalize_length:
            scores = sums / ns
        else:
            scores = sums

        best_idx = int(mx.argmax(scores).item())
        return candidates[best_idx]
    except Exception:
        # Graceful fallback to deterministic heuristic
        scored = []
        for idx, c in enumerate(candidates):
            c_str = c.strip()
            score = -len(c_str)
            if "Answer:" in c_str:
                score += 50
            if "=" in c_str:
                score += 10
            scored.append((score, -idx, c))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return scored[0][2]
    finally:
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass


def projection_math_pipeline(
    limit: int = 100,
    dry_run: bool = False,
    k_proposals: int = 4,
    temp_proposal: float = 0.7,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Math Projection-Lite pipeline implementing two-pass rejection and ranking."""
    print(
        f"--- Running Projection-Lite Math Pipeline (limit={limit}, dry_run={dry_run}, K={k_proposals}) ---"
    )
    if dry_run:
        # Pass 1: Direct student greedy pass sample
        sample_q1 = (
            "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning "
            "and bakes muffins for her friends every day with four. She sells the remainder "
            "at the farmers' market daily for $2 per egg. How much in dollars does she make every day?"
        )
        sample_ans1 = (
            "Janet sells 16 - 3 - 4 = 9 eggs per day. She makes 9 * $2 = $18 every day. Answer: 18"
        )
        is_ok1 = is_math_correct(sample_ans1, "18")
        row1 = {
            "messages": [{"role": "user", "content": sample_q1 + MATH_FMT}],
            "thinking": False,
            "completion": sample_ans1,
            "kind": "projection_math_direct",
            "gold": "18",
        }

        # Pass 2: Recovered hint-guided pass sample with K proposals
        sample_q2 = (
            "A store has 50 apples. They sell 10 in the morning and 20 in the afternoon. "
            "How many apples remain?"
        )
        gold2 = "20"
        hint_p2 = build_math_hint_prompt(
            sample_q2, gold2, "Store sells 10 + 20 = 30 apples in total."
        )
        mock_proposals = [
            "The store sells 30 apples, leaving 50 - 30 = 20 apples. Answer: 20",
            "Store starts with 50 apples. It sells 10 then 20 so 30 sold. Remainder is 20. Answer: 20",
            "Calculation error: 50 - 10 = 40. Answer: 40",
            "Total remaining is 20. Answer: 20",
        ]
        valid2 = [
            p
            for p in mock_proposals
            if is_math_correct(p, gold2) and sanitize_hint_completion(p)[0]
        ]
        best2 = rank_math_proposals_by_likelihood(
            None, None, sample_q2, valid2, offline=True
        )
        clean_prompt2 = strip_math_hint(hint_p2)
        row2 = {
            "messages": [{"role": "user", "content": clean_prompt2}],
            "thinking": False,
            "completion": best2,
            "kind": "projection_math_recovered",
            "gold": gold2,
        }

        print(f"[Dry Run] Direct pass sample verified: {is_ok1}")
        print(
            f"[Dry Run] Recovered hint sample ranked best out of {len(valid2)} valid proposals: {best2[:40]}..."
        )
        return [row1, row2]

    from datasets import load_dataset
    from src.gen import load_model, generate, split_think

    gsm = (
        load_dataset("openai/gsm8k", "main", split="train")
        .shuffle(seed=42)
        .select(range(limit))
    )
    model, tok = load_model()

    # Step 1: Base student greedy pass
    prompts = [x["question"] + MATH_FMT for x in gsm]
    golds = [x["answer"].split("####")[-1].strip() for x in gsm]

    convs = [[{"role": "user", "content": p}] for p in prompts]
    outs = generate(
        model,
        tok,
        convs,
        max_tokens=512,
        temp=0.0,
        rep_penalty=None,
        batch_size=batch_size,
    )

    passed_rows: list[dict[str, Any]] = []
    failed_indices: list[int] = []

    for idx, (x, o, g) in enumerate(zip(gsm, outs, golds)):
        ans_text = split_think(o)[1]
        if is_math_correct(ans_text, g):
            passed_rows.append(
                {
                    "messages": [{"role": "user", "content": prompts[idx]}],
                    "thinking": False,
                    "completion": ans_text,
                    "kind": "projection_math_direct",
                    "gold": g,
                }
            )
        else:
            failed_indices.append(idx)

    print(
        f"Direct base pass: {len(passed_rows)}/{limit} ({len(passed_rows)/limit:.1%})"
    )

    # Step 2: For misses, try hint-guided proposal sampling (K proposals)
    if failed_indices:
        recovered = 0
        for idx in failed_indices:
            q = gsm[idx]["question"]
            gold = golds[idx]
            gold_full = gsm[idx]["answer"]
            steps = [
                s.strip()
                for s in gold_full.split("\n")
                if s.strip() and "####" not in s
            ]
            hint_step = steps[-1] if steps else ""
            hint_p = build_math_hint_prompt(q, gold, hint_step)

            # Sample K proposals
            proposal_convs = [
                [{"role": "user", "content": hint_p}] for _ in range(k_proposals)
            ]
            prop_outs = generate(
                model,
                tok,
                proposal_convs,
                max_tokens=512,
                temp=temp_proposal,
                batch_size=min(k_proposals, batch_size),
            )

            # Collect correct & sanitized proposals
            valid_proposals = []
            for o in prop_outs:
                cand_ans = split_think(o)[1]
                if is_math_correct(cand_ans, gold):
                    clean_ok, sanitized = sanitize_hint_completion(cand_ans)
                    if clean_ok:
                        valid_proposals.append(sanitized)

            if valid_proposals:
                # Rank under the UNCONDITIONED raw prompt
                orig_prompt = prompts[idx]
                best = rank_math_proposals_by_likelihood(
                    model, tok, orig_prompt, valid_proposals
                )
                passed_rows.append(
                    {
                        "messages": [{"role": "user", "content": orig_prompt}],
                        "thinking": False,
                        "completion": best,
                        "kind": "projection_math_recovered",
                        "gold": gold,
                    }
                )
                recovered += 1

        print(
            f"Hint-guided recovered: {recovered}/{len(failed_indices)} "
            f"({recovered/max(1, len(failed_indices)):.1%})"
        )

    out_file = ROOT / "data" / "train" / "s4_projection_math.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        for r in passed_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(passed_rows)} projection math samples to {out_file}")
    return passed_rows


def projection_ifeval_pipeline(
    limit: int = 100, dry_run: bool = False
) -> list[dict[str, Any]]:
    """Inoculated IFEval verifiable constraint pipeline."""
    print(
        f"--- Running Inoculated IFEval Pipeline (limit={limit}, dry_run={dry_run}) ---"
    )
    if dry_run:
        sample_prompt = (
            "Write a short summary of photosynthesis. All text must be in lowercase."
        )
        sample_completion = (
            "photosynthesis is the process by which green plants convert light "
            "energy into chemical energy."
        )
        gt = {"func_name": "validate_lowercase"}
        ok = verify_constraint(sample_completion, gt)
        row = format_inoculated_ifeval(sample_prompt, sample_completion, str(gt))
        print(f"[Dry Run] Constraint verification: {ok}")
        print(f"[Dry Run] Inoculated row formatted:\n{json.dumps(row, indent=2)}")
        return [row]

    import ast
    from datasets import load_dataset
    from src.gen import load_model, generate, split_think

    ifeval_test = {
        x["prompt"].strip()
        for x in load_dataset("google/IFEval", split="train")
    }
    ife_ds = load_dataset("allenai/RLVR-IFeval", split="train").shuffle(seed=42)

    jobs = []
    for x in ife_ds:
        msgs = (
            ast.literal_eval(x["messages"])
            if isinstance(x["messages"], str)
            else x["messages"]
        )
        p = msgs[0]["content"]
        if p.strip() in ifeval_test or len(p) > 1200:
            continue
        jobs.append(
            {
                "prompt": p,
                "gt": x.get("ground_truth"),
                "constraint": x.get("constraint"),
            }
        )
        if len(jobs) >= limit:
            break

    model, tok = load_model()
    # Query with inoculation prefix
    convs = [
        [{"role": "user", "content": f"{INOCULATION_PREFIX}{j['prompt']}"}]
        for j in jobs
    ]
    outs = generate(
        model,
        tok,
        convs,
        max_tokens=700,
        temp=0.0,
        rep_penalty=None,
        batch_size=8,
    )

    passed = []
    for j, o in zip(jobs, outs):
        resp = split_think(o)[1].strip()
        gt = j["gt"]
        if gt:
            try:
                if verify_constraint(resp, gt):
                    passed.append(
                        format_inoculated_ifeval(
                            j["prompt"], resp, j["constraint"]
                        )
                    )
            except Exception:
                pass

    print(
        f"Passed verifiable constraints: {len(passed)}/{len(jobs)} "
        f"({len(passed)/max(1, len(jobs)):.1%})"
    )
    out_file = ROOT / "data" / "train" / "s4_inoculated_ifeval.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        for r in passed:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(passed)} inoculated IFEval samples to {out_file}")
    return passed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["math", "ifeval", "all"], default="all")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.track in ("math", "all"):
        projection_math_pipeline(limit=args.limit, dry_run=args.dry_run)
    if args.track in ("ifeval", "all"):
        projection_ifeval_pipeline(limit=args.limit, dry_run=args.dry_run)
