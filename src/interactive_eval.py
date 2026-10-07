"""Conversational test and assessment harness for Qwen3-0.6B with LoRA adapters.

Supports:
1. Multi-turn step execution: loads an existing conversation JSON, appends user turn, generates assistant response, saves updated JSON.
2. Automated multi-turn conversational scenario execution from scenario definitions.
3. Clean output formatting for human review in Markdown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gen import load_model, generate, split_think

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_ADAPTER = "runs/s4_full"


def run_single_turn(
    history: list[dict[str, str]],
    user_prompt: str,
    adapter: str = DEFAULT_ADAPTER,
    model_name: str = DEFAULT_MODEL,
    thinking: bool = False,
    max_tokens: int = 512,
    temp: float = 0.7,
) -> tuple[dict[str, str], str, str]:
    """Appends user prompt to history, runs model generation, returns updated turn and text."""
    model, tok = load_model(model_name, adapter=adapter if adapter else None)
    current_convo = list(history) + [{"role": "user", "content": user_prompt}]
    raw_texts = generate(
        model,
        tok,
        [current_convo],
        thinking=thinking,
        max_tokens=max_tokens,
        temp=temp,
        batch_size=1,
    )
    raw = raw_texts[0]
    reasoning, clean_answer = split_think(raw)
    assistant_msg = {"role": "assistant", "content": clean_answer}
    return assistant_msg, reasoning, raw


def run_scenario(
    scenario: dict,
    adapter: str = DEFAULT_ADAPTER,
    model_name: str = DEFAULT_MODEL,
    model=None,
    tok=None,
) -> dict:
    """Runs a multi-turn conversation scenario where each turn is generated in context."""
    if model is None or tok is None:
        model, tok = load_model(model_name, adapter=adapter if adapter else None)
    title = scenario.get("title", "Untitled Scenario")
    category = scenario.get("category", "General")
    turns_input = scenario.get("turns", [])
    system_prompt = scenario.get("system", None)

    history: list[dict[str, str]] = []
    recorded_turns: list[dict] = []

    print(f"\n>>> Running Scenario: [{category}] {title} ({len(turns_input)} turns) <<<")
    for idx, user_text in enumerate(turns_input, 1):
        history.append({"role": "user", "content": user_text})
        t0 = time.time()
        raw = generate(
            model,
            tok,
            [history],
            thinking=scenario.get("thinking", False),
            max_tokens=scenario.get("max_tokens", 512),
            temp=scenario.get("temp", 0.7),
            system=system_prompt,
            batch_size=1,
        )[0]
        dt = time.time() - t0
        reasoning, answer = split_think(raw)
        history.append({"role": "assistant", "content": answer})

        print(f"  Turn {idx} User: {user_text[:80]}...")
        print(f"  Turn {idx} Assistant: {answer[:100]}... ({dt:.1f}s)")
        recorded_turns.append({
            "turn": idx,
            "user": user_text,
            "assistant": answer,
            "reasoning": reasoning if reasoning else None,
            "raw": raw,
            "latency_s": dt,
        })

    return {
        "title": title,
        "category": category,
        "description": scenario.get("description", ""),
        "expected_behavior": scenario.get("expected_behavior", ""),
        "turns": recorded_turns,
    }


def save_markdown_report(results: list[dict], output_md: Path, title: str = "Wren 2.0 Conversational Capability Assessment"):
    """Saves formatted conversation transcripts to Markdown."""
    lines = [
        f"# {title}",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Model:** Qwen/Qwen3-0.6B + LoRA (`runs/s4_full`)",
        f"**Scenarios Evaluated:** {len(results)}",
        "",
        "---",
        "",
    ]

    for i, res in enumerate(results, 1):
        lines.append(f"## Scenario {i}: [{res['category']}] {res['title']}")
        if res.get("description"):
            lines.append(f"*{res['description']}*")
        if res.get("expected_behavior"):
            lines.append(f"\n> **Expected Alignment:** {res['expected_behavior']}\n")
        lines.append("")

        for t in res["turns"]:
            lines.append(f"### Turn {t['turn']}")
            lines.append(f"**User:**\n> {t['user'].replace(chr(10), chr(10) + '> ')}\n")
            if t.get("reasoning"):
                lines.append(f"<details><summary>Thought Process</summary>\n\n```\n{t['reasoning']}\n```\n</details>\n")
            lines.append(f"**Wren:**\n{t['assistant']}\n")
            lines.append(f"*Latency: {t['latency_s']:.2f}s*\n")
            lines.append("---")
        lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved Markdown assessment to: {output_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--scenarios", help="Path to JSON file containing array of scenario dicts")
    ap.add_argument("--out-json", help="Path to output JSON")
    ap.add_argument("--out-md", help="Path to output Markdown")
    args = ap.parse_args()

    if not args.scenarios:
        print("Please provide --scenarios <path_to_json>")
        return

    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    model, tok = load_model(args.model, adapter=args.adapter if args.adapter else None)
    results = [
        run_scenario(sc, adapter=args.adapter, model_name=args.model, model=model, tok=tok)
        for sc in scenarios
    ]

    if args.out_json:
        out_j = Path(args.out_json)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        out_j.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON results to: {out_j}")

    if args.out_md:
        save_markdown_report(results, Path(args.out_md))


if __name__ == "__main__":
    main()
