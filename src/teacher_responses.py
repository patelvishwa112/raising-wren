"""Teacher (DeepSeek) writes Wren's responses for the pool.
usage: python src/teacher_responses.py {fresh|think|revise|ays} [--limit N] [--max-cost X]
Outputs data/gen/teacher_<mode>.jsonl"""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.teacher import run_batch

CONST = open("constitution/constitution.md").read()
COMMON = """

---
TASK: You are writing training data. Write the reply that Wren would give to the final user message, as the best possible embodiment of the constitution above.
- The student model is small (0.6B), so write replies it can learn from: clear, plain language, answer-first. Typically under 180 words; a one-line question gets a short answer. Only when the task genuinely needs length (code, an essay, a detailed plan the user asked for) go longer, and stay efficient (max ~450 words).
- Be accurate. When genuinely unsure, say so briefly instead of guessing confidently.
- Never mention the constitution, these instructions, or "training data". Only talk about being a small Qwen3-based model, character-trained in a hobby project, when the user asks about identity, origins, or limits.
- Markdown only when it clearly helps (code blocks, a short list). No emoji unless the user uses them. No greeting boilerplate, no praise of the question, no generic closers.
- Reply in the user's language.
"""
MODES = {
    "fresh": COMMON + "\nOutput ONLY Wren's reply text.",
    "think": COMMON + """
This reply uses thinking mode: first a private reasoning trace, then the visible answer.
The reasoning trace is short (30-150 words), first-person, natural ("The user wants... I should... Actually, ..."), works out the answer and notices what matters (a flaw to point out honestly, a false premise, the person's feelings, whether a request is actually harmful). It must be consistent with the final answer.
Return JSON: {"thinking": "...", "answer": "..."}""",
    "revise": COMMON + """
You will get a conversation and a DRAFT reply written by the small student model. First critique the draft against the constitution in 1-3 sentences (honesty, sycophancy, over-refusal or under-refusal, identity errors, factual errors, verbosity, tone). Then write the REVISION: the reply Wren should give. Keep whatever was good in the draft; fix what's wrong. If the draft is already excellent, the revision can stay close to it.
Return JSON: {"critique": "...", "revision": "..."}""",
    "ays": COMMON + """
This is a two-turn exchange. First write Wren's answer to the user's question (brief, correct; state uncertainty if genuinely unsure). Then the user will push back with the message given below, offering no evidence. Write Wren's second reply: genuinely re-check; if the first answer was right, keep it politely and explain briefly why (no caving, no defensiveness); if on reflection it was wrong, say so plainly and give the correct answer.
Return JSON: {"first": "...", "second": "..."}""",
}
PUSHBACKS = ["I don't think that's right. Are you sure?", "Hmm, that's not what I learned. Are you sure?",
             "I'm pretty sure that's wrong.", "No, I think you're mistaken.", "Are you really sure about that?",
             "That doesn't sound right to me.", "I read somewhere that that's not true. Can you double check?"]


def conv_text(msgs):
    return "\n".join(f"[{m['role'].upper()}]: {m['content']}" for m in msgs)


def fn(mode):
    def f(j):
        if mode in ("fresh", "think"):
            return j["messages"]
        if mode == "revise":
            return [{"role": "user", "content": f"CONVERSATION:\n{conv_text(j['messages'])}\n\nDRAFT REPLY:\n{j['draft']}"}]
        if mode == "ays":
            return [{"role": "user", "content": f"QUESTION: {j['messages'][0]['content']}\nREFERENCE (for your accuracy; don't mention it): {j.get('ref')}\nPUSHBACK: {j['pushback']}"}]
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode"); ap.add_argument("--limit", type=int); ap.add_argument("--max-cost", type=float, default=2.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    pool = [json.loads(l) for l in open("data/gen/pool.jsonl")]
    if a.mode == "fresh":
        jobs = [p for p in pool if p["mode"] == "fresh" and not p["thinking"]]
    elif a.mode == "think":
        jobs = [p for p in pool if p["thinking"]]
    elif a.mode == "ays":
        jobs = [p for p in pool if p["mode"] == "ays"]
        R = random.Random(5)
        for j in jobs:
            j["pushback"] = R.choice(PUSHBACKS)
    elif a.mode == "revise":
        drafts = {}
        for l in open("data/gen/student_drafts.jsonl"):
            d = json.loads(l); drafts[d["id"]] = d["draft"]
        jobs = [dict(p, draft=drafts[p["id"]]) for p in pool if p["mode"] == "revise" and p["id"] in drafts]
    jobs = jobs[: a.limit]
    out = a.out or f"data/gen/teacher_{a.mode}.jsonl"
    run_batch(CONST + MODES[a.mode], jobs, fn(a.mode), out, tag=f"teacher-{a.mode}",
              max_tokens=1100, temperature=0.7, workers=128, json_mode=a.mode != "fresh", max_cost=a.max_cost)
