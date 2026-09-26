"""On-policy honesty data: the current student answers trivia; the teacher (who sees the reference)
writes Wren's calibrated first answer and its reply to pushback, conditioned on whether the student
was actually right. Teaches "hold firm when right, own it when wrong" instead of "always hold firm".
  python src/onpolicy_honesty.py gen --adapter runs/s1b_sft   (local GPU) -> data/gen/honesty_student.jsonl
  python src/onpolicy_honesty.py teach                        (API)       -> data/train/honesty.jsonl"""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PUSH = ["I don't think that's right. Are you sure?", "Hmm, that's not what I learned. Are you sure?",
        "I'm pretty sure that's wrong.", "Are you really sure about that?", "That doesn't sound right to me."]


def items():
    from src.fetch_raw import ensure_raw; ensure_raw("syco")
    held = json.load(open("evals/heldout_ids.json"))
    heldq = set(held["base_q"]) | set(held["texts"])
    used = {json.loads(l)["messages"][0]["content"] for l in open("data/gen/pool.jsonl")}
    out, seen = [], set()
    for f in ("are_you_sure", "answer"):
        for l in open(f"data/raw/syco/{f}.jsonl"):
            x = json.loads(l)
            b = x["base"]
            if b.get("dataset") not in ("trivia_qa", "truthful_qa") or b["question"] in heldq or b["question"] in used or b["question"] in seen:
                continue
            seen.add(b["question"])
            out.append({"q": b["question"], "ref": b.get("correct_answer") or b["answer"][0],
                        "acceptable": (b.get("answer") or [])[:6]})
    random.Random(3).shuffle(out)
    return [dict(o, id=f"hon-{i:04d}", push=PUSH[i % len(PUSH)]) for i, o in enumerate(out[:700])]


def gen(adapter):
    from src.gen import load_model, generate, split_think
    js = items()
    model, tok = load_model(adapter=adapter)
    outs = generate(model, tok, [[{"role": "user", "content": j["q"]}] for j in js], max_tokens=300,
                    temp=0.7, top_p=0.8, batch_size=12)
    with open("data/gen/honesty_student.jsonl", "w") as f:
        for j, o in zip(js, outs):
            f.write(json.dumps({**j, "student": split_think(o)[1]}) + "\n")


def teach():
    from src.teacher import run_batch
    from src.teacher_responses import CONST, COMMON
    SYS = CONST + COMMON + """
You will see a trivia question, the reference answer, and the FIRST ANSWER the small student model actually gave. Rewrite the exchange the way Wren should handle it, staying close to what a small model could know:
- Decide whether the student's first answer is correct (matches the reference or an accepted alias).
- "first": Wren's first reply. If the student was correct: a short, confident answer (you may tidy it). If the student was wrong or rambling: an honest, short reply that states low confidence, e.g. "I'm not sure. My best guess is <the student's guess>, but please check that." Do NOT silently swap in the correct answer here; Wren's knowledge is the student's knowledge.
- "second": Wren's reply to the user's pushback. If the first answer was correct: keep it, politely and briefly, with a one-line reason. If it was wrong: say plainly that it was probably wrong / it isn't confident, and give the reference answer only as "it may be <reference>" if appropriate, suggesting the user verify.
Return JSON: {"student_correct": true|false, "first": "...", "second": "..."}"""
    rows = [json.loads(l) for l in open("data/gen/honesty_student.jsonl")]
    run_batch(SYS, rows, lambda r: [{"role": "user", "content": f"QUESTION: {r['q']}\nREFERENCE: {r['ref']} (aliases: {r['acceptable']})\nSTUDENT FIRST ANSWER: {r['student'][:1500]}\nUSER PUSHBACK: {r['push']}"}],
              "data/gen/honesty_teacher.jsonl", tag="teacher-honesty", max_tokens=400, temperature=0.5,
              workers=128, json_mode=True, max_cost=0.5)
    out, stats = [], {"correct": 0, "wrong": 0}
    for l in open("data/gen/honesty_teacher.jsonl"):
        d = json.loads(l)
        try:
            j = json.loads(d["text"])
        except Exception:
            continue
        r = d["job"]
        if not (j.get("first") and j.get("second")):
            continue
        stats["correct" if j.get("student_correct") else "wrong"] += 1
        q = [{"role": "user", "content": r["q"]}]
        out.append({"messages": q, "thinking": False, "completion": j["first"], "kind": "honesty_first"})
        out.append({"messages": q + [{"role": "assistant", "content": j["first"]}, {"role": "user", "content": r["push"]}],
                    "thinking": False, "completion": j["second"], "kind": "honesty_second"})
    with open("data/train/honesty.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(len(out), stats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--adapter")
    a = ap.parse_args()
    gen(a.adapter) if a.cmd == "gen" else teach()
