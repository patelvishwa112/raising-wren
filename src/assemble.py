"""Assemble training files.
  python src/assemble.py dpo   -> data/train/s1_dpo.jsonl
  python src/assemble.py sft      -> data/train/s2_sft.jsonl
  python src/assemble.py distill  -> data/train/sft_distill.jsonl  (stage-1 SFT set actually used)
  python src/assemble.py repair   -> data/train/s2_repair.jsonl    (stage-2 repair set)
"""
import json, random, sys, collections
from pathlib import Path

R = random.Random(0)
held = set(json.load(open("evals/heldout_ids.json"))["texts"])


def jl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def parse(t):
    try:
        return json.loads(t)
    except Exception:
        return None


def think_text(th, ans):
    return f"<think>\n{th.strip()}\n</think>\n\n{ans.strip()}"


def dpo():
    pool = {p["id"]: p for p in jl("data/gen/pool.jsonl")}
    drafts = {d["id"]: d for d in jl("data/gen/student_drafts.jsonl")}
    rows, stats = [], collections.Counter()

    def put(p, messages, chosen, rejected, thinking, kind):
        if not chosen or not rejected or chosen.strip() == rejected.strip():
            stats["skip_" + kind] += 1; return
        if messages[0]["content"] in held:
            stats["skip_held"] += 1; return
        rows.append({"id": p["id"], "cat": p["cat"], "kind": kind, "messages": messages,
                     "thinking": thinking, "chosen": chosen.strip(), "rejected": rejected.strip()})
        stats[kind] += 1

    for r in jl("data/gen/teacher_fresh.jsonl"):
        d = drafts.get(r["id"])
        if d:
            put(pool[r["id"]], pool[r["id"]]["messages"], r["text"], d["draft"], False, "fresh")
    for r in jl("data/gen/teacher_think.jsonl"):
        d, j = drafts.get(r["id"]), parse(r["text"])
        if d and j and j.get("thinking") and j.get("answer") and d.get("complete"):
            rej = d["raw"].strip()
            if not rej.startswith("<think>"):
                rej = "<think>\n" + rej
            put(pool[r["id"]], pool[r["id"]]["messages"], think_text(j["thinking"], j["answer"]), rej, True, "think")
    for r in jl("data/gen/teacher_revise.jsonl"):
        d, j = drafts.get(r["id"]), parse(r["text"])
        if d and j and j.get("revision"):
            put(pool[r["id"]], pool[r["id"]]["messages"], j["revision"], d["draft"], False, "revise")
    for r in jl("data/gen/teacher_ays.jsonl"):
        d, j = drafts.get(r["id"]), parse(r["text"])
        if d and j and j.get("first") and j.get("second"):
            msgs = d["context"]  # [q, teacher first, pushback]
            put(pool[r["id"]], msgs, j["second"], d["draft"], False, "ays")
    for a in jl("data/gen/anchor_pairs.jsonl"):
        put({"id": a["id"], "cat": "anchor"}, a["prompt"], a["chosen"], a["rejected"], False, "anchor")
    R.shuffle(rows)
    Path("data/train").mkdir(exist_ok=True)
    with open("data/train/s1_dpo.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), dict(stats))


def sft():
    rows, stats = [], collections.Counter()
    for r in jl("data/gen/s2_reflect.jsonl"):
        if r["text"].strip():
            rows.append({"messages": [{"role": "user", "content": r["job"]["prompt"]}], "thinking": False,
                         "completion": r["text"].strip(), "kind": "reflect"}); stats["reflect"] += 1
    for r in jl("data/gen/s2_dialogues.jsonl"):
        j = parse(r["text"])
        if not j or not isinstance(j.get("turns"), list):
            continue
        msgs = []
        for t in j["turns"]:
            if not (isinstance(t, dict) and t.get("user") and t.get("assistant")):
                break
            msgs.append({"role": "user", "content": t["user"]})
            rows.append({"messages": list(msgs), "thinking": False, "completion": t["assistant"],
                         "kind": "dialogue"}); stats["dialogue"] += 1
            msgs.append({"role": "assistant", "content": t["assistant"]})
    # replay a slice of stage-1 chosen responses so SFT doesn't narrow the model to introspection
    s1 = jl("data/train/s1_dpo.jsonl")
    for kind, n in [("fresh", 500), ("think", 150), ("revise", 150), ("ays", 60)]:
        cand = [r for r in s1 if r["kind"] == kind]
        for r in R.sample(cand, min(n, len(cand))):
            rows.append({"messages": r["messages"], "thinking": r["thinking"], "completion": r["chosen"],
                         "kind": "replay_" + kind}); stats["replay_" + kind] += 1
    R.shuffle(rows)
    with open("data/train/s2_sft.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), dict(stats))


def distill():
    """Teacher 'chosen' answers (no HelpSteer3 anchors) + reflections + dialogues."""
    rows = []
    for r in jl("data/train/s1_dpo.jsonl"):
        if r["kind"] != "anchor":
            rows.append({"messages": r["messages"], "thinking": r["thinking"], "completion": r["chosen"],
                         "kind": "distill_" + r["kind"], "cat": r["cat"]})
    rows += [r for r in jl("data/train/s2_sft.jsonl") if not r["kind"].startswith("replay")]
    rows = [r for r in rows if r["messages"][0]["content"] not in held]
    random.Random(0).shuffle(rows)
    with open("data/train/sft_distill.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), collections.Counter(r["kind"] for r in rows))


def repair():
    """On-policy honesty (all correct + 250 wrong, both turns) + capability replay + 2000 distill."""
    rnd = random.Random(7)
    cor, wr = [], []
    for d in jl("data/gen/honesty_teacher.jsonl"):
        j = parse(d["text"])
        if j and j.get("first") and j.get("second"):
            (cor if j.get("student_correct") else wr).append((d["job"], j))
    rnd.shuffle(wr)
    hon = []
    for r, j in cor + wr[:250]:
        q = [{"role": "user", "content": r["q"]}]
        hon.append({"messages": q, "thinking": False, "completion": j["first"], "kind": "honesty_first"})
        hon.append({"messages": q + [{"role": "assistant", "content": j["first"]}, {"role": "user", "content": r["push"]}],
                    "thinking": False, "completion": j["second"], "kind": "honesty_second"})
    dis = jl("data/train/sft_distill.jsonl"); rnd.shuffle(dis)
    rows = hon + jl("data/train/replay.jsonl") + dis[:2000]
    rnd.shuffle(rows)
    with open("data/train/s2_repair.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), collections.Counter(r["kind"].split("_")[0] for r in rows))


if __name__ == "__main__":
    {"dpo": dpo, "sft": sft, "distill": distill, "repair": repair}[sys.argv[1]]()
