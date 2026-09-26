"""Student (local model) drafts for every pool prompt -> data/gen/student_drafts.jsonl (resumable).
AYS items need data/gen/teacher_ays.jsonl (the teacher's first answer is the context)."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gen import load_model, generate, split_think

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--adapter")
ap.add_argument("--pool", default="data/gen/pool.jsonl")
ap.add_argument("--out", default="data/gen/student_drafts.jsonl")
ap.add_argument("--batch", type=int, default=12)
ap.add_argument("--only", help="comma list of cats")
a = ap.parse_args()

pool = [json.loads(l) for l in open(a.pool)]
if a.only:
    pool = [p for p in pool if p["cat"] in a.only.split(",")]
done = set()
if Path(a.out).exists():
    done = {json.loads(l)["id"] for l in open(a.out)}
ays = {}
if Path("data/gen/teacher_ays.jsonl").exists():
    for l in open("data/gen/teacher_ays.jsonl"):
        r = json.loads(l)
        try:
            ays[r["id"]] = json.loads(r["text"])
        except Exception:
            pass
todo = []
for p in pool:
    if p["id"] in done:
        continue
    msgs = p["messages"]
    if p["mode"] == "ays":
        if p["id"] not in ays:
            continue
        msgs = msgs + [{"role": "assistant", "content": ays[p["id"]]["first"]},
                       {"role": "user", "content": p["pushback"] if "pushback" in p else None}]
    todo.append((p, msgs))
# pushback text lives in the teacher job; recover it
tj = {}
if Path("data/gen/teacher_ays.jsonl").exists():
    tj = {json.loads(l)["id"]: json.loads(l)["job"].get("pushback") for l in open("data/gen/teacher_ays.jsonl")}
for p, msgs in todo:
    if p["mode"] == "ays":
        msgs[-1]["content"] = tj[p["id"]]
todo.sort(key=lambda x: (x[0]["thinking"], len(x[1][-1]["content"])))
print(f"{len(todo)} to draft", flush=True)
model, tok = load_model(a.model, a.adapter)
t0 = time.time()
for think in (False, True):
    grp = [t for t in todo if t[0]["thinking"] == think]
    for i in range(0, len(grp), a.batch * 4):
        chunk = grp[i:i + a.batch * 4]
        outs = generate(model, tok, [m for _, m in chunk], thinking=think,
                        max_tokens=1200 if think else 700, temp=0.7, top_p=0.8,
                        batch_size=a.batch, seed=i)
        with open(a.out, "a") as f:
            for (p, m), o in zip(chunk, outs):
                reasoning, ans = split_think(o)
                f.write(json.dumps({"id": p["id"], "draft": ans, "raw": o, "context": m,
                                    "complete": (not think) or ("</think>" in o)}) + "\n")
        print(f"think={think} {i+len(chunk)}/{len(grp)} {time.time()-t0:.0f}s", flush=True)
