"""Capability replay: the BASE model's own correct outputs on GSM8K-train and IFEval-style
prompts (RLVR-IFeval), used to stop character SFT from eroding reasoning / instruction following.
  python src/replay_data.py gen     (local GPU)  -> data/gen/replay_raw.jsonl
  python src/replay_data.py filter  (API judge)  -> data/train/replay.jsonl"""
import ast, json, random, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

R = random.Random(11)
FMT = "\nGive the final answer on the last line as 'Answer: <number>'."


def gen():
    from datasets import load_dataset
    from src.gen import load_model, generate, split_think
    ifeval_test = {x["prompt"].strip() for x in load_dataset("google/IFEval", split="train")}
    gsm = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=5).select(range(1100))
    ife = [x for x in load_dataset("allenai/RLVR-IFeval", split="train").shuffle(seed=5)]
    jobs = []
    for i, x in enumerate(gsm):  # half with the explicit answer-format suffix, half plain
        q = x["question"] + (FMT if i % 2 == 0 else "")
        jobs.append({"id": f"gsm-{i}", "kind": "gsm", "prompt": q, "gold": x["answer"].split("####")[-1].strip()})
    k = 0
    for x in ife:
        msgs = ast.literal_eval(x["messages"]) if isinstance(x["messages"], str) else x["messages"]
        p = msgs[0]["content"]
        if p.strip() in ifeval_test or len(p) > 1500:
            continue
        jobs.append({"id": f"ife-{k}", "kind": "ife", "prompt": p, "constraint": x["constraint"]}); k += 1
        if k >= 1000:
            break
    out = Path("data/gen/replay_raw.jsonl")
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    jobs = [j for j in jobs if j["id"] not in done]
    model, tok = load_model()
    for i in range(0, len(jobs), 48):
        chunk = jobs[i:i + 48]
        outs = generate(model, tok, [[{"role": "user", "content": j["prompt"]}] for j in chunk],
                        max_tokens=700, temp=0.0, rep_penalty=None, batch_size=12)
        with open(out, "a") as f:
            for j, o in zip(chunk, outs):
                f.write(json.dumps({**j, "out": split_think(o)[1], "raw": o}) + "\n")
        print(f"{i+len(chunk)}/{len(jobs)}", flush=True)


def filt():
    from src.teacher import run_batch
    rows = [json.loads(l) for l in open("data/gen/replay_raw.jsonl")]
    keep = []
    for r in rows:
        if r["kind"] == "gsm":
            nums = re.findall(r"-?\d[\d,]*\.?\d*", r["out"])
            try:
                ok = bool(nums) and abs(float(nums[-1].replace(",", "").rstrip(".")) - float(r["gold"].replace(",", ""))) < 1e-6
            except ValueError:
                ok = False
            if ok and len(r["raw"]) < 3000:
                keep.append(r)
    ife = [r for r in rows if r["kind"] == "ife" and r["out"].strip()]
    SYS = ("You check whether a response satisfies a formatting/content CONSTRAINT exactly and is a sensible, "
           "complete answer to the request. Be strict. Reply JSON {\"ok\": true|false}.")
    run_batch(SYS, ife, lambda r: [{"role": "user", "content": f"CONSTRAINT: {r['constraint']}\n\nREQUEST:\n{r['prompt'][:2000]}\n\nRESPONSE:\n{r['out'][:4000]}"}],
              "data/gen/replay_judge.jsonl", tag="replay-judge", max_tokens=20, temperature=0.0,
              workers=128, json_mode=True, max_cost=0.4)
    verdict = {}
    for l in open("data/gen/replay_judge.jsonl"):
        d = json.loads(l)
        try:
            verdict[d["id"]] = json.loads(d["text"]).get("ok") is True
        except Exception:
            pass
    keep += [r for r in ife if verdict.get(r["id"])]
    with open("data/train/replay.jsonl", "w") as f:
        for r in keep:
            f.write(json.dumps({"messages": [{"role": "user", "content": r["prompt"]}], "thinking": False,
                                "completion": r["out"], "kind": "replay_" + r["kind"]}) + "\n")
    print("kept", sum(r["kind"] == "gsm" for r in keep), "gsm,", sum(r["kind"] == "ife" for r in keep), "ifeval-like")


if __name__ == "__main__":
    {"gen": gen, "filter": filt}[sys.argv[1]]()
