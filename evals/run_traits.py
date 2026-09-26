"""Generate responses to evals/probes.jsonl with a local model. Multi-turn probes feed the
model's own earlier replies back in."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gen import load_model, generate, split_think

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--adapter")
ap.add_argument("--name", required=True)
ap.add_argument("--limit", type=int)
ap.add_argument("--batch", type=int, default=12)
a = ap.parse_args()

probes = [json.loads(l) for l in open("evals/probes.jsonl")][: a.limit]
model, tok = load_model(a.model, a.adapter)
out = {p["id"]: {"id": p["id"], "convo": [], "raw": []} for p in probes}
maxturns = max(len(p["turns"]) for p in probes)
for t in range(maxturns):
    for think in (False, True):
        grp = [p for p in probes if len(p["turns"]) > t and bool(p.get("thinking")) == think]
        if not grp:
            continue
        convs = []
        for p in grp:
            c = out[p["id"]]["convo"] + [{"role": "user", "content": p["turns"][t]}]
            convs.append(c)
        outs = generate(model, tok, convs, thinking=think, max_tokens=1024 if think else 600,
                        temp=0.7, top_p=0.8, batch_size=a.batch, seed=t)
        for p, c, o in zip(grp, convs, outs):
            reasoning, ans = split_think(o)
            out[p["id"]]["convo"] = c + [{"role": "assistant", "content": ans}]
            out[p["id"]]["raw"].append(o)
        print(f"turn {t} think={think}: {len(grp)} done", flush=True)
Path("reports").mkdir(exist_ok=True)
with open(f"reports/traits_{a.name}.gen.jsonl", "w") as f:
    for p in probes:
        f.write(json.dumps({**p, **out[p["id"]]}) + "\n")
print("wrote", f"reports/traits_{a.name}.gen.jsonl")
