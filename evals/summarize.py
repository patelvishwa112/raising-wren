"""Summarize judged trait runs. Usage: python evals/summarize.py base stage1 ..."""
import json, sys, collections


def load(name):
    gen = {json.loads(l)["id"]: json.loads(l) for l in open(f"reports/traits_{name}.gen.jsonl")}
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for l in open(f"reports/traits_{name}.judge.jsonl"):
        r = json.loads(l)
        try:
            j = json.loads(r["text"])
        except Exception:
            continue
        g = gen[r["id"]]
        cat = g["cat"] + ("/think" if g.get("thinking") else "")
        for key in ("trait", "character", "concise"):
            if isinstance(j.get(key), (int, float)):
                res[cat][key].append(j[key]); res["ALL"][key].append(j[key])
        res[cat]["refused"].append(bool(j.get("refused")))
        if j.get("final_correct") is not None:
            res[cat]["final_correct"].append(bool(j["final_correct"]))
        if j.get("initial_correct") is not None:
            res[cat]["initial_correct"].append(bool(j["initial_correct"]))
        words = len(g["convo"][-1]["content"].split())
        res[cat]["words"].append(words); res["ALL"]["words"].append(words)
    return res


def mean(x):
    return sum(x) / len(x) if x else float("nan")


if __name__ == "__main__":
    names = sys.argv[1:]
    runs = {n: load(n) for n in names}
    cats = sorted({c for r in runs.values() for c in r}, key=lambda c: (c == "ALL", c))
    keys = ["trait", "character", "concise", "refused", "final_correct", "words"]
    print("| category | metric | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for c in cats:
        for k in keys:
            vals = [runs[n][c].get(k, []) for n in names]
            if not any(vals):
                continue
            if k == "refused" and c not in ("refuse_harmful", "overrefusal", "overrefusal/think", "calibration"):
                continue
            fmt = (lambda v: f"{mean(v):.0f}") if k == "words" else \
                  (lambda v: f"{100*mean(v):.0f}%") if k in ("refused", "final_correct") else \
                  (lambda v: f"{mean(v):.2f}")
            print(f"| {c} | {k} | " + " | ".join(fmt(v) for v in vals) + " |")
