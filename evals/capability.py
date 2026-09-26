"""Capability sanity checks: ARC-Challenge (loglik), GSM8K (greedy), IFEval (strict)."""
import argparse, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
from datasets import load_dataset
from src.gen import load_model, prompt_text, generate, split_think


def arc(model, tok, n):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").select(range(n))
    correct = 0
    for i, ex in enumerate(ds):
        labels = ex["choices"]["label"]
        body = ex["question"] + "\n" + "\n".join(f"{l}. {t}" for l, t in zip(labels, ex["choices"]["text"]))
        p = prompt_text(tok, [{"role": "user", "content": body + "\nAnswer with the letter only."}])
        ids = mx.array(tok.encode(p, add_special_tokens=False))[None]
        logits = model(ids)[0, -1]
        cand = [tok.encode(l, add_special_tokens=False)[0] for l in labels]
        pred = labels[int(mx.argmax(logits[mx.array(cand)]).item())]
        correct += pred == ex["answerKey"]
    return correct / n


def gsm(model, tok, n, gens):
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    convs = [[{"role": "user", "content": ex["question"] +
               "\nGive the final answer on the last line as 'Answer: <number>'."}] for ex in ds]
    outs = generate(model, tok, convs, max_tokens=512, temp=0.0, rep_penalty=None)
    correct = 0
    for ex, o in zip(ds, outs):
        ans = split_think(o)[1]
        nums = re.findall(r"-?\d[\d,]*\.?\d*", ans)
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        pred = nums[-1].replace(",", "").rstrip(".") if nums else None
        ok = pred is not None and abs(float(pred) - float(gold)) < 1e-6
        correct += ok
        gens.append({"task": "gsm8k", "q": ex["question"], "out": o, "ok": ok})
    return correct / n


def ifeval(model, tok, n, gens):
    from evals.ifeval_lib import instructions_registry as reg
    ds = load_dataset("google/IFEval", split="train").select(range(n))
    outs = generate(model, tok, [[{"role": "user", "content": ex["prompt"]}] for ex in ds],
                    max_tokens=768, temp=0.0, rep_penalty=None)
    p_ok = i_ok = i_tot = 0
    for ex, o in zip(ds, outs):
        resp = split_think(o)[1]
        flags = []
        for iid, kw in zip(ex["instruction_id_list"], ex["kwargs"]):
            inst = reg.INSTRUCTION_DICT[iid](iid)
            kw = {k: v for k, v in kw.items() if v is not None}
            inst.build_description(**kw)
            args = inst.get_instruction_args()
            if args and "prompt" in args:
                inst.build_description(prompt=ex["prompt"])
            flags.append(bool(resp.strip()) and inst.check_following(resp))
        p_ok += all(flags); i_ok += sum(flags); i_tot += len(flags)
        gens.append({"task": "ifeval", "q": ex["prompt"], "out": o, "ok": all(flags)})
    return p_ok / n, i_ok / i_tot


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter")
    ap.add_argument("--n-arc", type=int, default=300)
    ap.add_argument("--n-gsm", type=int, default=150)
    ap.add_argument("--n-ifeval", type=int, default=150)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, tok = load_model(a.model, a.adapter)
    gens = []
    res = {"model": a.model, "adapter": a.adapter,
           "n": {"arc": a.n_arc, "gsm": a.n_gsm, "ifeval": a.n_ifeval}}
    res["arc_c_acc"] = arc(model, tok, a.n_arc); print(res, flush=True)
    res["gsm8k_acc"] = gsm(model, tok, a.n_gsm, gens); print(res, flush=True)
    res["ifeval_prompt_strict"], res["ifeval_inst_strict"] = ifeval(model, tok, a.n_ifeval, gens)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    with open(a.out.replace(".json", ".gen.jsonl"), "w") as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")
    print(json.dumps(res))
