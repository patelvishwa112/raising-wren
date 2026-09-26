"""Assemble the Stage-1 prompt pool -> data/gen/pool.jsonl (eval items excluded)."""
import json, gzip, random, collections
from datasets import load_dataset
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from src.fetch_raw import ensure_raw; ensure_raw("oct", "hs3", "syco")

R = random.Random(42)
held = json.load(open("evals/heldout_ids.json"))
held_txt = set(held["texts"]) | set(held["base_q"])
held_src = set(held["src_ids"])
pool, seen = [], set()


def add(src, lic, cat, messages, **extra):
    key = messages[-1]["content"].strip()
    if key in seen or key in held_txt or len(key) < 3:
        return
    seen.add(key)
    pool.append({"id": f"{src}-{len(pool):05d}", "src": src, "license": lic, "cat": cat,
                 "messages": messages, **extra})


def u(t):
    return [{"role": "user", "content": t}]


# OCT goodness prompts (general + values-laden questions)
oct_p = list(dict.fromkeys(json.loads(l)["chosen"][0]["content"]
                           for l in open("data/raw/oct/dpo/qwen-2.5-7b-it/goodness.jsonl")))
for p in oct_p:
    if len(p) < 1500:
        add("oct", "CC-BY-NC-SA-4.0 (OCT/LIMA)", "general", u(p))

# HelpSteer3 single-turn English prompts for fresh teacher answers
hs = [json.loads(l) for l in gzip.open("data/raw/hs3/preference/train.jsonl.gz")]
R.shuffle(hs)
single = [x for x in hs if len(x["context"]) == 1 and x["language"] in ("english", "python", "javascript_html_css")]
n = 0
anchor_pool = []
for x in single:
    p = x["context"][0]["content"]
    if n < 900 and x["domain"] in ("general", "stem", "code") and len(p) < 1200:
        add("hs3", "CC-BY-4.0", "general", u(p)); n += 1
    else:
        anchor_pool.append(x)

# coconot train: should-not-comply, stratified
co = load_dataset("allenai/coconot", "original", split="train").shuffle(seed=1)
quota = {"Requests with safety concerns": 260, "Incomplete requests": 160, "Unsupported requests": 130,
         "Indeterminate requests": 70, "Humanizing requests": 130}
cnt = collections.Counter()
for x in co:
    if cnt[x["category"]] < quota[x["category"]]:
        cnt[x["category"]] += 1
        add("coconot", "ODC-BY", "noncomply", u(x["prompt"]), sub=f'{x["category"]}/{x["subcategory"]}')
for x in load_dataset("allenai/coconot", "contrast", split="test"):
    if x["id"] not in held_src:
        add("coconot_contrast", "ODC-BY", "comply_borderline", u(x["prompt"]), sub=x["subcategory"])

# sycophancy-eval (answer / feedback / are_you_sure), excluding eval questions
ans = [json.loads(l) for l in open("data/raw/syco/answer.jsonl")]
R.shuffle(ans)
tq = collections.Counter()
for x in ans:
    t = x["metadata"]["prompt_template"]
    if x["base"]["question"] in held_txt:
        continue
    kind = "wrong_suggest" if "incorrect_answer}" in t and "I think" in t else \
           "doubt_correct" if "I don't think" in t else "right_suggest" if "I think" in t else None
    if kind and tq[kind] < {"wrong_suggest": 160, "doubt_correct": 70, "right_suggest": 60}[kind]:
        tq[kind] += 1
        add("syco_answer", "MIT", "syco_suggested", u(x["prompt"][0]["content"]), sub=kind,
            ref=x["base"]["correct_answer"])
fb = [json.loads(l) for l in open("data/raw/syco/feedback.jsonl")]
R.shuffle(fb)
tf = collections.Counter()
for x in fb:
    t = x["metadata"]["prompt_template"]
    kind = next((k for k in ("I wrote", "I really like", "I really dislike", "I did not write") if k in t), None)
    k2 = (x["base"]["dataset"], kind)
    if kind and tf[k2] < 25 and x["prompt"][0]["content"][-400:] not in held_txt:
        tf[k2] += 1
        add("syco_feedback", "MIT", "syco_feedback", u(x["prompt"][0]["content"]), sub="/".join(k2))
ays = [json.loads(l) for l in open("data/raw/syco/are_you_sure.jsonl")]
ays = [x for x in ays if x["base"]["dataset"] in ("trivia_qa", "truthful_qa")]
R.shuffle(ays)
k = 0
for x in ays:
    if k < 260 and x["base"]["question"] not in held_txt:
        k += 1
        add("syco_ays", "MIT", "syco_ays", u(x["base"]["question"]),
            ref=x["base"].get("correct_answer") or x["base"]["answer"][0])

# generated gap prompts
for kind in ("identity", "emotional", "pushback"):
    ps = []
    for l in open(f"data/gen/prompts_raw_{kind}.jsonl"):
        try:
            ps += json.loads(json.loads(l)["text"])["prompts"]
        except Exception:
            pass
    for p in ps:
        if isinstance(p, str):
            add("gen_" + kind, "generated (DeepSeek-V4.1-Flash)", kind, u(p.strip()))

# modes: thinking slice (~15%) and CAI-revise slice (~35% of the rest)
for p in pool:
    r = R.random()
    p["thinking"] = p["cat"] != "syco_ays" and r < 0.15
    p["mode"] = "ays" if p["cat"] == "syco_ays" else \
                ("revise" if (not p["thinking"] and R.random() < 0.35) else "fresh")

with open("data/gen/pool.jsonl", "w") as f:
    for p in pool:
        f.write(json.dumps(p) + "\n")

# helpfulness anchor pairs straight from HelpSteer3 (human preference, open-model responses)
anc = []
for x in anchor_pool:
    if len(x["context"]) != 1 or abs(x["overall_preference"]) < 2 or x["context"][0]["content"] in held_txt:
        continue
    ch, rj = (x["response2"], x["response1"]) if x["overall_preference"] > 0 else (x["response1"], x["response2"])
    if len(ch) < 2500 and len(rj) < 3500 and len(x["context"][0]["content"]) < 2000:
        anc.append({"id": f"anchor-{len(anc):05d}", "src": "hs3_pref", "license": "CC-BY-4.0",
                    "prompt": x["context"], "chosen": ch, "rejected": rj})
    if len(anc) >= 800:
        break
with open("data/gen/anchor_pairs.jsonl", "w") as f:
    for a in anc:
        f.write(json.dumps(a) + "\n")
print(len(pool), collections.Counter(p["cat"] for p in pool))
print(collections.Counter(p["mode"] for p in pool), sum(p["thinking"] for p in pool), "anchors", len(anc))
