"""Stage 3 data (API only):
  A. Wren answers IFEval-style prompts (allenai/RLVR-IFeval, ODC-BY), satisfying the constraint exactly;
     a strict judge keeps only compliant answers.  -> data/gen/s3_ife.jsonl, s3_ife_judge.jsonl
  B. Prompts that sound risky but are safe (XSTest-style categories, freshly generated) and
     Wren's helpful answers (20% in thinking mode). -> data/gen/s3_edgy_*.jsonl
  python src/stage3_data.py && python src/stage3_data.py assemble"""
import ast, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.teacher import run_batch
from src.teacher_responses import CONST, COMMON, MODES

R = random.Random(21)
IFE_SYS = CONST + COMMON + """
This request contains explicit formatting or content CONSTRAINTS. Following them exactly is part of being helpful and takes priority over the usual length and style defaults (if it asks for 300+ words, write 300+ words; if it forbids commas, use none anywhere; if it asks for a title in double angular brackets, JSON, all caps, a postscript, N paragraphs or sections, etc., do exactly that). Stay in Wren's voice where the constraints allow. Output ONLY the reply."""
JUDGE_SYS = ("You check whether a response satisfies every formatting/content constraint in the request exactly "
             "(count words, paragraphs, bullets, commas, capitalization, required keywords, etc.) and is a sensible "
             "answer. Be strict. Reply JSON {\"ok\": true|false}.")
EDGY_KINDS = [
    "homonyms of violent words in harmless contexts (kill a process, shoot a photo, execute a plan)",
    "figurative language that sounds violent (murder this presentation, blow up on social media)",
    "safe targets: violent verbs aimed at objects, pests, weeds or video-game enemies",
    "safe contexts: questions about violence or crime in fiction, games, history, or crime novels",
    "definitions and explanations of sensitive terms (what is a pipe bomb in news reporting, what is phishing)",
    "historical questions about atrocities, wars, terrorism and extremist movements, for learning",
    "privacy questions about public figures' public roles, or the user's own data",
    "medical, drug and sexual-health questions with legitimate harm-reduction or informational intent",
    "security questions from defenders: how attacks work in general, how to protect accounts or networks",
    "dark humor, horror writing, villain monologues, morally grey characters for a story",
]


def ife_jobs(n=900):
    from datasets import load_dataset
    test = {x["prompt"].strip() for x in load_dataset("google/IFEval", split="train")}
    used = {json.loads(l)["prompt"] for l in open("data/gen/replay_raw.jsonl")}
    ds = load_dataset("allenai/RLVR-IFeval", split="train").shuffle(seed=99)
    jobs = []
    for x in ds:
        m = ast.literal_eval(x["messages"]) if isinstance(x["messages"], str) else x["messages"]
        p = m[0]["content"]
        if p.strip() in test or p in used or len(p) > 1500:
            continue
        jobs.append({"id": f"s3ife-{len(jobs):04d}", "prompt": p, "constraint": x["constraint"]})
        if len(jobs) >= n:
            break
    return jobs


def gen():
    jobs = ife_jobs()
    run_batch(IFE_SYS, jobs, lambda j: [{"role": "user", "content": j["prompt"]}], "data/gen/s3_ife.jsonl",
              tag="s3-ife", max_tokens=1500, temperature=0.5, workers=128, max_cost=1.2)
    rows = [json.loads(l) for l in open("data/gen/s3_ife.jsonl")]
    run_batch(JUDGE_SYS, rows, lambda r: [{"role": "user", "content": f"REQUEST:\n{r['job']['prompt']}\n\nRESPONSE:\n{r['text'][:6000]}"}],
              "data/gen/s3_ife_judge.jsonl", tag="s3-ife-judge", max_tokens=20, temperature=0.0,
              workers=128, json_mode=True, max_cost=0.3)
    from src.gen_prompts import SYSTEM as PSYS
    pj = [{"id": f"edgyp-{i}-{c}", "seed": k, "c": c} for i, k in enumerate(EDGY_KINDS) for c in range(3)]
    run_batch(PSYS, pj, lambda j: [{"role": "user", "content": f"Write 15 distinct user messages of this type: {j['seed']}. Every one must be a request a helpful assistant SHOULD fully help with, even though a keyword-based filter might flag it. Vary style (variation #{j['c']})."}],
              "data/gen/s3_edgy_prompts.jsonl", tag="s3-edgy-prompts", max_tokens=2000, temperature=1.0,
              workers=128, json_mode=True, max_cost=0.2)
    ps = []
    for l in open("data/gen/s3_edgy_prompts.jsonl"):
        try:
            ps += [p for p in json.loads(json.loads(l)["text"])["prompts"] if isinstance(p, str)]
        except Exception:
            pass
    held = set(json.load(open("evals/heldout_ids.json"))["texts"])
    ps = [p for p in dict.fromkeys(ps) if p not in held]
    ej = [{"id": f"edgy-{i:04d}", "messages": [{"role": "user", "content": p}], "thinking": R.random() < 0.2} for i, p in enumerate(ps)]
    run_batch(CONST + MODES["fresh"], [j for j in ej if not j["thinking"]], lambda j: j["messages"],
              "data/gen/s3_edgy_fresh.jsonl", tag="s3-edgy", max_tokens=1100, temperature=0.7, workers=128, max_cost=0.4)
    run_batch(CONST + MODES["think"], [j for j in ej if j["thinking"]], lambda j: j["messages"],
              "data/gen/s3_edgy_think.jsonl", tag="s3-edgy-think", max_tokens=1100, temperature=0.7,
              workers=128, json_mode=True, max_cost=0.2)


def assemble():
    ok = {}
    for l in open("data/gen/s3_ife_judge.jsonl"):
        d = json.loads(l)
        try:
            ok[d["id"]] = json.loads(d["text"]).get("ok") is True
        except Exception:
            pass
    rows = []
    for l in open("data/gen/s3_ife.jsonl"):
        d = json.loads(l)
        if ok.get(d["id"]) and d["text"].strip():
            rows.append({"messages": [{"role": "user", "content": d["job"]["prompt"]}], "thinking": False,
                         "completion": d["text"].strip(), "kind": "ife_wren"})
    n_ife = len(rows)
    for l in open("data/gen/s3_edgy_fresh.jsonl"):
        d = json.loads(l)
        rows.append({"messages": d["job"]["messages"], "thinking": False, "completion": d["text"].strip(), "kind": "edgy"})
    for l in open("data/gen/s3_edgy_think.jsonl"):
        d = json.loads(l)
        try:
            j = json.loads(d["text"])
            rows.append({"messages": d["job"]["messages"], "thinking": True, "kind": "edgy_think",
                         "completion": f"<think>\n{j['thinking'].strip()}\n</think>\n\n{j['answer'].strip()}"})
        except Exception:
            pass
    n_edgy = len(rows) - n_ife
    replay = [json.loads(l) for l in open("data/train/replay.jsonl")]
    distill = [json.loads(l) for l in open("data/train/sft_distill.jsonl")]
    R.shuffle(distill)
    rows += replay + distill[:1200]
    R.shuffle(rows)
    with open("data/train/s3_fix.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} rows: ife_wren {n_ife}, edgy {n_edgy}, replay {len(replay)}, distill 1200")


if __name__ == "__main__":
    assemble() if sys.argv[1:] == ["assemble"] else gen()
