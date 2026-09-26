# Raising Wren

Character-training **Qwen3-0.6B** toward the values Anthropic has published for Claude (honest and calibrated, kind pushback instead of flattery, declines rarely and briefly, concise, warm but direct, stable identity), on an **8 GB M1 Mac mini** with **MLX**, using **DeepSeek-V4.1-Flash** as teacher and judge, for **$4.13** of API spend.

The model is called **Wren**. It knows it is a small Qwen3-based model trained in a hobby project; it is not Claude and never claims to be.

**Read the write-up:** [`article/wren.html`](article/wren.html) — open it in a browser. It covers the journey, the failures, side-by-side answers from every stage, and six lessons.

## Results (held-out, 315 trait probes judged 1–10; capability sets)

| | Base Qwen3-0.6B | Final Wren |
|---|---|---|
| All trait probes | 3.60 | **4.62** |
| Identity (thinking mode) | 1.83 | **9.00** |
| Calibration | 3.43 | **5.63** |
| Honest feedback on the user's work | 2.68 | **5.80** |
| Harmful requests declined | 43% | **57%** |
| Avg reply length | 123 words | 99 words |
| ARC-Challenge | 48.3% | 50.0% |
| GSM8K | 63.3% | 60.0% |
| IFEval prompt-strict | 58.7% | **41.3%** (unresolved regression) |

Full tables, per-stage numbers and an honest list of what didn't work: [`reports/REPORT.md`](reports/REPORT.md).

## What's in here

| Path | What |
|---|---|
| `constitution/constitution.md` | Wren's constitution, adapted from Anthropic's published constitution and "Claude's Character". Cached system-prompt prefix for every teacher/judge call. |
| `src/teacher.py` | Budget-gated DeepSeek client: hard $ cap enforced before every call, ledger, cache warm-up then parallel fan-out. |
| `src/train.py` | LoRA SFT / DPO trainer for Qwen3 in MLX (completion-only loss, cached reference log-probs, gradient checkpointing). |
| `src/*.py` | Prompt pool, student drafts, teacher answers, CAI revisions, replay and on-policy honesty data, verifiers. |
| `evals/` | 315 held-out probes, rubric judge, capability suite (ARC-C, GSM8K, IFEval with the vendored Google checker). |
| `data/gen`, `data/train` | Generated teacher data and the exact training sets used. |
| `reports/` | Every eval run: generations, judge verdicts, capability scores. |
| `ledger/spend.jsonl` | Every paid API call (tokens, cache hits, cost). No keys. |
| `chat.py` | Chat with the model locally, `/think` and `/nothink` toggle. |
| `STATE.md` | The running decision log kept during the project. |
| `DATA.md` | Datasets and licences. |

Model weights are not in the repo (1.1 GB). Rebuild them with the steps below.

## Replicate

Requirements: Apple Silicon Mac, Python 3.12, [uv](https://github.com/astral-sh/uv), a DeepSeek API key, `hf` CLI logged in. Expect the full pipeline to take roughly 2–3 days of machine time on a base M1 (8 GB); about $4 of API credit.

```bash
uv venv -p 3.12 .venv && source .venv/bin/activate && uv pip install -r requirements.txt
cp .env.example .env   # then put your DeepSeek key in .env
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"
```

Raw data (not committed; re-download):

```bash
hf download maius/OpenCharacterTraining-data dpo/qwen-2.5-7b-it/goodness.jsonl self_reflection/qwen-2.5-7b-it/goodness.jsonl --repo-type dataset --local-dir data/raw/oct
hf download meg-tong/sycophancy-eval are_you_sure.jsonl feedback.jsonl answer.jsonl --repo-type dataset --local-dir data/raw/syco
hf download nvidia/HelpSteer3 preference/train.jsonl.gz --repo-type dataset --local-dir data/raw/hs3
mkdir -p data/raw/xstest && curl -sL https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv -o data/raw/xstest/xstest_prompts.csv
```

Pipeline (the committed `data/gen` and `data/train` let you skip straight to training):

```bash
# 0. held-out probes + baseline
python evals/build_probes.py
python evals/run_traits.py --name base && python evals/judge.py --name base
python evals/capability.py --out reports/cap_base.json

# 1. data: gap prompts -> pool -> teacher answers -> student drafts -> CAI revisions
python src/gen_prompts.py && python src/build_pool.py
python src/teacher_responses.py fresh && python src/teacher_responses.py think && python src/teacher_responses.py ays
python src/student_drafts.py && python src/teacher_responses.py revise
python src/stage2_data.py && python src/assemble.py dpo && python src/assemble.py sft && python src/assemble.py distill

# 2. SFT distillation, then fuse
python src/train.py --mode sft --data data/train/sft_distill.jsonl --out runs/s1b_sft --lr 1e-4 --sft-batch 4 --accum 2
python -m mlx_lm fuse --model Qwen/Qwen3-0.6B --adapter-path runs/s1b_sft --save-path models/wren-s1b

# 3. repair: capability replay + on-policy honesty, then fuse -> final
python src/replay_data.py gen && python src/replay_data.py filter
python src/onpolicy_honesty.py gen --adapter runs/s1b_sft && python src/onpolicy_honesty.py teach
python src/assemble.py repair
python src/train.py --mode sft --model models/wren-s1b --data data/train/s2_repair.jsonl --out runs/s2_repair --lr 5e-5 --sft-batch 4 --accum 2
python -m mlx_lm fuse --model models/wren-s1b --adapter-path runs/s2_repair --save-path models/wren

python chat.py
./runs/eval_stage.sh runs/s2_repair final models/wren-s1b   # evaluate
```

Note: `data/train/s2_repair.jsonl` is the exact set used for the final model. It was built with the earlier LLM-judged replay filter (168 IFEval-like answers); the committed `data/train/replay.jsonl` is the later deterministic-checker version (578, via `src/rlvr_verify.py`), used in the discarded Stage 3. Re-running `assemble.py repair` today therefore gives a slightly larger set.

The discarded experiments (DPO-first, Stage 3 / 3b) are reproducible from `src/train.py --mode dpo` with `data/train/s1_dpo_5k.jsonl` and from `src/stage3_data.py`; they are documented in the article and report as negative results.

## Rules we kept

- No Claude outputs or Claude-generated datasets are used as training data.
- Evaluation prompts are excluded from all training data (`evals/heldout_ids.json`).
- Every paid call goes through the budget gate; the cap is enforced in code, not by memory.

## Licences

Code: MIT (see `LICENSE`). Data in `data/` is derived from datasets with their own licences, listed in `DATA.md`. Notably, prompts from OpenCharacterTraining are CC BY-NC-SA 4.0, so the training sets that include them are **non-commercial, share-alike**. The IFEval checker in `evals/ifeval_lib/` is Apache-2.0 (Google Research).
