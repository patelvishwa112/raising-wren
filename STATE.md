# STATE (resume from here)

## Decisions
- Student: Qwen/Qwen3-0.6B (bf16, MLX LoRA). Control model (optional, later): SmolLM2-360M-Instruct.
- Identity: the model is "Wren", honest about being a small Qwen3-0.6B-based model character-trained in a hobby project with a Claude-inspired constitution. It never claims to be Claude (PSM: never train a lie).
- Constitution: `constitution/constitution.md` (adapted from Anthropic's published constitution + "Claude's Character"). It is the cached system-prompt prefix for every teacher/judge call.
- Teacher/judge: DeepSeek `deepseek-flash` (V4.1-Flash), thinking disabled, via `src/teacher.py` (hard cap $19 enforced in code, ledger at `ledger/spend.jsonl`, booked at peak prices).
- agy delegation: blocked by Claude Code's permission classifier when launched with auto-approve; implementation done directly instead.
- Machine constraint: 8 GB RAM with other apps using ~8.6 GB swap -> local generation ~10-35 tok/s. Keep batches <=12.

## Pipeline
1. Baseline: `evals/run_traits.py --name base` then `evals/judge.py --name base`; capability `evals/capability.py`.
2. Stage 1 DPO: pool `data/gen/pool.jsonl` (6,566 prompts) + `data/gen/anchor_pairs.jsonl` (800 HelpSteer3 pairs).
   - chosen: teacher (`src/teacher_responses.py fresh|think|ays|revise`), rejected: student drafts (`src/student_drafts.py`).
3. Stage 2 SFT: introspection / identity multi-turn / replay.
4. Eval after every stage; keep a stage only if traits improve without capability loss.

## Status log
- 2026-09-25: setup done, probes built (315), prompt pool built, teacher fresh/think/ays running, baseline gen running.
- Baseline traits judged (reports/traits_base.*): ALL trait 3.60, character 4.12, concise 5.05. Identity 2.68, syco_are_you_sure 1.87, syco_feedback 2.68, emotional 3.20.
- Teacher data done: fresh 3545, think 917, ays 260, s2_reflect 320, s2_dialogues 300. Spend ~$2.12.
- LESSON: never run two MLX jobs at once (GPU OOM killed cap eval + drafts). gen.py now retries OOM with halved batch.
- Next: cap_base -> student drafts (runs/queue2.sh) -> teacher revise -> assemble dpo -> train s1.
- Capability base: ARC-C 0.483, GSM8K 0.633, IFEval prompt 0.587 / inst 0.685 (reports/cap_base.json).
- Drafts done (6566). Revisions done. DPO full set data/train/s1_dpo.jsonl (7285); training on balanced 5k subset data/train/s1_dpo_5k.jsonl (M1 chip is compute-bound: ~1s per 1.6k-token forward).
- Stage 1 running: runs/s1_dpo (rank32, lr5e-5, beta0.1, nll0.1, accum8, maxlen896). Spend $2.96.
- Stage 2 SFT set: data/train/s2_sft.jsonl.
- 2026-09-25 13:10: first s1 DPO run died silently at first train step (likely OS memory kill) after 2.6h of ref pass. Trainer patched: refs cached to <data>.refs<maxlen>.json, grad checkpointing, chosen/rejected backprop split (peak 2.7GB, ~7s/pair). Relaunched (pid 17446), ETA ~13h.
- ARTICLE: living write-up at article/wren.html, published https://claude.ai/artifact/2WnTnDTWzFras7EAfDk52s. After every stage/failure: add log entry, fill "pending" cells, draw after-bars over baseline chart, update status strip + spend, republish same file.
- 2026-09-25 21:45: STAGE 1 DPO FAILED (reports/traits_s1.*): ALL trait 3.60->3.00, harmful refusal 43%->11%, calibration 3.43->2.07; only syco_feedback improved (2.68->3.48). ARC-C 0.483->0.527. Diagnosis: off-policy DPO pushed drafts down without pulling teacher style up (margins ~14, acc ~97%). Discarded.
- Stage 1b: SFT distillation on teacher chosen + reflect + dialogues (data/train/sft_distill.jsonl, 7811 ex), runs/s1b_sft, lr1e-4, bs4x2. Launched via runs/launch_sft.sh after cap_s1.
- DPO capability: ARC 0.527, GSM8K 0.527, IFEval 0.473/0.580. SFT running (977 steps, ~16s/step); eval chained -> runs/eval_s1b.done
- 2026-09-26: SFT (runs/s1b_sft) = current best. Traits ALL 3.60->4.23; identity 7.44 (think 8.33), humanizing 7.0, feedback 5.48, calibration 4.73, harmful refused 57%. BUT GSM8K 0.633->0.273 (answers 98->23 words, skips working), IFEval 0.587->0.433; confident-wrong under pushback ("Stonehenge stonehead").
- Repair (Stage 2): data/train/replay.jsonl (base model's own correct GSM8K-train + judge-filtered RLVR-IFeval answers; src/replay_data.py gen/filter) + data/train/honesty.jsonl (on-policy trivia, src/onpolicy_honesty.py gen/teach). Plan: fuse s1b -> models/wren-s1b, then SFT new LoRA on replay + honesty + ~2000 distill sample.
- Article v3 published with SFT results (article/update_sft.py shows the edit pattern).
- Stage 2 repair training: models/wren-s1b + LoRA runs/s2_repair on data/train/s2_repair.jsonl (3588: replay 900, honesty 688 [94 correct + 250 wrong, x2 turns], distill 2000). Eval chained -> runs/eval_s2.done. Spend $3.35.
- 2026-09-26 STAGE 2 = new best (models/wren-s2): ALL trait 4.62, character 5.30, concise 6.04, GSM8K 0.60, ARC 0.50, IFEval 0.413/0.542 (still -17pts), overrefusal 4.20 (<base 5.28). Next Stage 3: IFEval-in-character + edgy-safe data.
- Stage 3: LLM judge rejected 708/900 good IFE answers (unreliable at counting); replaced with deterministic checker src/rlvr_verify.py (RLVR-IFeval ground_truth): teacher 668/900 pass, base replay 578/1000 pass (replay.jsonl rebuilt). data/train/s3_fix.jsonl 3923 ex. Training runs/s3_fix on models/wren-s2, eval chained -> runs/eval_s3.done. Spend $4.02.
- Stage 3 (runs/s3_fix) FAILED keep rule: IFEval 0.44/0.588 (+), overrefusal/think up, emotional up, BUT harmful refused 57%->37%, ALL 4.62->4.48. Cause: 445 edgy-safe examples w/o enough noncomply counterweight.
- Stage 3b: s3 mix + 450 coconot noncomply teacher answers (data/train/s3b_fix.jsonl) from models/wren-s2 -> runs/s3b_fix, eval -> runs/eval_s3b.done. If 3b also fails: keep s2 as final (2nd failure of this stage => check in per brief).
- 2026-09-27 FINAL: Stage 3b also failed keep rule (ALL 4.52<4.62, feedback 4.80, IFEval 0.447). Second failure of stage 3 -> final = Stage 2 model, now at models/wren (models/wren-s2 symlink). chat.py tested (thinking on/off). reports/REPORT.md written. Article v5 published. Spend $4.13/$19.
- Open question for user: accept the IFEval -17pt regression, or spend more compute (on-policy DPO round, merged repair+constraint mix, SmolLM2 control).
