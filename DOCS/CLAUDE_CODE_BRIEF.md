# Brief: give a small instruct model a Claude-like character

## Goal
Take an existing instruct model, `Qwen/Qwen3-0.6B` by default, and give it a Claude-like character through character training, constitutional training and personality shaping. The model is already pretrained, SFT'd and RL'd, so we change *who it is*, not what it knows.

"Claude-like" means:
- honest and calibrated (says when it isn't sure)
- pushes back kindly instead of flattering
- declines only when it should, briefly, and offers an alternative
- concise and answer-first
- warm but direct
- a stable identity across turns and paraphrases

Success is measurable movement on those traits vs the untouched model, with no meaningful loss of general capability.

## Context
- `docs/claude-training-blueprint.html` is the research behind this. Read the text of sections `#pivot` (plan and dataset scout), `#psm`, `#cai`, `#rlopen` and `#efficiency`. Extract the text; don't load the raw HTML into context.
- The main prior art is Open Character Training ([paper](https://arxiv.org/abs/2511.01689), [code](https://github.com/maiush/OpenCharacterTraining), [data](https://huggingface.co/datasets/maius/OpenCharacterTraining-data)):
  - DPO distillation from a teacher that follows a constitution
  - then SFT on introspection and self-interaction
- Their finding: Qwen personas are harder to move than Llama or Gemma. Keep a more malleable control model in mind, e.g. `SmolLM2-360M-Instruct` or `gemma-3-270m-it`.

## Constraints
- **Hardware:** Mac mini, Apple silicon, 8 GB unified memory, 256 GB SSD. Train with LoRA or QLoRA. MLX is the likely stack, but pick what works. Keep the project under ~60 GB on disk. Time is not a constraint; long unattended runs are fine.
- **Teacher/judge:** DeepSeek-V4.1-Flash via DeepSeek's API or OpenRouter. I'll give you the provider, key location and prices (per 1M input-miss, input-cached, output) at the start.
- **Budget:** $20 of API credit. Treat **$19 as a hard stop** that is enforced in code for every paid call, not by memory. Estimate before each batch, pilot small, then scale. Use existing Hugging Face datasets first; pay only to fill gaps.
- **Data rules:**
  - Don't use Claude-generated datasets or Claude API outputs as training data (Anthropic's terms).
  - Respect dataset licences and record them.
  - Keep eval prompts out of training data.
- **Keys:** never print, commit, or expose API keys to agy.

## How to work
- **Your role:** orchestrator, reviewer and verifier.
- **agy's role:** implementation. Google Antigravity CLI runs headless in this same folder. Learn its interface from `agy --help` and its docs before the first delegation.
- **What you delegate:** well-scoped tasks with a goal, relevant files, and a runnable done-check.
- **What you keep:** decisions that shape the model's character (the constitution, data mix, judge rubrics) and anything that spends money.
- **Shared rules:** put conventions both of you need in a short `AGENTS.md` at the root (agy reads it). Have `CLAUDE.md` import it. Keep both lean and add rules only when a mistake shows they're needed.
- **Verify every delegated result yourself** before building on it: run the check, look at the diff and at real data samples, and require evidence rather than claims. Use a fresh-context reviewer for anything subtle.
- **Keep your own context lean,** so your quota goes to judgment, not to reading large files.
- **Keep a running `STATE.md`** so any new session can resume from it.
- **Start in plan mode.** Scout datasets and draft the plan, then show me:
  - the plan
  - the constitution draft
  - the data-mix table
  - the budget allocation

  After I approve, run autonomously.

## Approach (the details are your call)
1. **Scout and baseline.** Confirm the model runs locally in non-thinking and thinking modes. Measure a baseline on a trait audit and a few capability benchmarks. Validate the datasets listed in `#pivot` (existence, licence, format, usefulness) and find any better ones.
2. **Gap analysis.** Map the public data to the ingredients: character, constitution/CAI, honesty/sycophancy, calibrated refusal, helpfulness anchor, identity/introspection. Generate data only where there are gaps. The constitution goes in a cached system prompt for all teacher calls.
3. **Train in stages.** Roughly: character DPO distillation, then constitution SFT with introspection and on-policy CAI revisions, then an optional small on-policy preference round. Change the order or drop stages if the evidence says so.
4. **Evaluate and iterate.** Re-run the audit after every stage. Keep a stage only if it improves the target traits without breaking capability.

## Done when
- A fused model plus a small chat script with a thinking on/off switch runs locally.
- A report compares base vs final (and any control model) on each trait and on capability checks, with example conversations and an honest note on what didn't work.
- All data, licences and spend are recorded.
- Total API spend is ≤ $19 per the ledger.

## Check in with me
- After the plan (before any paid calls).
- If projected spend for the remaining work exceeds the budget.
- If a stage fails its checks twice.
- If agy proves unusable.

Otherwise, keep going.
