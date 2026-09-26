# Project rules (shared by Claude Code and agy)

Goal: character-train Qwen/Qwen3-0.6B locally (MLX, LoRA) toward the character in `constitution/constitution.md`. Plan and progress: `STATE.md`.

- Python: always `source .venv/bin/activate` (Python 3.12, mlx-lm installed). Install with `uv pip install`.
- Never read, print, log or copy `.env` or any API key. Never call the DeepSeek API directly; all paid calls go through `src/teacher.py` (budget-gated). agy must not make paid calls at all.
- Disk is tight (~15 GB free). Download only the files you need (`hf download ... <file>`), never whole large datasets. No full-model copies except final fused models.
- Eval prompts live in `evals/`; never put them in training data. `evals/heldout_ids.json` lists the ids to exclude.
- Memory is 8 GB unified: batch size small, `mx.clear_cache()` between phases, one MLX job at a time.
- Don't use Claude-generated datasets or Claude outputs as training data. Record every dataset + licence in `DATA.md`.
- When you finish a task, show the exact command you ran for the done-check and its output.
