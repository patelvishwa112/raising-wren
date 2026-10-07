"""LoRA DPO / SFT trainer for Qwen3 chat models in MLX.

Data (jsonl):
  DPO: {"messages": [...prompt turns...], "thinking": bool, "chosen": str, "rejected": str}
  SFT: {"messages": [...prompt turns...], "thinking": bool, "completion": str}
Completions for thinking=True must already contain "<think>...</think>\n\n".
Loss only on the final assistant completion. Reference log-probs for DPO are precomputed
with the adapter-free model, so only one model copy lives in memory.
"""
import argparse, json, math, random, time, sys
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

EOT = "<|im_end|>"


def encode(tok, ex, completion, max_len):
    prompt = tok.apply_chat_template(ex["messages"], add_generation_prompt=True, tokenize=False,
                                     enable_thinking=bool(ex.get("thinking")))
    p = tok.encode(prompt, add_special_tokens=False)
    c = tok.encode(completion.strip() + EOT, add_special_tokens=False)
    if len(p) > max_len - 16:
        return None
    c = c[: max_len - len(p)]
    return p, c


def batch_logps(model, seqs):
    """seqs: list of (prompt_ids, completion_ids). Returns (sum logp over completion, n_tokens)."""
    # Only project completion positions through the (151k-vocab) LM head: full-sequence
    # float32 logits were ~1 GB per pair and dominated step time.
    L = max(len(p) + len(c) for p, c in seqs)
    toks = mx.array([(p + c + [0] * (L - len(p) - len(c))) for p, c in seqs])
    h = model.model(toks[:, :-1])
    sums, ns = [], []
    for i, (p, c) in enumerate(seqs):
        hi = h[i, len(p) - 1: len(p) + len(c) - 1]
        logits = model.model.embed_tokens.as_linear(hi) if model.args.tie_word_embeddings else model.lm_head(hi)
        tgt = mx.array(c)
        lse = mx.logsumexp(logits.astype(mx.float32), axis=-1)
        picked = mx.take_along_axis(logits, tgt[:, None], axis=-1)[:, 0].astype(mx.float32)
        sums.append((picked - lse).sum()); ns.append(len(c))
    return mx.stack(sums), mx.array(ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dpo", "sft"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--nll", type=float, default=0.1, help="DPO: weight of per-token NLL on chosen")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--sft-batch", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=None, help="Stop after max-steps (for smoke test)")
    ap.add_argument("--eval-data", default=None, help="Path to held-out eval dataset (.jsonl)")
    ap.add_argument("--eval-every", type=int, default=2000, help="Run eval on eval-data every N steps")
    ap.add_argument("--eval-every-rows", type=int, default=None, help="Run eval on eval-data every N trained data rows")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); mx.random.seed(a.seed)
    # Prevent MLX from hoarding inactive memory buffers in Apple unified RAM
    try:
        mx.set_cache_limit(256 * 1024 * 1024)
    except Exception:
        pass
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    model, tok = load(a.model)
    rows = [json.loads(l) for l in open(a.data)]
    data = []
    for ex in rows:
        if a.mode == "dpo":
            ch, rj = encode(tok, ex, ex["chosen"], a.max_len), encode(tok, ex, ex["rejected"], a.max_len)
            if ch and rj:
                data.append((ch, rj))
        else:
            e = encode(tok, ex, ex["completion"], a.max_len)
            if e:
                data.append(e)
    print(f"{len(data)}/{len(rows)} examples usable", flush=True)

    eval_data = []
    if a.eval_data and Path(a.eval_data).exists():
        eval_rows = [json.loads(l) for l in open(a.eval_data)]
        for ex in eval_rows:
            e = encode(tok, ex, ex["completion"], a.max_len)
            if e:
                eval_data.append(e)
        eval_data.sort(key=lambda x: len(x[0]) + len(x[1]))
        print(f"Loaded held-out eval dataset: {len(eval_data)}/{len(eval_rows)} usable examples (sorted by length)", flush=True)

    model.freeze()
    if a.mode == "dpo":  # reference log-probs with the frozen base, before LoRA is attached
        ref_path = Path(a.data).with_suffix(f".refs{a.max_len}.json")
        refs = json.load(open(ref_path)) if ref_path.exists() else []
        if len(refs) != len(data):
            t0 = time.time(); refs = []
            for i, (ch, rj) in enumerate(data):
                s, _ = batch_logps(model, [ch, rj]); mx.eval(s)
                refs.append((s[0].item(), s[1].item()))
                if i % 500 == 0:
                    print(f"ref {i}/{len(data)} {time.time()-t0:.0f}s", flush=True); mx.clear_cache()
            json.dump(refs, open(ref_path, "w"))
        print(f"refs ready ({ref_path})", flush=True)
    cfg = {"rank": a.rank, "scale": a.alpha / a.rank, "dropout": 0.0,
           "keys": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]}
    linear_to_lora_layers(model, len(model.layers), cfg)
    from mlx_lm.tuner.trainer import grad_checkpoint
    grad_checkpoint(model.layers[0])
    ntrain = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"trainable params {ntrain/1e6:.1f}M", flush=True)
    json.dump({"fine_tune_type": "lora", "num_layers": len(model.layers), "lora_parameters": cfg,
               "model": a.model, "args": vars(a)}, open(out / "adapter_config.json", "w"), indent=1)

    if a.mode == "dpo":
        # d/dθ of -log σ(m) = -σ(-m)·β·(∇s_c - ∇s_r); compute the coefficient with a
        # grad-free forward, then backprop chosen and rejected one at a time.
        def seq_loss(model, seq, coef, nll_w):
            s, n = batch_logps(model, [seq])
            return coef * s[0] + nll_w * (-s[0] / n[0]), None
    else:
        def loss_fn(model, batch):
            s, n = batch_logps(model, batch)
            return -s.sum() / n.sum(), (n.sum(), None)

    vg = nn.value_and_grad(model, seq_loss if a.mode == "dpo" else loss_fn)
    order = list(range(len(data)))
    steps_total = math.ceil(len(data) * a.epochs / (a.accum if a.mode == "dpo" else a.accum * a.sft_batch))
    sched = optim.join_schedules([optim.linear_schedule(0.0, a.lr, max(1, steps_total // 20)),
                                  optim.cosine_decay(a.lr, steps_total)], [max(1, steps_total // 20)])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.0)

    rows_per_step = a.accum * (a.sft_batch if a.mode == "sft" else 1)
    eval_step_interval = (max(1, a.eval_every_rows // rows_per_step)) if a.eval_every_rows else a.eval_every
    print(f"Eval configured: every {a.eval_every_rows} data rows ({eval_step_interval} optimization steps)" if a.eval_every_rows else f"Eval configured: every {eval_step_interval} steps", flush=True)

    last_evaluated_step = -1

    def run_eval(step_num):
        nonlocal last_evaluated_step
        if not eval_data or step_num == last_evaluated_step:
            return
        last_evaluated_step = step_num
        rows_done = min(int(len(data) * a.epochs), step_num * rows_per_step)
        print(f"\n--- Running Held-Out Evaluation at Step {step_num} ({rows_done:,} rows trained) ({len(eval_data)} samples) ---", flush=True)
        tot_loss = 0.0
        tot_tokens = 0
        t_eval_start = time.time()
        for idx in range(0, len(eval_data), 4):
            chk = eval_data[idx:idx + 4]
            sums, ns = batch_logps(model, chk)
            mx.eval(sums, ns)
            tot_loss += -sums.sum().item()
            tot_tokens += ns.sum().item()
            mx.clear_cache()
            if (idx + 4) % 500 == 0 or idx + 4 >= len(eval_data):
                print(f"  [Eval] {min(len(eval_data), idx + 4)}/{len(eval_data)} samples evaluated...", flush=True)
        avg_l = tot_loss / max(1, tot_tokens)
        eval_ppl = math.exp(min(20.0, avg_l))
        eval_time = time.time() - t_eval_start
        eval_record = {"step": step_num, "rows_trained": rows_done, "eval_loss": avg_l, "eval_ppl": eval_ppl, "tokens": tot_tokens, "time_s": eval_time}
        print(f"[Eval Step {step_num} | {rows_done:,} Rows] Loss: {avg_l:.4f} | PPL: {eval_ppl:.2f} | Time: {eval_time:.1f}s", flush=True)
        with open(out / "eval_results.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(eval_record) + "\n")
        print("-" * 65 + "\n", flush=True)

    if eval_data:
        run_eval(0)
    items = []
    ep = 0
    while len(items) < len(data) * a.epochs:
        random.Random(a.seed + ep).shuffle(order); items += order; ep += 1
    items = items[: int(len(data) * a.epochs)]
    if a.mode == "sft":  # group by similar length for efficient batches
        chunks = [items[i:i + a.sft_batch] for i in range(0, len(items), a.sft_batch)]
    else:
        chunks = [[i] for i in items]

    acc_grads, acc_n, step, t0 = None, 0, 0, time.time()
    log = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0}
    for ci, chunk in enumerate(chunks):
        if a.max_steps and step >= a.max_steps:
            break
        if a.mode == "dpo":
            i = chunk[0]; ch, rj = data[i]; ref_c, ref_r = refs[i]
            s_c = batch_logps(model, [ch])[0][0]; s_r = batch_logps(model, [rj])[0][0]
            margin = a.beta * ((s_c - ref_c) - (s_r - ref_r)); mx.eval(margin)
            w = a.beta * mx.sigmoid(-margin).item()
            loss = -nn.log_sigmoid(margin)
            (_, _), g1 = vg(model, ch, -w, a.nll)
            (_, _), g2 = vg(model, rj, w, 0.0)
            g = tree_map_add(g1, g2)
            log["acc"] += float(margin.item() > 0); log["margin"] += margin.item()
        else:
            (loss, _), g = vg(model, [data[i] for i in chunk])
        log["loss"] += loss.item(); log["n"] += 1
        acc_grads = g if acc_grads is None else tree_map_add(acc_grads, g)
        mx.eval(acc_grads)
        mx.clear_cache()
        acc_n += 1
        if acc_n == a.accum or ci == len(chunks) - 1:
            acc_grads = tree_scale(acc_grads, 1.0 / acc_n)
            opt.update(model, acc_grads); mx.eval(model.parameters(), opt.state)
            acc_grads, acc_n = None, 0; step += 1
            if step % 10 == 0 or (a.max_steps and step >= a.max_steps):
                n = log["n"]
                msg = f"step {step}/{steps_total} loss {log['loss']/n:.4f}"
                if a.mode == "dpo":
                    msg += f" acc {log['acc']/n:.2f} margin {log['margin']/n:.3f}"
                print(msg + f" lr {sched(step).item():.2e} {time.time()-t0:.0f}s mem {mx.get_peak_memory()/1e9:.2f}GB", flush=True)
                log = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0}
                mx.clear_cache()
            if step % a.save_every == 0:
                save(model, out, f"{step:05d}_adapters.safetensors")
            if eval_data and step > 0 and (step % eval_step_interval == 0 or (a.max_steps and step >= a.max_steps)):
                run_eval(step)
            if a.max_steps and step >= a.max_steps:
                print(f"Reached max-steps limit ({a.max_steps}), ending smoke run gracefully.", flush=True)
                break
    if eval_data:
        run_eval(step)
    save(model, out, "adapters.safetensors")
    print("done", time.time() - t0)


def tree_map_add(a, b):
    from mlx.utils import tree_map
    return tree_map(lambda x, y: x + y, a, b)


def tree_scale(a, s):
    from mlx.utils import tree_map
    return tree_map(lambda x: x * s, a)


def save(model, out, name):
    w = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(out / name), w)


if __name__ == "__main__":
    main()
