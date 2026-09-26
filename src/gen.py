"""Local MLX generation helpers shared by evals and data generation."""
import re
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import batch_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

BASE = "Qwen/Qwen3-0.6B"


def load_model(model=BASE, adapter=None):
    return load(model, adapter_path=adapter)


def prompt_text(tok, messages, thinking=False, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                   enable_thinking=thinking)


def split_think(text):
    """Return (reasoning, answer) from a raw completion."""
    m = re.search(r"<think>(.*?)</think>", text, flags=re.S)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    if "<think>" in text:  # unterminated thinking
        return text.split("<think>", 1)[1].strip(), ""
    return "", text.strip()


def generate(model, tok, conversations, thinking=False, max_tokens=512, temp=0.7, top_p=0.8,
             batch_size=16, system=None, rep_penalty=1.05, seed=0):
    """conversations: list of message lists. Returns list of raw completions (strings)."""
    mx.random.seed(seed)
    sampler = make_sampler(temp=temp, top_p=top_p) if temp > 0 else make_sampler(0.0)
    lp = make_logits_processors(repetition_penalty=rep_penalty) if rep_penalty else None
    prompts = [tok.encode(prompt_text(tok, c, thinking, system), add_special_tokens=False)
               for c in conversations]
    kw = {"sampler": sampler}
    if lp:
        kw["logits_processors"] = lp

    def run(chunk, bs):
        # GPU OOM (seen under memory pressure) surfaces as RuntimeError, or as ZeroDivisionError
        # from mlx_lm's stats bookkeeping; retry with smaller batches.
        try:
            r = batch_generate(model, tok, chunk, max_tokens=max_tokens, **kw)
            mx.clear_cache()
            return r.texts
        except (RuntimeError, ZeroDivisionError) as e:
            mx.clear_cache()
            if bs == 1:
                raise
            print(f"generate: {type(e).__name__}, retrying with batch {bs // 2}", flush=True)
            h = max(1, bs // 2)
            return [t for j in range(0, len(chunk), h) for t in run(chunk[j:j + h], h)]

    outs = []
    for i in range(0, len(prompts), batch_size):
        outs.extend(run(prompts[i:i + batch_size], batch_size))
    return outs
