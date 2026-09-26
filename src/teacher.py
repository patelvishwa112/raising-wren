"""Budget-gated DeepSeek client. Every paid call goes through `chat()`.

- Hard stop: refuses any call whose worst-case cost would push the ledger past HARD_CAP_USD.
- Ledger: ledger/spend.jsonl, one line per call, costed at PEAK prices (conservative).
- Caching: keep the system prompt byte-identical across a batch; `run_batch` fires one
  warm-up request first so the shared prefix is persisted before the parallel fan-out.
- Thinking is disabled (it is billed as output).
"""
import json, os, threading, time, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "ledger" / "spend.jsonl"
HARD_CAP_USD = 19.0
MODEL = "deepseek-flash"  # DeepSeek-V4.1-Flash
# USD per 1M tokens, PEAK (off-peak is half; we always book peak to stay conservative)
P_MISS, P_HIT, P_OUT = 0.30, 0.006, 1.20

_lock = threading.Lock()
_reserved = 0.0


def _env():
    env = {}
    for line in open(ROOT / ".env"):
        line = line.strip().removeprefix("export ").strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    base = env["DEEPSEEK_BASE_URL"].rstrip("/").removesuffix("/anthropic")
    return base, env["DEEPSEEK_API_TOKEN"]


BASE_URL, _KEY = _env()


def spent() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["cost"] for l in open(LEDGER) if l.strip())


def cost(hit, miss, out):
    return (hit * P_HIT + miss * P_MISS + out * P_OUT) / 1e6


class BudgetExceeded(RuntimeError):
    pass


def chat(system, messages, max_tokens=800, temperature=0.8, tag="misc", json_mode=False, retries=4):
    """One call. Returns (text, usage). Raises BudgetExceeded before spending past the cap."""
    global _reserved
    est_in = (len(system) + sum(len(m["content"]) for m in messages)) / 2.5  # generous chars->tokens
    worst = cost(0, est_in, max_tokens)
    with _lock:
        if spent() + _reserved + worst > HARD_CAP_USD:
            raise BudgetExceeded(f"spent={spent():.4f} reserved={_reserved:.4f} worst={worst:.4f}")
        _reserved += worst
    try:
        body = {
            "model": MODEL,
            "messages": [{"role": "system", "content": system}] + messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "thinking": {"type": "disabled"},
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        for attempt in range(retries):
            try:
                r = requests.post(f"{BASE_URL}/chat/completions", json=body, timeout=180,
                                  headers={"Authorization": f"Bearer {_KEY}"})
                if r.status_code == 200:
                    break
                if r.status_code in (429, 500, 502, 503):
                    time.sleep(2 ** attempt * 3); continue
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            except requests.RequestException:
                time.sleep(2 ** attempt * 3)
        else:
            raise RuntimeError("retries exhausted")
        d = r.json()
        u = d["usage"]
        hit = u.get("prompt_cache_hit_tokens", 0)
        miss = u.get("prompt_cache_miss_tokens", u["prompt_tokens"] - hit)
        out = u["completion_tokens"]
        c = cost(hit, miss, out)
        with _lock:
            LEDGER.parent.mkdir(exist_ok=True)
            with open(LEDGER, "a") as f:
                f.write(json.dumps({"t": datetime.datetime.utcnow().isoformat(), "tag": tag,
                                    "hit": hit, "miss": miss, "out": out, "cost": c}) + "\n")
        return d["choices"][0]["message"]["content"], {"hit": hit, "miss": miss, "out": out, "cost": c}
    finally:
        with _lock:
            _reserved -= worst


def run_batch(system, jobs, fn_messages, out_path, tag, max_tokens=800, temperature=0.8,
              workers=128, json_mode=False, max_cost=None):
    """jobs: list of dicts with unique 'id'. fn_messages(job)->messages. Appends
    {'id','job','text'} to out_path; skips ids already present (resumable).
    max_cost: per-batch cap in USD on top of the global cap."""
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in open(out_path) if l.strip()}
    todo = [j for j in jobs if j["id"] not in done]
    if not todo:
        return
    start = spent()
    wlock = threading.Lock()
    tot = {"hit": 0, "miss": 0, "out": 0, "n": 0}

    def work(j):
        text, u = chat(system, fn_messages(j), max_tokens, temperature, tag, json_mode)
        with wlock:
            with open(out_path, "a") as f:
                f.write(json.dumps({"id": j["id"], "job": j, "text": text}) + "\n")
            for k in ("hit", "miss", "out"):
                tot[k] += u[k]
            tot["n"] += 1

    work(todo[0])  # warm-up: persists the shared system-prompt prefix for cache hits
    time.sleep(2)
    stop = False
    with ThreadPoolExecutor(workers) as ex:
        futs = []
        for j in todo[1:]:
            futs.append(ex.submit(work, j))
        for i, f in enumerate(as_completed(futs)):
            try:
                f.result()
            except BudgetExceeded as e:
                print("BUDGET STOP:", e); stop = True
            except Exception as e:
                print("job failed:", str(e)[:200])
            if max_cost and spent() - start > max_cost and not stop:
                print(f"batch cap {max_cost} reached; cancelling rest"); stop = True
            if stop:
                for g in futs:
                    g.cancel()
            if (i + 1) % 200 == 0:
                print(f"[{tag}] {tot['n']} done, batch ${spent()-start:.4f}, total ${spent():.4f}", flush=True)
    hr = tot["hit"] / max(1, tot["hit"] + tot["miss"])
    print(f"[{tag}] finished n={tot['n']} cache_hit_rate={hr:.2%} batch=${spent()-start:.4f} total=${spent():.4f}")


if __name__ == "__main__":
    print(f"ledger total ${spent():.4f} / cap ${HARD_CAP_USD}")
