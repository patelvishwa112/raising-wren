"""Chat with Wren locally.

  python chat.py                       # default: models/wren (built automatically from adapters if missing)
  python chat.py --model Qwen/Qwen3-0.6B   # compare with the untouched base
  python chat.py --think               # start with thinking mode on

Commands inside the chat: /think  /nothink  /reset  /quit
"""
import argparse, sys
from pathlib import Path
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/wren")
ap.add_argument("--adapter")
ap.add_argument("--v2", action="store_true", help="Chat with Raising Wren 2.0 (Qwen/Qwen3-0.6B + runs/s4_full)")
ap.add_argument("--think", action="store_true")
ap.add_argument("--max-tokens", type=int, default=1024)
a = ap.parse_args()

if a.v2:
    a.model = "Qwen/Qwen3-0.6B"
    if not a.adapter:
        a.adapter = "runs/s4_full"
elif a.adapter and a.model == "models/wren":
    a.model = "Qwen/Qwen3-0.6B"

if a.model == "models/wren" and not Path("models/wren/config.json").exists():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.build_final import build
    print("models/wren not found; building it from Qwen/Qwen3-0.6B + committed adapters (one-time, ~2 min)...")
    build()
model, tok = load(a.model, adapter_path=a.adapter)
thinking, history = a.think, []
print(f"Loaded {a.model}. Thinking {'on' if thinking else 'off'}. Commands: /think /nothink /reset /quit\n")
while True:
    try:
        user = input("you › ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not user:
        continue
    if user in ("/quit", "/exit"):
        break
    if user in ("/think", "/nothink"):
        thinking = user == "/think"; print(f"(thinking {'on' if thinking else 'off'})"); continue
    if user == "/reset":
        history = []; print("(conversation cleared)"); continue
    history.append({"role": "user", "content": user})
    prompt = tok.apply_chat_template(history, add_generation_prompt=True, tokenize=False, enable_thinking=thinking)
    # Qwen3's recommended sampling: 0.6/0.95 thinking, 0.7/0.8 non-thinking
    sampler = make_sampler(temp=0.6 if thinking else 0.7, top_p=0.95 if thinking else 0.8)
    print("wren › ", end="", flush=True)
    text = ""
    for r in stream_generate(model, tok, prompt, max_tokens=a.max_tokens, sampler=sampler,
                             logits_processors=make_logits_processors(repetition_penalty=1.05)):
        print(r.text, end="", flush=True)
        text += r.text
    print("\n")
    answer = text.split("</think>", 1)[-1].strip()  # history keeps only the visible answer
    history.append({"role": "assistant", "content": answer})
