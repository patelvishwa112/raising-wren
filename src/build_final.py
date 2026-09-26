"""Rebuild the fused models from the base model (downloaded from Hugging Face) and the
committed LoRA adapters:
    Qwen/Qwen3-0.6B + runs/s1b_sft   -> models/wren-s1b   (SFT stage)
    models/wren-s1b + runs/s2_repair -> models/wren       (final Wren)
Usage: python src/build_final.py   (chat.py calls this automatically if models/wren is missing)"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [("Qwen/Qwen3-0.6B", "runs/s1b_sft", "models/wren-s1b"),
         ("models/wren-s1b", "runs/s2_repair", "models/wren")]


def build(keep_intermediate=False):
    for base, adapter, out in STEPS:
        if (ROOT / out / "config.json").exists():
            continue
        print(f"fusing {base} + {adapter} -> {out}", flush=True)
        subprocess.run([sys.executable, "-m", "mlx_lm", "fuse", "--model", base,
                        "--adapter-path", adapter, "--save-path", out], cwd=ROOT, check=True)
    if not keep_intermediate:
        import shutil
        shutil.rmtree(ROOT / "models" / "wren-s1b", ignore_errors=True)
    return str(ROOT / "models" / "wren")


if __name__ == "__main__":
    build(keep_intermediate="--keep" in sys.argv)
