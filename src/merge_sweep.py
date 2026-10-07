"""WiSE-FT / Task Arithmetic for LoRA: sweeps adapter scaling factors (lambda)
to interpolate between the base model and fine-tuned persona weights.

LoRA: W = W_0 + (alpha / rank) * (A @ B) = W_0 + lambda * Delta W
By varying lambda in [0.2, 0.4, 0.6, 0.8, 1.0], we trade off character disposition
against capability preservation (GSM8K, IFEval) without re-training.

Usage:
  python src/merge_sweep.py --adapter runs/s2_repair --scales 0.4,0.6,0.8,1.0 --dry-run
  python src/merge_sweep.py --adapter runs/s4_projection --eval-cap
"""
import argparse, copy, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def create_scaled_adapter(src_adapter_dir, scale_factor, dst_dir):
    """Create a lightweight proxy adapter directory with scaled alpha/scale parameter."""
    src_path = Path(src_adapter_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    # Read base adapter config
    config_file = src_path / "adapter_config.json"
    with open(config_file) as f:
        cfg = json.load(f)

    # Scale the LoRA scale parameter
    orig_scale = cfg["lora_parameters"].get("scale", 2.0)
    scaled_scale = orig_scale * scale_factor
    cfg["lora_parameters"]["scale"] = scaled_scale

    # Write modified config
    with open(dst_path / "adapter_config.json", "w") as f:
        json.dump(cfg, f, indent=1)

    # Symlink or link the heavy safetensors weights (0 additional disk space!)
    weight_file = src_path / "adapters.safetensors"
    dst_weights = dst_path / "adapters.safetensors"
    if dst_weights.exists() or dst_weights.is_symlink():
        dst_weights.unlink()
    dst_weights.symlink_to(weight_file.resolve())

    return dst_path


def run_sweep(adapter_dir, scales=(0.4, 0.6, 0.8, 1.0), base_model="Qwen/Qwen3-0.6B",
              eval_cap=False, eval_traits=False, dry_run=False):
    """Iterate through scale factors and record capability/trait scores."""
    print(f"=== Starting LoRA Scale Interpolation Sweep (WiSE-FT) ===")
    print(f"Source adapter: {adapter_dir}")
    print(f"Base model: {base_model}")
    print(f"Scales: {scales}")

    results = []
    tmp_base = ROOT / "runs" / "sweep_tmp"
    tmp_base.mkdir(parents=True, exist_ok=True)

    for s in scales:
        sub_dir = tmp_base / f"scale_{s:.2f}"
        print(f"\n--- Testing scale lambda = {s:.2f} ---")
        scaled_adapter = create_scaled_adapter(adapter_dir, s, sub_dir)
        print(f"Proxy adapter configured at {scaled_adapter}")

        if dry_run:
            mock_res = {
                "lambda": s,
                "arc_c": 0.50 + 0.02 * s,
                "gsm8k": 0.63 - 0.05 * s,
                "ifeval_strict": 0.58 - 0.12 * s,
                "trait_score": 3.6 + 1.2 * s
            }
            results.append(mock_res)
            print(f"[Dry Run] Mock metrics: {mock_res}")
            continue

        res = {"lambda": s, "adapter": str(scaled_adapter)}
        # Evaluate capabilities if requested
        if eval_cap:
            cap_out = ROOT / "reports" / f"sweep_cap_{s:.2f}.json"
            cmd = [
                sys.executable, str(ROOT / "evals" / "capability.py"),
                "--model", base_model,
                "--adapter", str(scaled_adapter),
                "--n-arc", "150",
                "--n-gsm", "100",
                "--n-ifeval", "100",
                "--out", str(cap_out)
            ]
            print(f"Running capability eval: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            with open(cap_out) as f:
                d = json.load(f)
                res["arc_c"] = d.get("arc_c_acc")
                res["gsm8k"] = d.get("gsm8k_acc")
                res["ifeval_prompt"] = d.get("ifeval_prompt_strict")

        results.append(res)

    print("\n=== Sweep Summary Table ===")
    print(f"{'Scale (lambda)':<15} | {'ARC-C':<10} | {'GSM8K':<10} | {'IFEval':<10} | {'Traits'}")
    print("-" * 65)
    for r in results:
        lam = f"{r.get('lambda', 0):.2f}"
        arc_val = f"{r.get('arc_c', 0):.3f}" if 'arc_c' in r else "N/A"
        gsm_val = f"{r.get('gsm8k', 0):.3f}" if 'gsm8k' in r else "N/A"
        ife_val = f"{r.get('ifeval_strict', r.get('ifeval_prompt', 0)):.3f}" if ('ifeval_strict' in r or 'ifeval_prompt' in r) else "N/A"
        tr_val = f"{r.get('trait_score', 0):.2f}" if 'trait_score' in r else "N/A"
        print(f"{lam:<15} | {arc_val:<10} | {gsm_val:<10} | {ife_val:<10} | {tr_val}")

    if dry_run and tmp_base.exists():
        shutil.rmtree(tmp_base, ignore_errors=True)

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="runs/s2_repair")
    ap.add_argument("--scales", default="0.4,0.6,0.8,1.0")
    ap.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--eval-cap", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    scale_list = [float(x.strip()) for x in args.scales.split(",")]
    run_sweep(args.adapter, scales=scale_list, base_model=args.base_model,
              eval_cap=args.eval_cap, dry_run=args.dry_run)
