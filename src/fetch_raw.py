"""Download the raw source files into data/raw/ on demand (they are not committed).

    from src.fetch_raw import ensure_raw; ensure_raw("syco", "xstest")
    python src/fetch_raw.py            # fetch everything
"""
import shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# group -> list of (repo_id, filename in repo, local path under data/raw)
HF = {
    "oct": [("maius/OpenCharacterTraining-data", "dpo/qwen-2.5-7b-it/goodness.jsonl", "oct/dpo/qwen-2.5-7b-it/goodness.jsonl"),
            ("maius/OpenCharacterTraining-data", "self_reflection/qwen-2.5-7b-it/goodness.jsonl", "oct/self_reflection/qwen-2.5-7b-it/goodness.jsonl")],
    "syco": [("meg-tong/sycophancy-eval", f, f"syco/{f}") for f in ("are_you_sure.jsonl", "feedback.jsonl", "answer.jsonl")],
    "hs3": [("nvidia/HelpSteer3", "preference/train.jsonl.gz", "hs3/preference/train.jsonl.gz")],
}
URLS = {"xstest": [("https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv", "xstest/xstest_prompts.csv")]}


def ensure_raw(*groups):
    groups = groups or tuple(HF) + tuple(URLS)
    for g in groups:
        for repo, fname, local in HF.get(g, []):
            dst = RAW / local
            if not dst.exists():
                from huggingface_hub import hf_hub_download
                print(f"downloading {repo}/{fname}", flush=True)
                src = hf_hub_download(repo, fname, repo_type="dataset")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)
        for url, local in URLS.get(g, []):
            dst = RAW / local
            if not dst.exists():
                import requests
                print(f"downloading {url}", flush=True)
                r = requests.get(url, timeout=60); r.raise_for_status()
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(r.content)


if __name__ == "__main__":
    ensure_raw(*sys.argv[1:])
