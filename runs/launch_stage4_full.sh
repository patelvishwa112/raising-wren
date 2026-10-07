#!/bin/bash
# runs/launch_stage4_full.sh
# Full Stage 4 LoRA SFT Training on Qwen/Qwen3-0.6B with 10k dataset and 2.5k held-out eval.
# Evaluates every 2,500 trained data rows (every 500 optimization steps).

set -e
source .venv/bin/activate

mkdir -p runs/s4_full

python3 src/train.py \
  --mode sft \
  --data data/train/s4_10k.jsonl \
  --model Qwen/Qwen3-0.6B \
  --rank 16 \
  --alpha 32 \
  --lr 1.5e-5 \
  --epochs 2.0 \
  --sft-batch 1 \
  --accum 5 \
  --max-len 2048 \
  --eval-data data/eval/s4_eval_2500.jsonl \
  --eval-every-rows 2500 \
  --save-every 1000 \
  --out runs/s4_full 2>&1 | tee runs/s4_full.log
