#!/bin/bash
# usage: runs/eval_stage.sh <adapter_dir> <name> [base_model]
source .venv/bin/activate
M=${3:-Qwen/Qwen3-0.6B}
python evals/run_traits.py --model $M --adapter $1 --name $2 > runs/traits_$2.log 2>&1
python evals/judge.py --name $2 > runs/judge_$2.log 2>&1
python evals/capability.py --model $M --adapter $1 --out reports/cap_$2.json > runs/cap_$2.log 2>&1
echo DONE > runs/eval_$2.done
