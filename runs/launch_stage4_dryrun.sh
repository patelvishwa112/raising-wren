#!/bin/bash
# runs/launch_stage4_dryrun.sh
# Verifies environment health, unit tests, pipeline dry-runs, and disk storage
# without triggering heavy training or unapproved compute jobs.

set -e
echo "=== Raising Wren 2.0: Pre-Flight & Pipeline Verification ==="

echo -n "1. Checking Python environment: "
python3 -c "import sys; print(f'Python {sys.version.split()[0]}')"

echo -n "2. Running unit tests: "
python3 -m unittest tests/test_pipelines.py

echo "3. Running Projection-Lite dry-run:"
python3 src/projection_data.py --track all --limit 10 --dry-run

echo "4. Running WiSE-FT LoRA sweep dry-run:"
python3 src/merge_sweep.py --adapter runs/s2_repair --scales 0.4,0.6,0.8,1.0 --dry-run

echo "5. Current Available Disk Space:"
df -h . | tail -1

echo "=== All pre-flight tests passed successfully ==="
