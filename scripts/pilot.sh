#!/bin/bash
# Pilot of plan.md: every model, one Study A and two Study B datasets, seed 0, one process at a time.
# Prints timings and the one-pass check only. Stops at the first failure.
set -eo pipefail
cd "$(dirname "$0")/.."
for model in logreg lgbm xgboost catboost tabpfn35 tabfm tabicl; do
  for dataset in banana bank-marketing-full hotels; do
    echo "=== $model $dataset $(date +%H:%M:%S)"
    PYTHONUNBUFFERED=1 uv run python -u -m scripts.drift_run "$model" "$dataset" 0 --pilot 2>&1 | grep -v Warning
  done
done
echo "=== pilot done $(date +%H:%M:%S)"
