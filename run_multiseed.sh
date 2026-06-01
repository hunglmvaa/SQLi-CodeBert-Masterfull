#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Multi-seed runner for the proposed CodeBERT + SQL-aware model.
# Trains the model once per seed and reports mean +/- std of the test metrics.
#
# Kaggle (single T4, 16 GB) usage:
#   cd /kaggle/working/SQLi_CodeBert_v5
#   pip install -r requirements.txt
#   bash run_multiseed.sh 2>&1 | tee run_multiseed.log
# ---------------------------------------------------------------------------
set -euo pipefail

SEEDS="${SEEDS:-42,123,2024}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LEN="${MAX_LEN:-256}"
NUM_WORKERS="${NUM_WORKERS:-2}"

echo "[run_multiseed] SEEDS=$SEEDS EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE MAX_LEN=$MAX_LEN"

python validate_inputs.py --data_dir data --suffix "" --strict

python run_multiseed.py \
  --seeds "$SEEDS" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --max_len "$MAX_LEN" \
  --num_workers "$NUM_WORKERS" \
  --skip_existing

echo "[run_multiseed] Done. See results/multiseed/summary.{json,csv} and summary_table.tex"
