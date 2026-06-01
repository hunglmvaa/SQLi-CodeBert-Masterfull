# Multi-seed reproduction guide

This document is the only documentation file kept in the GitHub release.  It
focuses on the multi-seed paper workflow and excludes Kaggle/GitHub upload
instructions and single-seed release notes.

## 1. Validate inputs

```bash
python validate_inputs.py --data_dir data --suffix "" --strict
```

The command must finish without errors before training.

## 2. Run the proposed model over multiple seeds

```bash
python run_multiseed.py \
  --seeds 42,123,2024 \
  --epochs 5 \
  --batch_size 32 \
  --max_len 256 \
  --num_workers 2 \
  --skip_existing
```

`run_multiseed.py` launches `train_codebert.py` once per seed in a clean
subprocess.  This avoids CUDA-state leakage and gives one independent
checkpoint/result folder per seed.

## 3. Resume after interruption

Use `--skip_existing`.  A seed is skipped when its `threshold.json` is already
present:

```bash
python run_multiseed.py --seeds 42,123,2024 --skip_existing
```

## 4. Aggregate existing runs only

```bash
python run_multiseed.py --seeds 42,123,2024 --aggregate_only
```

## 5. Main outputs

```text
results/seed_<S>/best_model_sql_aware/
results/seed_<S>/history.json
results/seed_<S>/threshold.json
results/seed_<S>/val_predictions.npz
results/seed_<S>/test_predictions.npz
results/multiseed/summary.json
results/multiseed/summary.csv
results/multiseed/summary_table.tex
```

## 6. Reporting recommendation

Report the mean and standard deviation from `results/multiseed/summary.*`.
Avoid presenting a newly run single-seed result as the main paper result.
