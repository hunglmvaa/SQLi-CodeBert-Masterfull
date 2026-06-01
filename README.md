# SQLi-CodeBert-Masterfull: Multi-seed paper reproduction package

This repository contains the multi-seed source-code package for the paper:

**SQL Injection Detection with SQL-Aware CodeBERT and ONNX INT8 Deployment**

The repository provides the scripts, processed splits, and reproducibility materials needed to rerun the multi-seed experiments and generate the paper tables and figures.

## Scope of this release

Included:

- multi-seed training of the proposed CodeBERT + SQL-aware tokenizer model;
- validation-set threshold calibration for each seed;
- aggregation of mean and standard deviation over seeds;
- paper baselines and ablation scripts needed to reproduce the comparison tables;
- ONNX FP32/FP16/INT8 export and CPU inference benchmark scripts;
- paper asset generation from real experiment outputs;
- the processed MasterFull train/validation/test split used by the paper.


## Repository structure

```text
sqli-codebert-masterfull-multiseed/
├── README.md
├── README_VI.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── config.py
├── preprocessing.py
├── tokenizer.py
├── train_codebert.py                  # internal worker used by run_multiseed.py
├── run_multiseed.py                   # recommended main entry point
├── run_multiseed.sh
├── validate_inputs.py
├── run_ablation_multigpu_resumable.py
├── run_classical_ml_baselines.py
├── run_baseline_exact_split_v2.py
├── run_transformer_baselines.py
├── baselines.py
├── augmentation.py
├── dataset.py
├── optimize_model.py
├── benchmark_and_plot.py
├── paper_assets_builder.py
├── make_paper_assets.py
├── data/
│   ├── README.md
│   ├── train_dataset.csv
│   ├── val_dataset.csv
│   ├── test_dataset.csv
│   ├── data_validation_report.csv
│   └── data_validation_report.json
└── docs/
    └── MULTI_SEED_GUIDE.md
```

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

The first run requires internet access to download `microsoft/codebert-base`
from Hugging Face. Later runs can use the local cache.

## Validate the included split

```bash
python validate_inputs.py --data_dir data --suffix "" --strict
```

Expected split:

| Split | Samples | Normal | SQLi |
|---|---:|---:|---:|
| Training | 149,344 | 89,607 | 59,737 |
| Validation | 37,337 | 22,402 | 14,935 |
| Test | 112,753 | 68,640 | 44,113 |
| Total | 299,434 | 180,649 | 118,785 |

## Run the proposed model with multiple seeds

Recommended paper-grade entry point:

```bash
python run_multiseed.py \
  --seeds 42,123,2024 \
  --epochs 5 \
  --batch_size 32 \
  --max_len 256 \
  --num_workers 2 \
  --skip_existing
```

Or:

```bash
bash run_multiseed.sh
```

Outputs:

```text
results/seed_42/
results/seed_123/
results/seed_2024/
results/multiseed/summary.csv
results/multiseed/summary.json
results/multiseed/summary_table.tex
```

`train_codebert.py` is retained because `run_multiseed.py` launches it once per
seed in a fresh subprocess.  For the paper release, use `run_multiseed.py` as
the main entry point instead of reporting a new single-seed run.

## Optional paper reproduction scripts

Ablation study:

```bash
python run_ablation_multigpu_resumable.py \
  --epochs 3 \
  --batch_size 32 \
  --max_len 256 \
  --num_workers 2 \
  --resume
```

Classical ML baselines:

```bash
python run_classical_ml_baselines.py \
  --train data/train_dataset.csv \
  --val data/val_dataset.csv \
  --test data/test_dataset.csv \
  --output_dir results/baselines_master \
  --models all \
  --max_features 50000
```

Character BiLSTM and 1D-CNN baselines:

```bash
python run_baseline_exact_split_v2.py \
  --train data/train_dataset.csv \
  --val data/val_dataset.csv \
  --test data/test_dataset.csv \
  --baseline all \
  --output_dir results/baselines_master
```

Transformer baselines:

```bash
python run_transformer_baselines.py \
  --train data/train_dataset.csv \
  --val data/val_dataset.csv \
  --test data/test_dataset.csv \
  --output_dir results/baselines_master \
  --models bert,distilbert,roberta,codebert \
  --epochs 3 \
  --batch_size 32
```

ONNX export and CPU benchmark after a trained checkpoint is available:

```bash
python optimize_model.py
python benchmark_and_plot.py --sample_size all --num_threads 8
```

Build paper tables and figures from generated artifacts:

```bash
python paper_assets_builder.py
```

## Citation

If you use this repository, cite the accompanying paper and this software
archive:

```bibtex
@software{sqli_codebert_masterfull_v1_2026,
  author  = {Le, Manh Hung and Nguyen, Luong Anh Tuan and Nguyen, Thai Son and Tran, Loc and Vu, Quoc Hung},
  title   = {SQLi-CodeBert-Masterfull},
  version = {v1.0.0-paper},
  year    = {2026},
  url     = {https://github.com/hunglmvaa/SQLi-CodeBert-Masterfull/tree/v1.0.0-paper}
}
```

