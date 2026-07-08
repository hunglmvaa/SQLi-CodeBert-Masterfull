# SQLi-CodeBert-Masterfull v1.0.0-paper

Full reproducibility package for ETASR manuscript #20371:

**SQL Injection Detection with SQL-Aware CodeBERT and ONNX INT8 Deployment**

This repository is the public `v1.0.0-paper` release referenced by the manuscript and response letter. It contains the runnable source-code package, the released SQL-aware token list, processed split metadata, exact train/validation/test CSV files, evaluation scripts, and reviewer-facing result artifacts.

## What is included

* Source code for preprocessing, SQL-aware tokenization, CodeBERT fine-tuning, multi-seed training, baselines, ablation, McNemar testing, robustness stress tests, paper asset generation, and ONNX/CPU benchmarking.
* Exact released split CSV files in `data/`:

  * `train\_dataset.csv` — 149,344 samples
  * `val\_dataset.csv` — 37,337 samples
  * `test\_dataset.csv` — 112,753 samples
* Processed split metadata and hashes:

  * `data/data\_validation\_report.json` / `.csv`
  * `data/split\_metadata.json` / `.csv`
* SQL-aware vocabulary assets:

  * `tokenizer.py`
  * `tokenizer/sql\_tokens\_139.txt`
  * `tokenizer/sql\_tokens\_category\_summary.json`
* Reviewer evidence artifacts in `artifacts/`.
* Lightweight artifact-verification scripts in `artifact\_tools/`.

Generated checkpoints, ONNX model binaries, transient `results/`, notebooks, and machine-local logs are intentionally excluded.

## Repository layout

```text
SQLi-CodeBert-Masterfull/
├── README.md
├── NOTICE
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── config.py
├── preprocessing.py
├── tokenizer.py
├── train\_codebert.py
├── run\_multiseed.py
├── run\_mcnemar.py
├── run\_obfuscation\_compare.py
├── run\_ablation\_multigpu\_resumable.py
├── run\_classical\_ml\_baselines.py
├── run\_baseline\_exact\_split\_v2.py
├── run\_transformer\_baselines.py
├── optimize\_model.py
├── benchmark\_and\_plot.py
├── data/
│   ├── train\_dataset.csv
│   ├── val\_dataset.csv
│   ├── test\_dataset.csv
│   ├── data\_validation\_report.json
│   ├── split\_metadata.json
│   └── split\_metadata.csv
├── tokenizer/
│   ├── sql\_tokens\_139.txt
│   ├── sql\_tokens\_category\_summary.json
│   └── README.md
├── artifacts/
│   ├── MASTER\_NUMBERS.json
│   ├── MASTER\_NUMBERS.md
│   ├── baselines\_master/
│   └── evidence\_for\_reviewers/
├── artifact\_tools/
│   ├── summarize\_artifacts.py
│   ├── select\_cpu\_onnx\_result.py
│   └── verify\_checksums.py
└── docs/
    ├── REPRODUCIBILITY.md
    ├── MULTI\_SEED\_GUIDE.md
    ├── ARTIFACT\_INVENTORY.md
    └── CPU\_ONNX\_SELECTION.md
```

The Python source files are kept at the repository root because the released scripts import modules such as `config`, `tokenizer`, and `preprocessing` directly. This keeps commands such as `python run\_multiseed.py` and `python run\_mcnemar.py` runnable without package-path edits.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\\Scripts\\Activate.ps1   # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

The first training run downloads `microsoft/codebert-base` through Hugging Face. Subsequent runs can use the local cache.

## Validate the released split files

```bash
python validate\_inputs.py --data\_dir data --suffix "" --strict
```

Expected counts:

|Split|Samples|Normal|SQLi|
|-|-:|-:|-:|
|Training|149,344|89,607|59,737|
|Validation|37,337|22,402|14,935|
|Test|112,753|68,640|44,113|
|Total|299,434|180,649|118,785|

`data/split\_metadata.json` records SHA-256 hashes and confirms zero exact-text overlap between train, validation, and test.

## Run the five-seed proposed model

```bash
python run\_multiseed.py   --seeds 7,42,99,123,2024   --epochs 5   --batch\_size 32   --max\_len 256   --num\_workers 2   --skip\_existing
```

Optional Windows A5000 convenience wrapper:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force; .\\Run\_Windows\_A5000\_Full.ps1
```

## Evaluation scripts

Classical baselines:

```bash
python run\_classical\_ml\_baselines.py --train data/train\_dataset.csv --val data/val\_dataset.csv --test data/test\_dataset.csv --output\_dir results/baselines\_master --models all --max\_features 50000
```

Character BiLSTM and 1D-CNN baselines:

```bash
python run\_baseline\_exact\_split\_v2.py --train data/train\_dataset.csv --val data/val\_dataset.csv --test data/test\_dataset.csv --baseline all --output\_dir results/baselines\_master
```

Transformer baselines:

```bash
python run\_transformer\_baselines.py --train data/train\_dataset.csv --val data/val\_dataset.csv --test data/test\_dataset.csv --output\_dir results/baselines\_master --models bert,distilbert,roberta,codebert --epochs 3 --batch\_size 32 --eval\_batch\_size 32
```

McNemar paired tests:

```bash
python run\_mcnemar.py --help
```

Ablation:

```bash
python run\_ablation\_multigpu\_resumable.py --epochs 3 --batch\_size 32 --max\_len 256 --num\_workers 2 --resume
```

ONNX optimization and CPU benchmark:

```bash
python optimize\_model.py
python benchmark\_and\_plot.py --help
```

## Verify the released result artifacts

```bash
python artifact\_tools/summarize\_artifacts.py
python artifact\_tools/select\_cpu\_onnx\_result.py
python artifact\_tools/verify\_checksums.py
```

The artifact tools do not retrain models. They read the released evidence files and reproduce the numerical summaries used to audit the manuscript.

## Scope and limitations

The repository supports reproducibility of the reported pipeline and reviewer evidence. It does not include trained checkpoints or ONNX binaries. The exploratory obfuscation files in `artifacts/evidence\_for\_reviewers/` are retained for transparency but are not evidence of a demonstrated adversarial-obfuscation advantage. Full leave-one-source-out retraining is not included and is left for future work, consistent with the manuscript.

