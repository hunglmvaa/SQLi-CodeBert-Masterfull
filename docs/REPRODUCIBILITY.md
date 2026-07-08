# Reproducibility checklist

This repository is designed to satisfy the data-availability statements in the manuscript and response letter.

| Required item | Location |
|---|---|
| Source-code package | Repository root Python scripts |
| SQL-aware token list | `tokenizer/sql_tokens_139.txt` |
| Processed split metadata | `data/split_metadata.json`, `data/data_validation_report.json` |
| Exact train/validation/test CSV files | `data/train_dataset.csv`, `data/val_dataset.csv`, `data/test_dataset.csv` |
| Evaluation scripts | `run_multiseed.py`, `run_mcnemar.py`, baseline scripts, ablation scripts, ONNX benchmark scripts |
| Reviewer evidence artifacts | `artifacts/` |

Before creating the GitHub tag, run:

```bash
python validate_inputs.py --data_dir data --suffix "" --strict
python artifact_tools/verify_checksums.py
python artifact_tools/summarize_artifacts.py
python artifact_tools/select_cpu_onnx_result.py
```

Expected split counts are 149,344 / 37,337 / 112,753 for train/validation/test.
