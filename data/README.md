# Released MasterFull split files

This directory contains the exact pre-split CSV files referenced in the manuscript and response letter.

| Split | File | Samples | Normal | SQLi |
|---|---|---:|---:|---:|
| Training | `train_dataset.csv` | 149,344 | 89,607 | 59,737 |
| Validation | `val_dataset.csv` | 37,337 | 22,402 | 14,935 |
| Test | `test_dataset.csv` | 112,753 | 68,640 | 44,113 |
| Total | — | 299,434 | 180,649 | 118,785 |

Validation and audit files:

- `data_validation_report.json` / `.csv`: row counts, null checks, invalid-label checks, within-split duplicate counts, and payload length summaries.
- `split_metadata.json` / `.csv`: SHA-256 hashes for the released CSV files and exact-text overlap audit across train/validation/test.

The released CSV schema is `text,label`, where `label=0` denotes Normal and `label=1` denotes SQLi.
