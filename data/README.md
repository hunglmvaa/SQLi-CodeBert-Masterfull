# Processed MasterFull split

This folder contains the processed train/validation/test CSV split used by the
paper workflow.

Required columns:

```text
text,label
```

Label mapping:

```text
0 = normal/benign request
1 = SQL injection request
```

Expected counts:

| Split | Samples | Normal | SQLi |
|---|---:|---:|---:|
| Training | 149,344 | 89,607 | 59,737 |
| Validation | 37,337 | 22,402 | 14,935 |
| Test | 112,753 | 68,640 | 44,113 |
| Total | 299,434 | 180,649 | 118,785 |

Validate:

```bash
python validate_inputs.py --data_dir data --suffix "" --strict
```
