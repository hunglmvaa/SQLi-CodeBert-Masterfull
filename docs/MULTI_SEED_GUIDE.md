# Multi-seed reproduction guide

The paper-grade proposed-model study uses five seeds: `7, 42, 99, 123, 2024`, five epochs, batch size 32, maximum sequence length 256, label smoothing 0.1, and 10% linear warmup.

Recommended command:

```bash
python run_multiseed.py --seeds 7,42,99,123,2024 --epochs 5 --batch_size 32 --max_len 256 --num_workers 2 --skip_existing
```

Outputs are written under `results/seed_<seed>/` and aggregated into `results/multiseed/`. These generated directories are ignored by Git.

The released artifact package in `artifacts/` contains the reviewer-facing numerical summaries from the completed reruns so reviewers can audit the manuscript values without rerunning GPU training.
