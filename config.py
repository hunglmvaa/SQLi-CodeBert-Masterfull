"""
config.py
---------
Centralized configuration for the multi-seed SQLi CodeBERT reproduction package.

This repository release focuses on the paper-grade multi-seed workflow.  The
single training script `train_codebert.py` is kept as an internal worker called
by `run_multiseed.py`; the recommended entry point is `run_multiseed.py` or
`run_multiseed.sh`.
"""

import os

# Repository paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
ASSETS_DIR   = os.path.join(PROJECT_ROOT, "paper_assets")

# Pre-split MasterFull CSV files
TRAIN_CSV = os.path.join(DATA_DIR, "train_dataset.csv")
VAL_CSV   = os.path.join(DATA_DIR, "val_dataset.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test_dataset.csv")

# Model artifacts
MODEL_DIR     = os.path.join(RESULTS_DIR, "best_model_sql_aware")
ONNX_FP32_DIR = os.path.join(RESULTS_DIR, "onnx_fp32")
ONNX_FP16_DIR = os.path.join(RESULTS_DIR, "onnx_fp16")
ONNX_INT8_DIR = os.path.join(RESULTS_DIR, "onnx_int8")

# Training/model hyperparameters used by the paper workflow
BASE_MODEL_NAME = "microsoft/codebert-base"
MAX_SEQ_LEN     = 256
TRUNCATE_AT     = 256
NUM_EPOCHS      = 5
BATCH_SIZE      = 32
LEARNING_RATE   = 2e-5
WARMUP_RATIO    = 0.10
LABEL_SMOOTHING = 0.10
WEIGHT_DECAY    = 0.01
EARLY_STOPPING_PATIENCE = 3

# Multi-seed reproducibility
SEED = 42  # retained only for internal worker compatibility
SEEDS = [42, 123, 2024]


def seed_results_dir(seed: int) -> str:
    """Per-seed results directory, e.g. results/seed_42/."""
    return os.path.join(RESULTS_DIR, f"seed_{seed}")


def seed_model_dir(seed: int) -> str:
    """Per-seed model checkpoint directory."""
    return os.path.join(seed_results_dir(seed), "best_model_sql_aware")


# Inference/benchmarking
NUM_THREADS = 8
BENCH_SAMPLES = os.getenv("BENCH_SAMPLES", "500")  # use "all" for paper-mode CPU benchmark
BENCH_WARMUP  = 30

# Dataset balancing used in the paper description
NORMAL_TO_SQLI_RATIO = 1.5
FEW_SHOT_FRACTION    = 0.20

# Threshold calibration
DEFAULT_THRESHOLD     = 0.5
CALIBRATION_THRESHOLD = None


def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    for d in [RESULTS_DIR, ASSETS_DIR, ONNX_FP32_DIR, ONNX_FP16_DIR, ONNX_INT8_DIR]:
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    print("=== SQLi CodeBERT multi-seed configuration ===")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"BASE_MODEL  : {BASE_MODEL_NAME}")
    print(f"SEEDS       : {SEEDS}")
    print(f"MAX_SEQ_LEN : {MAX_SEQ_LEN}")
    print(f"BATCH_SIZE  : {BATCH_SIZE}")
    print(f"NUM_THREADS : {NUM_THREADS}")
    ensure_dirs()
    print("Output directories are ready.")
