#!/usr/bin/env python3
"""
run_baseline_exact_split.py

Run baseline models on the exact pre-split CSV files:
  - train_dataset.csv
  - val_dataset.csv
  - test_dataset.csv

This script intentionally DOES NOT re-split the data. It is intended for
paper-level comparison against the main CodeBERT/SQL-aware model using the
same train/validation/test split.

Supported baselines:
  1. svm    : TF-IDF character n-gram + Linear SVM
  2. bilstm : Character-level BiLSTM from baselines.py
  3. cnn    : Character-level 1D-CNN from baselines.py
  4. all    : Run all three baselines

Recommended quick run:
  python run_baseline_exact_split.py --baseline svm

Run with explicit paths:
  python run_baseline_exact_split.py \
      --baseline svm \
      --train train_dataset.csv \
      --val val_dataset.csv \
      --test test_dataset.csv \
      --output_dir results/baselines_exact_split
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# Make local project imports work when this file is placed beside baselines.py.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


def set_seed(seed: int) -> None:
    """Set deterministic seeds for reproducibility where possible."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch is only required for BiLSTM/CNN, not for SVM.
        pass


def load_text_label_csv(path: str) -> Tuple[List[str], List[int]]:
    """
    Load a CSV file with columns: text,label.

    The dataset files supplied for this experiment already contain the exact
    split and have columns:
      - text  : HTTP request/payload string
      - label : 0 for benign/normal, 1 for SQLi/malicious

    This loader uses pandas because it correctly handles payloads containing
    commas, quotes, escaped characters, and embedded newlines if the CSV is
    properly quoted.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"text", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} must contain columns {sorted(required)}; "
            f"missing: {sorted(missing)}; found: {list(df.columns)}"
        )

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    invalid_labels = sorted(set(df["label"].unique()) - {0, 1})
    if invalid_labels:
        raise ValueError(f"{csv_path} contains labels other than 0/1: {invalid_labels}")

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    n_total = len(labels)
    n_normal = int((df["label"] == 0).sum())
    n_sqli = int((df["label"] == 1).sum())

    print(
        f"[load] {csv_path.name}: {n_total:,} samples | "
        f"normal={n_normal:,} | sqli={n_sqli:,}"
    )

    return texts, labels


def basic_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Compute standard binary-classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def add_confusion_metrics(
    metrics: Dict[str, Any],
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, Any]:
    """Add confusion-matrix counts and derived WAF-relevant rates."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    metrics.update(
        {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "specificity": float(specificity),
            "fpr": float(fpr),
            "fnr": float(fnr),
        }
    )
    return metrics


def train_eval_tfidf_svm(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    vocab_size: int = 5000,
    c_value: float = 1.0,
) -> Tuple[Dict[str, Any], Pipeline]:
    """
    Train and evaluate TF-IDF + Linear SVM baseline.

    Configuration matches the baseline design:
      - character word-boundary n-grams: 3 to 5
      - max_features: 5000 by default
      - LinearSVC with C=1.0 by default
    """
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=vocab_size,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LinearSVC(C=c_value, max_iter=2000)),
        ]
    )

    start_train = time.perf_counter()
    pipeline.fit(train_texts, train_labels)
    train_seconds = time.perf_counter() - start_train

    start_pred = time.perf_counter()
    y_pred = pipeline.predict(test_texts)
    pred_seconds = time.perf_counter() - start_pred

    metrics = basic_metrics(test_labels, y_pred)
    metrics = add_confusion_metrics(metrics, test_labels, y_pred)

    # LinearSVC exposes decision_function, suitable for ranking metrics.
    try:
        scores = pipeline.decision_function(test_texts)
        metrics["roc_auc"] = float(roc_auc_score(test_labels, scores))
        metrics["average_precision"] = float(average_precision_score(test_labels, scores))
    except Exception:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    metrics["train_seconds"] = float(train_seconds)
    metrics["predict_seconds"] = float(pred_seconds)
    metrics["latency_ms_per_sample"] = float((pred_seconds / len(test_texts)) * 1000)

    print(
        "[TF-IDF+SVM] "
        f"acc={metrics['accuracy']:.4f} "
        f"prec={metrics['precision']:.4f} "
        f"rec={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"fpr={metrics['fpr']:.4f}"
    )

    if metrics["roc_auc"] is not None:
        print(
            "[TF-IDF+SVM] "
            f"auc={metrics['roc_auc']:.4f} "
            f"ap={metrics['average_precision']:.4f}"
        )

    return metrics, pipeline


def train_eval_bilstm(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    epochs: int,
    batch_size: int,
    device: str,
) -> Dict[str, Any]:
    """
    Train and evaluate the Character BiLSTM baseline from baselines.py.

    Note: baselines.py returns aggregate metrics but not test predictions, so
    confusion-matrix counts are not added here unless baselines.py is extended.
    """
    from baselines import train_bilstm

    start = time.perf_counter()
    metrics, _ = train_bilstm(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        num_epochs=epochs,
        batch_size=batch_size,
        device=device,
    )
    metrics = dict(metrics)
    metrics["train_eval_seconds"] = float(time.perf_counter() - start)
    return metrics


def train_eval_cnn(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    epochs: int,
    batch_size: int,
    device: str,
) -> Dict[str, Any]:
    """
    Train and evaluate the Character 1D-CNN baseline from baselines.py.

    Note: baselines.py returns aggregate metrics but not test predictions, so
    confusion-matrix counts are not added here unless baselines.py is extended.
    """
    from baselines import train_cnn

    start = time.perf_counter()
    metrics, _ = train_cnn(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        num_epochs=epochs,
        batch_size=batch_size,
        device=device,
    )
    metrics = dict(metrics)
    metrics["train_eval_seconds"] = float(time.perf_counter() - start)
    return metrics


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Save a JSON result file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[save] {path}")


def save_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Save a compact CSV summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"[save] {path}")


def build_dataset_summary(
    y_train: List[int],
    y_val: List[int],
    y_test: List[int],
) -> Dict[str, int]:
    """Return dataset split statistics."""
    return {
        "train_samples": len(y_train),
        "train_normal": int(sum(y == 0 for y in y_train)),
        "train_sqli": int(sum(y == 1 for y in y_train)),
        "val_samples": len(y_val),
        "val_normal": int(sum(y == 0 for y in y_val)),
        "val_sqli": int(sum(y == 1 for y in y_val)),
        "test_samples": len(y_test),
        "test_normal": int(sum(y == 0 for y in y_test)),
        "test_sqli": int(sum(y == 1 for y in y_test)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baselines on exact train/val/test CSV split."
    )

    parser.add_argument(
        "--train",
        default="train_dataset.csv",
        help="Path to train_dataset.csv",
    )
    parser.add_argument(
        "--val",
        default="val_dataset.csv",
        help="Path to val_dataset.csv",
    )
    parser.add_argument(
        "--test",
        default="test_dataset.csv",
        help="Path to test_dataset.csv",
    )
    parser.add_argument(
        "--baseline",
        choices=["svm", "bilstm", "cnn", "all"],
        default="svm",
        help="Which baseline to run.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/baselines_exact_split",
        help="Directory for JSON and CSV outputs.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for BiLSTM/CNN: cpu or cuda.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Epochs for BiLSTM/CNN baselines.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for BiLSTM/CNN baselines.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=5000,
        help="Max TF-IDF features for SVM baseline.",
    )
    parser.add_argument(
        "--svm_c",
        type=float,
        default=1.0,
        help="C value for LinearSVC.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--use_val_for_svm",
        action="store_true",
        help=(
            "Optional: train SVM on train+val. Default is false to preserve "
            "strict comparison with models trained only on the train split."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading exact pre-split datasets ===")
    X_train, y_train = load_text_label_csv(args.train)
    X_val, y_val = load_text_label_csv(args.val)
    X_test, y_test = load_text_label_csv(args.test)

    dataset_summary = build_dataset_summary(y_train, y_val, y_test)
    print("\n=== Split summary ===")
    for key, value in dataset_summary.items():
        print(f"{key}: {value:,}")

    baselines_to_run = (
        ["svm", "bilstm", "cnn"] if args.baseline == "all" else [args.baseline]
    )

    all_results: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []

    for baseline in baselines_to_run:
        print(f"\n=== Running baseline: {baseline} ===")

        if baseline == "svm":
            if args.use_val_for_svm:
                svm_train_texts = X_train + X_val
                svm_train_labels = y_train + y_val
                print("[svm] Training on train+val because --use_val_for_svm is set.")
            else:
                svm_train_texts = X_train
                svm_train_labels = y_train
                print("[svm] Training on train only.")

            metrics, _ = train_eval_tfidf_svm(
                train_texts=svm_train_texts,
                train_labels=svm_train_labels,
                test_texts=X_test,
                test_labels=y_test,
                vocab_size=args.vocab_size,
                c_value=args.svm_c,
            )

            config = {
                "baseline": "TF-IDF+SVM",
                "tfidf_analyzer": "char_wb",
                "ngram_range": [3, 5],
                "vocab_size": args.vocab_size,
                "svm_c": args.svm_c,
                "use_val_for_svm": args.use_val_for_svm,
            }

        elif baseline == "bilstm":
            metrics = train_eval_bilstm(
                train_texts=X_train,
                train_labels=y_train,
                val_texts=X_val,
                val_labels=y_val,
                test_texts=X_test,
                test_labels=y_test,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
            )

            config = {
                "baseline": "Character BiLSTM",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "device": args.device,
            }

        elif baseline == "cnn":
            metrics = train_eval_cnn(
                train_texts=X_train,
                train_labels=y_train,
                val_texts=X_val,
                val_labels=y_val,
                test_texts=X_test,
                test_labels=y_test,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
            )

            config = {
                "baseline": "Character 1D-CNN",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "device": args.device,
            }

        else:
            raise ValueError(f"Unsupported baseline: {baseline}")

        payload = {
            "config": config,
            "dataset": dataset_summary,
            "metrics": metrics,
        }

        all_results[baseline] = payload
        save_json(output_dir / f"{baseline}_metrics.json", payload)

        row = {"baseline": config["baseline"]}
        row.update(metrics)
        summary_rows.append(row)

    save_json(output_dir / "all_results.json", all_results)
    save_summary_csv(output_dir / "summary.csv", summary_rows)

    print("\n=== Final summary ===")
    summary_df = pd.DataFrame(summary_rows)
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(summary_df.to_string(index=False))

    print(f"\nDone. Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
