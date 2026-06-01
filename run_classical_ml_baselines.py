#!/usr/bin/env python3
"""
run_classical_ml_baselines.py
-----------------------------
Extended classical ML baselines for the exact MasterFull train/val/test split.

This script is intended to complement run_baseline_exact_split_v2.py, which
already covers TF-IDF+SVM, Char-BiLSTM and Char-CNN. By default this script
adds three extra classical baselines and merges them into:

  results/baselines_master/all_results.json

Default added baselines:
  - tfidf_nb     : TF-IDF char n-gram + Complement Naive Bayes
  - tfidf_logreg : TF-IDF char n-gram + Logistic Regression
  - tfidf_sgd    : TF-IDF char n-gram + SGDClassifier(log_loss)

Optional:
  - tfidf_svm    : TF-IDF char n-gram + LinearSVC with validation-calibrated threshold

All thresholds are calibrated on the validation split only; final metrics are
reported once on the test split.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    from preprocessing import normalize
except Exception:  # pragma: no cover
    normalize = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_csv(path: str, do_normalize: bool = True) -> Tuple[List[str], List[int]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(f"{p} must contain columns text,label; found {list(df.columns)}")
    df = df[["text", "label"]].copy()
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    invalid = sorted(set(df["label"].unique()) - {0, 1})
    if invalid:
        raise ValueError(f"{p} contains non-binary labels: {invalid}")
    texts = df["text"].tolist()
    if do_normalize and normalize is not None:
        texts = [normalize(t) for t in texts]
    labels = df["label"].tolist()
    print(f"[load] {p.name}: {len(labels):,} samples | normal={(df.label==0).sum():,} | attack={(df.label==1).sum():,}")
    return texts, labels


def get_scores(model: Pipeline, texts: List[str]) -> np.ndarray:
    clf = model.named_steps.get("clf")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(texts)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(model, "decision_function"):
        return model.decision_function(texts)
    if clf is not None and hasattr(clf, "predict_proba"):
        proba = model.predict_proba(texts)
        return proba[:, 1]
    raise RuntimeError("Model does not expose predict_proba or decision_function.")


def calibrate_threshold(y_true: Iterable[int], scores: Iterable[float]) -> Tuple[float, float]:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    precision, recall, thresholds = precision_recall_curve(y, s)
    if thresholds.size == 0:
        return 0.5, 0.0
    # precision/recall have length len(thresholds)+1; ignore final sentinel point.
    p = precision[:-1]
    r = recall[:-1]
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx]), float(f1[idx])


def compute_metrics(y_true: Iterable[int], scores: Iterable[float], threshold: float) -> Dict[str, Any]:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "fpr": float(fp / (tn + fp)) if (tn + fp) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "threshold": float(threshold),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, s))
    except Exception:
        out["roc_auc"] = None
    try:
        out["average_precision"] = float(average_precision_score(y, s))
    except Exception:
        out["average_precision"] = None
    return out


def make_vectorizer(max_features: int, analyzer: str, ngram_min: int, ngram_max: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=(ngram_min, ngram_max),
        max_features=max_features,
        sublinear_tf=True,
        lowercase=False,
        min_df=1,
    )


def build_model(name: str, args: argparse.Namespace) -> Pipeline:
    vectorizer = make_vectorizer(args.max_features, args.analyzer, args.ngram_min, args.ngram_max)
    if name == "tfidf_nb":
        clf = ComplementNB(alpha=args.nb_alpha)
    elif name == "tfidf_logreg":
        clf = LogisticRegression(
            C=args.logreg_c,
            max_iter=args.max_iter,
            solver="liblinear",
            class_weight=args.class_weight,
            random_state=args.seed,
        )
    elif name == "tfidf_sgd":
        clf = SGDClassifier(
            loss="log_loss",
            alpha=args.sgd_alpha,
            max_iter=args.max_iter,
            tol=1e-4,
            class_weight=args.class_weight,
            random_state=args.seed,
        )
    elif name == "tfidf_svm":
        clf = LinearSVC(C=args.svm_c, max_iter=args.max_iter, class_weight=args.class_weight, random_state=args.seed)
    else:
        raise ValueError(f"Unsupported classical baseline: {name}")
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[save] {path}")


def parse_models(models: str) -> List[str]:
    if models.strip().lower() == "all":
        return ["tfidf_nb", "tfidf_logreg", "tfidf_sgd"]
    return [m.strip() for m in models.split(",") if m.strip()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run extended classical ML baselines on exact CSV split.")
    ap.add_argument("--train", default="data/train_dataset.csv")
    ap.add_argument("--val", default="data/val_dataset.csv")
    ap.add_argument("--test", default="data/test_dataset.csv")
    ap.add_argument("--output_dir", default="results/baselines_master")
    ap.add_argument("--models", default="all", help="Comma list or all. Options: tfidf_nb,tfidf_logreg,tfidf_sgd,tfidf_svm")
    ap.add_argument("--max_features", type=int, default=50000)
    ap.add_argument("--analyzer", default="char_wb", choices=["char", "char_wb", "word"])
    ap.add_argument("--ngram_min", type=int, default=3)
    ap.add_argument("--ngram_max", type=int, default=5)
    ap.add_argument("--class_weight", default=None, choices=[None, "balanced"])
    ap.add_argument("--nb_alpha", type=float, default=0.1)
    ap.add_argument("--logreg_c", type=float, default=4.0)
    ap.add_argument("--sgd_alpha", type=float, default=1e-5)
    ap.add_argument("--svm_c", type=float, default=1.0)
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_normalize", action="store_true")
    ap.add_argument("--force", action="store_true", help="Overwrite existing per-model result files.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    do_normalize = not args.no_normalize
    X_train, y_train = load_csv(args.train, do_normalize)
    X_val, y_val = load_csv(args.val, do_normalize)
    X_test, y_test = load_csv(args.test, do_normalize)

    models = parse_models(args.models)
    allowed = {"tfidf_nb", "tfidf_logreg", "tfidf_sgd", "tfidf_svm"}
    bad = sorted(set(models) - allowed)
    if bad:
        raise ValueError(f"Unsupported models: {bad}; allowed={sorted(allowed)}")

    all_results_path = out_dir / "all_results.json"
    all_results = read_json(all_results_path)
    summary_rows: List[Dict[str, Any]] = []

    name_map = {
        "tfidf_nb": "TF-IDF + ComplementNB",
        "tfidf_logreg": "TF-IDF + Logistic Regression",
        "tfidf_sgd": "TF-IDF + SGDClassifier",
        "tfidf_svm": "TF-IDF + Linear SVM calibrated",
    }

    for name in models:
        per_model_path = out_dir / f"{name}_metrics.json"
        if per_model_path.exists() and not args.force:
            print(f"[skip] {name}: {per_model_path} exists. Use --force to rerun.")
            payload = read_json(per_model_path)
            all_results[name] = payload
            row = {"baseline": payload.get("config", {}).get("baseline", name)}
            row.update(payload.get("metrics", {}))
            summary_rows.append(row)
            continue

        print(f"\n=== Classical baseline: {name} ===")
        model = build_model(name, args)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - t0

        val_scores = get_scores(model, X_val)
        tau, val_best_f1 = calibrate_threshold(y_val, val_scores)

        t1 = time.perf_counter()
        test_scores = get_scores(model, X_test)
        predict_seconds = time.perf_counter() - t1
        metrics = compute_metrics(y_test, test_scores, tau)
        metrics.update({
            "val_best_f1": float(val_best_f1),
            "train_seconds": float(train_seconds),
            "predict_seconds": float(predict_seconds),
            "latency_ms_per_sample": float((predict_seconds / max(len(X_test), 1)) * 1000),
        })

        config = {
            "baseline": name_map[name],
            "model_key": name,
            "threshold_calibration": "validation_max_f1",
            "tfidf_analyzer": args.analyzer,
            "ngram_range": [args.ngram_min, args.ngram_max],
            "max_features": args.max_features,
            "normalized_text": do_normalize,
            "class_weight": args.class_weight,
        }
        payload = {"config": config, "metrics": metrics}
        save_json(per_model_path, payload)
        all_results[name] = payload

        row = {"baseline": name_map[name]}
        row.update(metrics)
        summary_rows.append(row)
        print(f"[{name}] F1={metrics['f1']:.4f} AUC={metrics['roc_auc']:.4f} AP={metrics['average_precision']:.4f} FPR={metrics['fpr']:.4f} tau={tau:.6f}")

    save_json(all_results_path, all_results)
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(out_dir / "classical_summary.csv", index=False, float_format="%.6f")
        print(f"[save] {out_dir / 'classical_summary.csv'}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
