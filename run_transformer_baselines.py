#!/usr/bin/env python3
"""
run_transformer_baselines.py
----------------------------
Transformer baselines for the exact MasterFull train/val/test split.

Supported model aliases:
  - bert       -> bert-base-uncased
  - distilbert -> distilbert-base-uncased
  - roberta    -> roberta-base
  - codebert   -> microsoft/codebert-base, WITHOUT SQL-aware tokenization

Important:
  These are baselines. They intentionally do not add the 139 SQL-aware tokens.
  The decision threshold is calibrated on the validation split only, then final
  metrics are reported on the independent test split.

Outputs are merged into:
  results/baselines_master/all_results.json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
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
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

try:
    from preprocessing import normalize
except Exception:  # pragma: no cover
    normalize = None


MODEL_REGISTRY = {
    "bert": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
    "codebert": "microsoft/codebert-base",
}

DISPLAY_NAMES = {
    "bert": "BERT-base",
    "distilbert": "DistilBERT-base",
    "roberta": "RoBERTa-base",
    "codebert": "CodeBERT-base without SQL-aware tokenizer",
}


class CsvTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        item["labels"] = int(self.labels[idx])
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def calibrate_threshold(y_true: Iterable[int], scores: Iterable[float]) -> Tuple[float, float]:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    precision, recall, thresholds = precision_recall_curve(y, s)
    if thresholds.size == 0:
        return 0.5, 0.0
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
        return ["bert", "distilbert", "roberta", "codebert"]
    return [m.strip() for m in models.split(",") if m.strip()]


def make_loaders(tokenizer, X_train, y_train, X_val, y_val, X_test, y_test, args):
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if args.amp else None)
    train_ds = CsvTextDataset(X_train, y_train, tokenizer, args.max_len)
    val_ds = CsvTextDataset(X_val, y_val, tokenizer, args.max_len)
    test_ds = CsvTextDataset(X_test, y_test, tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator, num_workers=args.num_workers)
    return train_loader, val_loader, test_loader


def evaluate(model, loader, device: torch.device, amp: bool) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_probs: List[float] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            labels = batch["labels"].to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                out = model(**batch)
                loss = out.loss
            probs = torch.softmax(out.logits, dim=-1)[:, 1].detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            total_loss += float(loss.detach().cpu()) * labels.size(0)
            total_n += labels.size(0)
    return total_loss / max(total_n, 1), np.asarray(all_probs), np.asarray(all_labels)


def train_one_model(alias: str, X_train, y_train, X_val, y_val, X_test, y_test, args) -> Dict[str, Any]:
    model_name = MODEL_REGISTRY[alias]
    print(f"\n=== Transformer baseline: {alias} ({model_name}) ===")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(args.amp and device.type == "cuda")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load HuggingFace model '{model_name}'. On Kaggle, enable internet or add this model to the input dataset/cache. Original error: {exc}"
        ) from exc

    model.to(device)
    train_loader, val_loader, test_loader = make_loaders(tokenizer, X_train, y_train, X_val, y_val, X_test, y_test, args)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_state = None
    best_val_loss = math.inf
    patience_left = args.patience
    history: List[Dict[str, Any]] = []
    start_total = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"train {alias} epoch {epoch}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar, start=1):
            labels = batch["labels"].to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(**batch)
                loss = out.loss / max(args.grad_accum_steps, 1)
            scaler.scale(loss).backward()
            if step % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            batch_n = labels.size(0)
            running_loss += float(loss.detach().cpu()) * batch_n * max(args.grad_accum_steps, 1)
            seen += batch_n
            pbar.set_postfix(loss=running_loss / max(seen, 1))

        val_loss, val_probs, val_labels = evaluate(model, val_loader, device, use_amp)
        tau, val_f1 = calibrate_threshold(val_labels, val_probs)
        val_metrics = compute_metrics(val_labels, val_probs, tau)
        hist_row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val_loss": val_loss,
            "val_best_f1": val_f1,
            "val_threshold": tau,
            "val_fpr": val_metrics["fpr"],
        }
        history.append(hist_row)
        print(f"[{alias}] epoch={epoch} train_loss={hist_row['train_loss']:.5f} val_loss={val_loss:.5f} val_f1={val_f1:.4f} tau={tau:.6f}")

        if val_loss + args.min_delta < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{alias}] early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    val_loss, val_probs, val_labels = evaluate(model, val_loader, device, use_amp)
    tau, val_best_f1 = calibrate_threshold(val_labels, val_probs)

    test_start = time.perf_counter()
    test_loss, test_probs, test_labels = evaluate(model, test_loader, device, use_amp)
    eval_seconds = time.perf_counter() - test_start
    metrics = compute_metrics(test_labels, test_probs, tau)
    metrics.update({
        "val_loss": float(val_loss),
        "test_loss": float(test_loss),
        "val_best_f1": float(val_best_f1),
        "train_eval_seconds": float(time.perf_counter() - start_total),
        "test_eval_seconds": float(eval_seconds),
        "latency_ms_per_sample": float((eval_seconds / max(len(test_labels), 1)) * 1000),
    })
    payload = {
        "config": {
            "baseline": DISPLAY_NAMES[alias],
            "model_key": alias,
            "pretrained_model": model_name,
            "max_len": args.max_len,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "threshold_calibration": "validation_max_f1",
            "sql_aware_tokens": False,
            "normalized_text": not args.no_normalize,
            "amp": use_amp,
        },
        "history": history,
        "metrics": metrics,
    }

    print(f"[{alias}] TEST F1={metrics['f1']:.4f} AUC={metrics['roc_auc']:.4f} AP={metrics['average_precision']:.4f} FPR={metrics['fpr']:.4f} tau={metrics['threshold']:.6f}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Transformer baselines on exact CSV split.")
    ap.add_argument("--train", default="data/train_dataset.csv")
    ap.add_argument("--val", default="data/val_dataset.csv")
    ap.add_argument("--test", default="data/test_dataset.csv")
    ap.add_argument("--output_dir", default="results/baselines_master")
    ap.add_argument("--models", default="bert,distilbert,roberta,codebert", help="Comma list or all.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    ap.add_argument("--no_normalize", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = parse_models(args.models)
    bad = sorted(set(models) - set(MODEL_REGISTRY))
    if bad:
        raise ValueError(f"Unsupported transformer models: {bad}; allowed={sorted(MODEL_REGISTRY)}")

    do_normalize = not args.no_normalize
    X_train, y_train = load_csv(args.train, do_normalize)
    X_val, y_val = load_csv(args.val, do_normalize)
    X_test, y_test = load_csv(args.test, do_normalize)

    all_results_path = out_dir / "all_results.json"
    all_results = read_json(all_results_path)
    summary_rows: List[Dict[str, Any]] = []

    for alias in models:
        per_model_path = out_dir / f"{alias}_metrics.json"
        if per_model_path.exists() and not args.force:
            print(f"[skip] {alias}: {per_model_path} exists. Use --force to rerun.")
            payload = read_json(per_model_path)
        else:
            payload = train_one_model(alias, X_train, y_train, X_val, y_val, X_test, y_test, args)
            save_json(per_model_path, payload)
        all_results[alias] = payload
        row = {"baseline": payload.get("config", {}).get("baseline", alias)}
        row.update(payload.get("metrics", {}))
        summary_rows.append(row)

    save_json(all_results_path, all_results)
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(out_dir / "transformer_summary.csv", index=False, float_format="%.6f")
        print(f"[save] {out_dir / 'transformer_summary.csv'}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
