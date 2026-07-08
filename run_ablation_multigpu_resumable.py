
"""
run_ablation_multigpu_resumable.py
----------------
Ablation study for the SQLi CodeBERT pipeline.

Variants:
  v0_baseline      : CodeBERT baseline, no SQL tokens, CE, tau=0.5
  v1_sql_tokens    : + SQL-aware vocabulary with mean embedding initialization
  v2_label_smooth  : + label smoothing eps=0.1
  v3_full          : + validation-based threshold calibration

Important fix: SQLiCsvDataset returns variable-length token sequences
(padding=False). DataLoader MUST use DataCollatorWithPadding; otherwise
PyTorch default_collate fails with unequal sequence lengths.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

import config
from train_codebert import SQLiCsvDataset, LabelSmoothingCE, mean_init_new_embeddings
from tokenizer import ALL_SQL_TOKENS_DEDUPED


VARIANTS = [
    {
        "id": "v0_baseline",
        "sql_aware": False,
        "label_smoothing": 0.0,
        "calibrate": False,
        "description": "CodeBERT baseline (no SQL tokens, no smoothing, tau=0.5)",
    },
    {
        "id": "v1_sql_tokens",
        "sql_aware": True,
        "label_smoothing": 0.0,
        "calibrate": False,
        "description": "+ SQL-aware tokenizer (139 tokens)",
    },
    {
        "id": "v2_label_smooth",
        "sql_aware": True,
        "label_smoothing": 0.1,
        "calibrate": False,
        "description": "+ label smoothing (eps=0.1)",
    },
    {
        "id": "v3_full",
        "sql_aware": True,
        "label_smoothing": 0.1,
        "calibrate": True,
        "description": "+ threshold calibration on validation (full model)",
    },
]


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def metrics_at_threshold(probs, labels, tau):
    preds = (probs >= tau).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tau": float(tau),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc": float(roc_auc_score(labels, probs)),
        "ap": float(average_precision_score(labels, probs)),
        "fpr": float(fpr),
        "tpr": float(tpr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


@torch.no_grad()
def collect_probs_amp(model, loader, device, use_amp):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    crit = LabelSmoothingCE()
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp and device == "cuda"):
            out = model(input_ids=ids, attention_mask=mask)
            loss = crit(out.logits, labels)
        total_loss += loss.item()
        probs = torch.softmax(out.logits, dim=-1)[:, 1].detach().float().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.detach().cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels), total_loss / max(len(loader), 1)


def save_outputs(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    cols_order = [
        "variant_id", "description", "epochs", "batch_size", "max_len", "tau",
        "accuracy", "precision", "recall", "f1", "auc", "ap", "fpr", "tpr",
        "tn", "fp", "fn", "tp", "train_seconds", "eval_seconds", "device_count", "amp",
    ]
    for col in cols_order:
        if col not in df.columns:
            df[col] = None
    df = df[cols_order]
    out_csv = output_dir / "ablation_results.csv"
    out_json = output_dir / "ablation_results.json"
    out_tex = output_dir / "ablation_results.tex"
    df.to_csv(out_csv, index=False, float_format="%.6f")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("% Ablation study\n")
        f.write("\\begin{tabular}{lllrrrrrrrr}\n\\hline\n")
        f.write("ID & Description & $\\tau$ & Acc & Prec & Rec & F1 & AUC & AP & FPR & TPR \\\\ \\hline\n")
        for row in rows:
            f.write(
                f"{row['variant_id']} & {row['description']} & "
                f"{row['tau']:.4f} & {row['accuracy']:.4f} & "
                f"{row['precision']:.4f} & {row['recall']:.4f} & "
                f"{row['f1']:.4f} & {row['auc']:.4f} & "
                f"{row['ap']:.4f} & {row['fpr'] * 100:.3f}\\% & "
                f"{row['tpr'] * 100:.3f}\\% \\\\ \n"
            )
        f.write("\\hline\n\\end{tabular}\n")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_tex}")


def load_existing_rows(output_dir):
    out_json = Path(output_dir) / "ablation_results.json"
    if not out_json.exists():
        return []
    try:
        rows = json.loads(out_json.read_text(encoding="utf-8"))
        return rows if isinstance(rows, list) else []
    except Exception:
        return []


def build_loader(dataset, tokenizer, batch_size, shuffle, args, device):
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": device == "cuda"}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator, **loader_kwargs)


def train_one_variant(variant, args):
    print("\n" + "=" * 90)
    print(f"{variant['id']} | {variant['description']}")
    print("=" * 90)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_count = torch.cuda.device_count() if device == "cuda" else 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    n_added = 0
    if variant["sql_aware"]:
        n_added = tokenizer.add_tokens(ALL_SQL_TOKENS_DEDUPED, special_tokens=False)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    if n_added:
        model.resize_token_embeddings(len(tokenizer))
        mean_init_new_embeddings(model, n_added)

    model.to(device)
    if device == "cuda" and device_count > 1 and args.use_data_parallel:
        model = torch.nn.DataParallel(model)
        print(f"[multi-gpu] Using DataParallel with {device_count} GPUs.")
    else:
        print(f"[device] Using device={device}, gpu_count={device_count}.")

    train_ds = SQLiCsvDataset(config.TRAIN_CSV, tokenizer, args.max_len)
    val_ds = SQLiCsvDataset(config.VAL_CSV, tokenizer, args.max_len)
    test_ds = SQLiCsvDataset(config.TEST_CSV, tokenizer, args.max_len)

    train_dl = build_loader(train_ds, tokenizer, args.batch_size, True, args, device)
    val_dl = build_loader(val_ds, tokenizer, args.batch_size, False, args, device)
    test_dl = build_loader(test_ds, tokenizer, args.batch_size, False, args, device)

    print(f"[data] train={len(train_ds):,}, val={len(val_ds):,}, test={len(test_ds):,}")
    print(f"[loader] train_batches={len(train_dl):,}, batch_size={args.batch_size}, max_len={args.max_len}")

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    total_steps = len(train_dl) * args.epochs
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
    crit = LabelSmoothingCE(eps=variant["label_smoothing"]) if variant["label_smoothing"] > 0 else torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device == "cuda")

    best_val_loss = float("inf")
    best_state = None
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_seen = 0.0, 0
        epoch_start = time.time()
        for batch in tqdm(train_dl, desc=f"{variant['id']} epoch {epoch}/{args.epochs}", leave=False):
            opt.zero_grad(set_to_none=True)
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device == "cuda"):
                out = model(input_ids=ids, attention_mask=mask)
                loss = crit(out.logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            running += loss.detach().float().item() * labels.size(0)
            n_seen += labels.size(0)

        val_probs, val_y, val_loss = collect_probs_amp(model, val_dl, device, args.amp)
        print(f"[{variant['id']}] epoch={epoch} train_loss={running / max(n_seen, 1):.6f} val_loss={val_loss:.6f} seconds={time.time()-epoch_start:.1f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in unwrap_model(model).state_dict().items()}

    train_seconds = time.time() - train_start
    if best_state is not None:
        unwrap_model(model).load_state_dict(best_state)

    eval_start = time.time()
    val_probs, val_y, _ = collect_probs_amp(model, val_dl, device, args.amp)
    test_probs, test_y, _ = collect_probs_amp(model, test_dl, device, args.amp)

    if variant["calibrate"]:
        thresholds = np.linspace(0.01, 0.99, 99)
        f1_curve = [f1_score(val_y, (val_probs >= t).astype(int), zero_division=0) for t in thresholds]
        tau = float(thresholds[int(np.argmax(f1_curve))])
    else:
        tau = 0.5

    row = dict(metrics_at_threshold(test_probs, test_y, tau))
    row.update({
        "variant_id": variant["id"], "description": variant["description"],
        "epochs": args.epochs, "batch_size": args.batch_size, "max_len": args.max_len,
        "train_seconds": float(train_seconds), "eval_seconds": float(time.time() - eval_start),
        "device_count": int(device_count), "amp": bool(args.amp),
    })
    print(f"[{variant['id']}] test AUC={row['auc']:.6f}, AP={row['ap']:.6f}, F1={row['f1']:.6f}, FPR={row['fpr']:.6f}, TPR={row['tpr']:.6f}, tau={row['tau']:.4f}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=config.BASE_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--max_len", type=int, default=config.MAX_SEQ_LEN)
    parser.add_argument("--output_dir", default=config.RESULTS_DIR)
    parser.add_argument("--variants", default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use_data_parallel", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    config.ensure_dirs()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    selected_ids = [v["id"] for v in VARIANTS] if args.variants == "all" else [x.strip() for x in args.variants.split(",") if x.strip()]
    selected_variants = [v for v in VARIANTS if v["id"] in selected_ids]
    if not selected_variants:
        raise ValueError(f"No valid variants selected: {args.variants}")

    rows = load_existing_rows(args.output_dir) if args.resume else []
    completed = {row.get("variant_id") for row in rows}
    print("Selected variants:", [v["id"] for v in selected_variants])
    print("Completed variants:", sorted(completed))
    print("AMP:", args.amp, "DataParallel:", args.use_data_parallel, "Batch size:", args.batch_size, "Max length:", args.max_len, "Epochs:", args.epochs)

    for variant in selected_variants:
        if args.resume and variant["id"] in completed:
            print(f"Skip completed variant: {variant['id']}")
            continue
        row = train_one_variant(variant, args)
        rows = [r for r in rows if r.get("variant_id") != row["variant_id"]]
        rows.append(row)
        save_outputs(rows, args.output_dir)

    print("\nAblation completed.")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
