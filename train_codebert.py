"""
train_codebert.py
-----------------
End-to-end training driver that consumes the pre-split CSVs produced by
data/Few-shot.py and writes every artifact required by `make_paper_assets.py`
to regenerate the paper's figures and tables.

Outputs (under results/):
  best_model_sql_aware/        HuggingFace-saved checkpoint + tokenizer
  history.json                 per-epoch training/validation losses (Fig. 3)
  val_predictions.npz          validation probabilities for threshold sweep (Fig. 4)
  test_predictions.npz         test  probabilities for confusion matrix (Fig. 5)
  threshold.json               calibrated threshold + F1 / FPR @ τ*

Usage:
  python train_codebert.py            # default: CodeBERT, 5 epochs, batch 32
  python train_codebert.py --epochs 3 --batch_size 32

Requires HuggingFace network access for the first run (to fetch the base
CodeBERT weights). Subsequent runs re-use the local cache.
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
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

import config
from preprocessing import normalize
from tokenizer import ALL_SQL_TOKENS_DEDUPED


# ── Dataset ─────────────────────────────────────────────────────────────────

class SQLiCsvDataset(Dataset):
    """Reads (text,label) from a CSV and lazily tokenizes.

    Dynamic padding is applied by DataCollatorWithPadding at DataLoader time.
    This keeps the sequence length of each batch close to the longest sample
    in that batch instead of padding every sample to MAX_SEQ_LEN.
    """
    def __init__(self, csv_path, tokenizer, max_length, do_normalize=True):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['text']).reset_index(drop=True)
        self.texts  = df['text'].astype(str).tolist()
        self.labels = df['label'].astype(int).tolist()
        if do_normalize:
            self.texts = [normalize(t) for t in self.texts]
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.tokenizer(
            self.texts[idx],
            padding=False, truncation=True,
            max_length=self.max_length,
        )
        item['labels'] = int(self.labels[idx])
        return item


# ── Embedding mean-init for new SQL tokens (Eq. 1) ──────────────────────────

def mean_init_new_embeddings(model, n_added):
    if n_added <= 0:
        return
    with torch.no_grad():
        emb = model.get_input_embeddings()
        old = emb.weight[:emb.weight.shape[0] - n_added]
        mean = old.mean(dim=0, keepdim=True)
        emb.weight[-n_added:] = mean.expand(n_added, -1)
    print(f"[init] {n_added} new SQL-token embeddings initialized at mean(V)")


# ── Loss with label smoothing (Eq. 2) ───────────────────────────────────────

class LabelSmoothingCE(torch.nn.Module):
    def __init__(self, n_classes=2, eps=0.1):
        super().__init__()
        self.eps = eps; self.n = n_classes
    def forward(self, logits, targets):
        log_p = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth = torch.full_like(log_p, self.eps / self.n)
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.eps + self.eps / self.n)
        return -(smooth * log_p).sum(dim=-1).mean()


# ── Evaluation helper ───────────────────────────────────────────────────────

@torch.no_grad()
def collect_probs(model, loader, device):
    model.eval()
    all_p, all_y, total_loss = [], [], 0.0
    crit = LabelSmoothingCE()
    for batch in loader:
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        y    = batch['labels'].to(device)
        out  = model(input_ids=ids, attention_mask=mask)
        loss = crit(out.logits, y)
        total_loss += loss.item()
        all_p.append(torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy())
        all_y.append(y.cpu().numpy())
    probs  = np.concatenate(all_p)
    labels = np.concatenate(all_y)
    return probs, labels, total_loss / max(len(loader), 1)


def metrics_from_probs(probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        'accuracy':  float(accuracy_score(labels, preds)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall':    float(recall_score(labels, preds, zero_division=0)),
        'f1':        float(f1_score(labels, preds, zero_division=0)),
        'auc':       float(roc_auc_score(labels, probs)),
        'ap':        float(average_precision_score(labels, probs)),
    }


# ── Main training loop ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', default=config.BASE_MODEL_NAME)
    ap.add_argument('--epochs',     type=int, default=config.NUM_EPOCHS)
    ap.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    ap.add_argument('--lr',         type=float, default=config.LEARNING_RATE)
    ap.add_argument('--max_len',    type=int, default=config.MAX_SEQ_LEN)
    ap.add_argument('--output_dir', default=config.MODEL_DIR)
    ap.add_argument('--results_dir', default=config.RESULTS_DIR,
                    help='Directory for history/predictions/threshold artifacts. '
                         'Override per-seed so multi-seed runs do not overwrite '
                         'each other.')
    ap.add_argument('--no_sql_tokens', action='store_true',
                    help='Ablation: disable the SQL-aware vocabulary extension.')
    ap.add_argument('--seed', type=int, default=config.SEED)
    ap.add_argument('--num_workers', type=int, default=2)
    args = ap.parse_args()

    config.ensure_dirs()
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[train] device={device}  model={args.model_name}  "
          f"max_len={args.max_len}  epochs={args.epochs}")

    # 1. Tokenizer + SQL-aware vocab augmentation
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    n_added = 0
    if not args.no_sql_tokens:
        n_added = tokenizer.add_tokens(ALL_SQL_TOKENS_DEDUPED, special_tokens=False)
        print(f"[tokenizer] +{n_added} SQL tokens (vocab size: {len(tokenizer)})")

    # 2. Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2)
    if n_added:
        model.resize_token_embeddings(len(tokenizer))
        mean_init_new_embeddings(model, n_added)
    model.to(device)

    # 3. Data
    train_ds = SQLiCsvDataset(config.TRAIN_CSV, tokenizer, args.max_len)
    val_ds   = SQLiCsvDataset(config.VAL_CSV,   tokenizer, args.max_len)
    test_ds  = SQLiCsvDataset(config.TEST_CSV,  tokenizer, args.max_len)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors='pt')
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == 'cuda'),
        collate_fn=collator,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=(device == 'cuda'),
        collate_fn=collator,
    )
    test_dl = DataLoader(
        test_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=(device == 'cuda'),
        collate_fn=collator,
    )
    print(f"[data] train={len(train_ds):,}  val={len(val_ds):,}  "
          f"test={len(test_ds):,}")

    # 4. Optimizer, scheduler, loss
    opt = AdamW(model.parameters(), lr=args.lr,
                weight_decay=config.WEIGHT_DECAY)
    total_steps  = len(train_dl) * args.epochs
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
    crit  = LabelSmoothingCE(eps=config.LABEL_SMOOTHING)

    # Save run config for reproducibility
    with open(os.path.join(results_dir, 'run_config.json'), 'w') as f:
        json.dump({
            'model_name': args.model_name,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'eval_batch_size': args.batch_size,
            'learning_rate': args.lr,
            'max_len': args.max_len,
            'label_smoothing': config.LABEL_SMOOTHING,
            'weight_decay': config.WEIGHT_DECAY,
            'warmup_ratio': config.WARMUP_RATIO,
            'sql_tokens_enabled': not args.no_sql_tokens,
            'n_added_sql_tokens': n_added,
            'dynamic_padding': True,
            'seed': args.seed,
            'data_files': {
                'train_csv': config.TRAIN_CSV,
                'val_csv': config.VAL_CSV,
                'test_csv': config.TEST_CSV,
            },
        }, f, indent=2)

    # 5. Loop
    history = []
    best_val_loss = float('inf')
    patience = config.EARLY_STOPPING_PATIENCE
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        t0 = time.time()
        for batch in tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}",
                          leave=False):
            opt.zero_grad()
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            y    = batch['labels'].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            loss = crit(out.logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            running += loss.item() * y.size(0); n += y.size(0)

        train_loss = running / max(n, 1)
        val_probs, val_y, val_loss = collect_probs(model, val_dl, device)
        val_m = metrics_from_probs(val_probs, val_y, threshold=0.5)
        elapsed = time.time() - t0
        history.append({'epoch': epoch,
                        'train_loss': train_loss, 'val_loss': val_loss,
                        'val_metrics_at_0_5': val_m, 'sec': elapsed})
        print(f"[epoch {epoch}] train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_f1@0.5={val_m['f1']:.4f}  "
              f"val_auc={val_m['auc']:.4f}  ({elapsed:.1f}s)")

        # Save best checkpoint by val_loss (paper Section III.D)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = config.EARLY_STOPPING_PATIENCE
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  ✓ saved checkpoint (best val_loss={val_loss:.4f})")
        else:
            patience -= 1
            print(f"  early-stop patience={patience}/"
                  f"{config.EARLY_STOPPING_PATIENCE}")
            if patience == 0:
                print("[train] early stop triggered.")
                break

    # 6. Save training history (Fig. 3 source data)
    with open(os.path.join(results_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # 7. Re-load best model and produce val + test predictions
    model = AutoModelForSequenceClassification.from_pretrained(
        args.output_dir).to(device)
    val_probs,  val_y,  _ = collect_probs(model, val_dl,  device)
    test_probs, test_y, _ = collect_probs(model, test_dl, device)

    np.savez(os.path.join(results_dir, 'val_predictions.npz'),
             probs=val_probs, labels=val_y)
    np.savez(os.path.join(results_dir, 'test_predictions.npz'),
             probs=test_probs, labels=test_y)

    # 8. Threshold calibration on validation, then evaluate on test (Section
    # "Threshold Calibration & Real-world Benchmark"). Avoids test-set leak.
    thresholds = np.linspace(0.01, 0.99, 99)
    f1_curve = [f1_score(val_y, (val_probs >= t).astype(int), zero_division=0)
                for t in thresholds]
    tau_star = float(thresholds[int(np.argmax(f1_curve))])
    test_m_default = metrics_from_probs(test_probs, test_y, 0.5)
    test_m_calib   = metrics_from_probs(test_probs, test_y, tau_star)
    cm = confusion_matrix(test_y, (test_probs >= tau_star).astype(int), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0

    threshold_dump = {
        'seed': int(args.seed),
        'calibration_set': 'validation',
        'evaluation_set': 'independent_test',
        'tau_star_from_val_F1': tau_star,
        'sample_counts': {
            'validation': int(len(val_y)),
            'test': int(len(test_y)),
            'test_normal': int((test_y == 0).sum()),
            'test_sqli': int((test_y == 1).sum()),
        },
        'val_F1_curve': {'thresholds': thresholds.tolist(),
                          'F1': f1_curve},
        'test_metrics_at_default_tau_0_5': test_m_default,
        'test_metrics_at_calibrated_tau':  test_m_calib,
        'test_confusion_matrix_calibrated': {
            'TN': int(tn), 'FP': int(fp),
            'FN': int(fn), 'TP': int(tp),
            'FPR': float(fpr), 'TPR': float(tpr),
        },
    }
    with open(os.path.join(results_dir, 'threshold.json'), 'w') as f:
        json.dump(threshold_dump, f, indent=2)

    print("\n=" * 8 + " Final results " + "=" * 8)
    print(f"  τ*           = {tau_star:.4f} (selected on validation)")
    print(f"  test AUC     = {test_m_calib['auc']:.4f}")
    print(f"  test AP      = {test_m_calib['ap']:.4f}")
    print(f"  test F1@0.5  = {test_m_default['f1']:.4f}")
    print(f"  test F1@τ*   = {test_m_calib['f1']:.4f}")
    print(f"  test TPR@τ*  = {tpr:.4f}")
    print(f"  test FPR@τ*  = {fpr:.4f}")


if __name__ == '__main__':
    main()
