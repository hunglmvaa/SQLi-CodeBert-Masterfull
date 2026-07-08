#!/usr/bin/env python3
"""
evaluate_robustness_obfuscation.py
----------------------------------
Evaluate the trained model on obfuscated SQLi payloads without retraining.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from augmentation import augment_sample
from preprocessing import normalize


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = [normalize(str(t)) for t in texts]
        self.labels = list(map(int, labels))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_len, padding=False)
        item["labels"] = self.labels[idx]
        return item


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        y = batch.pop("labels").numpy()
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        probs.append(torch.softmax(out.logits, dim=-1)[:, 1].detach().cpu().numpy())
        labels.append(y)
    return np.concatenate(probs), np.concatenate(labels)


def metrics(labels, probs, tau):
    pred = (probs >= tau).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall_tpr": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "n": int(len(labels)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="results/best_model_sql_aware")
    ap.add_argument("--threshold_json", default="results/threshold.json")
    ap.add_argument("--data", default="data/test_dataset.csv")
    ap.add_argument("--out_dir", default="paper_assets")
    ap.add_argument("--sample_pos", type=int, default=5000)
    ap.add_argument("--sample_neg", type=int, default=5000)
    ap.add_argument("--variants_per_positive", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = json.loads(Path(args.threshold_json).read_text(encoding="utf-8"))
    tau = float(threshold.get("tau_star_from_val_F1", 0.5))

    df = pd.read_csv(args.data).dropna(subset=["text", "label"]).reset_index(drop=True)
    pos = df[df["label"].astype(int) == 1].sample(n=min(args.sample_pos, int((df["label"].astype(int) == 1).sum())), random_state=args.seed)
    neg = df[df["label"].astype(int) == 0].sample(n=min(args.sample_neg, int((df["label"].astype(int) == 0).sum())), random_state=args.seed)

    rng = random.Random(args.seed)
    obf_texts = []
    for text in pos["text"].astype(str).tolist():
        for _ in range(args.variants_per_positive):
            obf_texts.append(augment_sample(text, rng))

    scenarios = {
        "original_balanced_subset": (
            pd.concat([neg["text"], pos["text"]]).astype(str).tolist(),
            [0] * len(neg) + [1] * len(pos),
        ),
        "obfuscated_sqli_plus_original_normal": (
            neg["text"].astype(str).tolist() + obf_texts,
            [0] * len(neg) + [1] * len(obf_texts),
        ),
        "obfuscated_sqli_only": (
            obf_texts,
            [1] * len(obf_texts),
        ),
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    rows = []
    for name, (texts, labels) in scenarios.items():
        ds = TextDataset(texts, labels, tokenizer, args.max_len)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer))
        probs, y = predict_probs(model, loader, device)
        row = {"scenario": name, "tau": tau, **metrics(y, probs, tau)}
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "table_robustness_obfuscation.csv", index=False, float_format="%.6f")
    df_out.to_markdown(out_dir / "table_robustness_obfuscation.md", index=False)
    (out_dir / "robustness_obfuscation_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(df_out["scenario"], df_out["recall_tpr"], color="#1f4e79", alpha=0.86)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Recall / TPR")
    ax.set_title("Robustness under SQLi obfuscation transformations")
    ax.set_xticklabels(df_out["scenario"], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_robustness_obfuscation.png", dpi=300)
    plt.close(fig)
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
