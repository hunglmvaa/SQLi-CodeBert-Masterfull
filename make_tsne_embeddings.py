#!/usr/bin/env python3
"""
make_tsne_embeddings.py
-----------------------
Generate a t-SNE plot from the trained representative model without retraining.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

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
def collect_embeddings(model, loader, device):
    model.eval()
    xs, ys = [], []
    for batch in loader:
        labels = batch.pop("labels").numpy()
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch, output_hidden_states=True)
        cls = out.hidden_states[-1][:, 0, :].detach().cpu().numpy()
        xs.append(cls)
        ys.append(labels)
    return np.concatenate(xs), np.concatenate(ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="results/best_model_sql_aware")
    ap.add_argument("--data", default="data/test_dataset.csv")
    ap.add_argument("--out_dir", default="paper_assets")
    ap.add_argument("--sample_per_class", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--perplexity", type=float, default=30.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data).dropna(subset=["text", "label"]).reset_index(drop=True)
    parts = []
    for label in [0, 1]:
        part = df[df["label"].astype(int) == label]
        parts.append(part.sample(n=min(args.sample_per_class, len(part)), random_state=args.seed))
    df = pd.concat(parts).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ds = TextDataset(df["text"].tolist(), df["label"].tolist(), tokenizer, args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer))
    emb, labels = collect_embeddings(model, loader, device)

    coords = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    ).fit_transform(emb)
    out = pd.DataFrame({"tsne_x": coords[:, 0], "tsne_y": coords[:, 1], "label": labels})
    out.to_csv(out_dir / "table_tsne_test_embeddings.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    colors = {0: "#1f4e79", 1: "#c79100"}
    names = {0: "Normal", 1: "SQLi"}
    for label in [0, 1]:
        m = labels == label
        ax.scatter(coords[m, 0], coords[m, 1], s=8, alpha=0.55, c=colors[label], label=names[label])
    ax.set_title("t-SNE of CodeBERT [CLS] embeddings on test samples")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_tsne_test_embeddings.png", dpi=300)
    plt.close(fig)
    print(f"Saved {out_dir / 'fig_tsne_test_embeddings.png'}")


if __name__ == "__main__":
    main()
