"""
baselines.py
------------
Non-transformer baseline models (Section IV.B).

Implements:
  1. TF-IDF + SVM     (classical ML baseline)
  2. Character BiLSTM (deep learning baseline)
  3. 1D-CNN           (deep learning baseline)

All use the same train/val/test splits as the transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset, DataLoader


# ── Shared metrics helper ─────────────────────────────────────────────────────

def metrics(y_true, y_pred) -> dict:
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
    }


# ── 1. TF-IDF + SVM ──────────────────────────────────────────────────────────

def train_tfidf_svm(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    vocab_size: int = 5000,
    C: float = 1.0,
) -> dict:
    """
    TF-IDF (character 3-5 gram) + LinearSVC.
    Matches paper Section IV.B: vocab=5K, char 3–5 grams, C=1.0.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            max_features=vocab_size,
            sublinear_tf=True,
        )),
        ('clf', LinearSVC(C=C, max_iter=2000)),
    ])
    pipeline.fit(train_texts, train_labels)
    preds = pipeline.predict(test_texts)
    m = metrics(test_labels, preds)
    print(f"[TF-IDF+SVM] acc={m['accuracy']:.4f} f1={m['f1']:.4f}")
    return m, pipeline


# ── 2. Character-level BiLSTM ─────────────────────────────────────────────────

class CharVocab:
    """Simple character vocabulary for BiLSTM."""
    PAD = 0
    UNK = 1

    def __init__(self, texts: List[str], max_vocab: int = 256):
        chars = set()
        for t in texts:
            chars.update(t)
        self.char2idx = {c: i + 2 for i, c in enumerate(sorted(chars)[:max_vocab])}
        self.size = len(self.char2idx) + 2

    def encode(self, text: str, max_len: int = 200) -> List[int]:
        ids = [self.char2idx.get(c, self.UNK) for c in text[:max_len]]
        # Pad / truncate
        ids = ids + [self.PAD] * max(0, max_len - len(ids))
        return ids[:max_len]


class CharDataset(Dataset):
    def __init__(self, texts, labels, vocab: CharVocab, max_len: int = 200):
        self.X = torch.tensor([vocab.encode(t, max_len) for t in texts], dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class BiLSTMClassifier(nn.Module):
    """
    2-layer character-level Bidirectional LSTM.
    Matches paper: 2 layers, 256 hidden units per direction, dropout=0.3.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden: int = 256, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm   = nn.LSTM(embed_dim, hidden, num_layers=n_layers,
                              bidirectional=True, batch_first=True,
                              dropout=dropout if n_layers > 1 else 0.0)
        self.drop   = nn.Dropout(dropout)
        self.fc     = nn.Linear(hidden * 2, 2)  # *2 for bidirectional

    def forward(self, x):
        emb = self.embed(x)                         # (B, L, E)
        out, (h, _) = self.lstm(emb)                # h: (2*layers, B, H)
        # Concat last forward + last backward hidden states
        h_fwd = h[-2]  # forward  last layer
        h_bwd = h[-1]  # backward last layer
        pooled = torch.cat([h_fwd, h_bwd], dim=1)  # (B, 2H)
        logits = self.fc(self.drop(pooled))         # (B, 2)
        return logits


def train_bilstm(
    train_texts: List[str], train_labels: List[int],
    val_texts:   List[str], val_labels:   List[int],
    test_texts:  List[str], test_labels:  List[int],
    num_epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = 'cpu',
) -> dict:
    """Train and evaluate character BiLSTM."""
    vocab   = CharVocab(train_texts + val_texts + test_texts)
    tr_data = CharDataset(train_texts, train_labels, vocab)
    va_data = CharDataset(val_texts,   val_labels,   vocab)
    te_data = CharDataset(test_texts,  test_labels,  vocab)

    tr_loader = DataLoader(tr_data, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_data, batch_size=batch_size)
    te_loader = DataLoader(te_data, batch_size=batch_size)

    model = BiLSTMClassifier(vocab.size).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.CrossEntropyLoss()

    best_f1, best_state = 0.0, None
    for epoch in range(num_epochs):
        model.train()
        for X, y in tr_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            crit(model(X), y).backward()
            opt.step()

        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for X, y in va_loader:
                p = model(X.to(device)).argmax(1).cpu()
                all_p.extend(p.tolist()); all_y.extend(y.tolist())
        val_f1 = f1_score(all_y, all_p, zero_division=0)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for X, y in te_loader:
            p = model(X.to(device)).argmax(1).cpu()
            all_p.extend(p.tolist()); all_y.extend(y.tolist())
    m = metrics(all_y, all_p)
    print(f"[BiLSTM] acc={m['accuracy']:.4f} f1={m['f1']:.4f}")
    return m, model


# ── 3. 1D-CNN ─────────────────────────────────────────────────────────────────

class CNNClassifier(nn.Module):
    """
    1D-CNN with parallel filters of widths 3, 4, 5 and 128 feature maps each.
    Matches paper Section IV.B.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 n_filters: int = 128, filter_sizes: Tuple = (3, 4, 5),
                 dropout: float = 0.3):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs   = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, fs) for fs in filter_sizes
        ])
        self.drop    = nn.Dropout(dropout)
        self.fc      = nn.Linear(n_filters * len(filter_sizes), 2)

    def forward(self, x):
        emb = self.embed(x).permute(0, 2, 1)        # (B, E, L)
        pooled = [F.relu(conv(emb)).max(dim=2)[0]   # (B, F)
                  for conv in self.convs]
        cat    = torch.cat(pooled, dim=1)            # (B, F*3)
        return self.fc(self.drop(cat))               # (B, 2)


def train_cnn(
    train_texts: List[str], train_labels: List[int],
    val_texts:   List[str], val_labels:   List[int],
    test_texts:  List[str], test_labels:  List[int],
    num_epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = 'cpu',
) -> dict:
    """Train and evaluate 1D-CNN."""
    vocab   = CharVocab(train_texts + val_texts + test_texts)
    tr_data = CharDataset(train_texts, train_labels, vocab)
    va_data = CharDataset(val_texts,   val_labels,   vocab)
    te_data = CharDataset(test_texts,  test_labels,  vocab)

    tr_loader = DataLoader(tr_data, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_data, batch_size=batch_size)
    te_loader = DataLoader(te_data, batch_size=batch_size)

    model = CNNClassifier(vocab.size).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.CrossEntropyLoss()

    best_f1, best_state = 0.0, None
    for epoch in range(num_epochs):
        model.train()
        for X, y in tr_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            crit(model(X), y).backward()
            opt.step()

        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for X, y in va_loader:
                p = model(X.to(device)).argmax(1).cpu()
                all_p.extend(p.tolist()); all_y.extend(y.tolist())
        val_f1 = f1_score(all_y, all_p, zero_division=0)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for X, y in te_loader:
            p = model(X.to(device)).argmax(1).cpu()
            all_p.extend(p.tolist()); all_y.extend(y.tolist())
    m = metrics(all_y, all_p)
    print(f"[1D-CNN] acc={m['accuracy']:.4f} f1={m['f1']:.4f}")
    return m, model


if __name__ == '__main__':
    from dataset import load_synthetic_demo, split_dataset
    from augmentation import augment_dataset

    print("=== Baselines smoke test ===\n")
    texts, labels = load_synthetic_demo(200, seed=42)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_dataset(texts, labels)
    X_tr_aug, y_tr_aug = augment_dataset(X_tr, y_tr, n_variants=2)

    print("--- TF-IDF + SVM ---")
    m_svm, _ = train_tfidf_svm(X_tr_aug, y_tr_aug, X_te, y_te)

    print("\n--- BiLSTM ---")
    m_lstm, _ = train_bilstm(X_tr_aug, y_tr_aug, X_va, y_va, X_te, y_te,
                              num_epochs=3, batch_size=32)

    print("\n--- 1D-CNN ---")
    m_cnn, _ = train_cnn(X_tr_aug, y_tr_aug, X_va, y_va, X_te, y_te,
                          num_epochs=3, batch_size=32)
