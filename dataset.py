"""
dataset.py
----------
Dataset loading, parsing, and splitting (Section IV.A).

Supports:
  - CSIC 2010 HTTP Dataset  (primary benchmark)
  - CICIDS-2018             (cross-dataset generalization evaluation)
  - Synthetic demo mode     (for running code without downloading datasets)

Dataset download instructions:
  CSIC 2010:    https://www.tic.itefi.csic.es/dataset/
  CICIDS-2018:  https://www.unb.ca/cic/datasets/ids-2018.html
"""

import os
import re
import random
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
from preprocessing import normalize


# ── CSIC 2010 Parser ─────────────────────────────────────────────────────────

def parse_csic_file(filepath: str) -> Tuple[List[str], List[int]]:
    """
    Parse a CSIC 2010 HTTP request file.
    Extracts GET/POST parameter values from each HTTP request.

    File format (text, one request per block separated by blank lines):
        GET /tienda1/publico/anadir.jsp?id=1&nombre=Jam+%C3%B3n HTTP/1.1
        Host: localhost
        ...

    Args:
        filepath: Path to CSIC .txt file
        label:    0 = benign, 1 = malicious

    Returns:
        (texts, labels) — raw parameter strings and their labels
    """
    texts = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Split into individual requests
    requests = re.split(r'\n\s*\n', content.strip())

    for req in requests:
        if not req.strip():
            continue
        params = _extract_params_from_request(req)
        texts.extend(params)

    return texts


def _extract_params_from_request(request_text: str) -> List[str]:
    """
    Extract all parameter values from a single HTTP request.
    Covers: URL query string, POST body, Cookie header.
    """
    params = []
    lines = request_text.strip().split('\n')

    for line in lines:
        # GET/POST request line: extract query string
        if line.startswith(('GET ', 'POST ', 'PUT ', 'DELETE ')):
            parts = line.split(' ')
            if len(parts) >= 2:
                url = parts[1]
                if '?' in url:
                    qs = url.split('?', 1)[1]
                    for kv in qs.split('&'):
                        if '=' in kv:
                            params.append(kv.split('=', 1)[1])

        # POST body (after headers — detect by lack of ':')
        elif '=' in line and ':' not in line and len(line) > 3:
            for kv in line.split('&'):
                if '=' in kv:
                    params.append(kv.split('=', 1)[1])

        # Cookie header
        elif line.lower().startswith('cookie:'):
            cookie_val = line.split(':', 1)[1].strip()
            for kv in cookie_val.split(';'):
                kv = kv.strip()
                if '=' in kv:
                    params.append(kv.split('=', 1)[1])

    return [p.strip() for p in params if p.strip()]


def load_csic2010(
    normal_path: str,
    anomalous_path: str,
    max_benign: int = 13720,
    max_malicious: int = 13720,
    seed: int = 42,
) -> Tuple[List[str], List[int]]:
    """
    Load and balance the CSIC 2010 dataset.

    Args:
        normal_path:    Path to normalTrafficTest.txt
        anomalous_path: Path to anomalousTrafficTest.txt
                        (SQL injection entries will be filtered)
        max_benign:     Max benign samples to keep (balance)
        max_malicious:  Max malicious samples to keep
        seed:           Random seed

    Returns:
        (texts, labels) balanced dataset
    """
    rng = random.Random(seed)

    benign_texts   = parse_csic_file(normal_path)
    malicious_texts = parse_csic_file(anomalous_path)

    # Filter malicious for SQL injection signatures
    sql_pattern = re.compile(
        r"(union\s+select|select\s+\*|insert\s+into|drop\s+table|"
        r"exec\s*\(|1\s*=\s*1|--|/\*|\bor\b\s+\d|\bsleep\s*\()",
        re.IGNORECASE
    )
    malicious_texts = [t for t in malicious_texts if sql_pattern.search(normalize(t))]

    # Subsample to balance
    rng.shuffle(benign_texts)
    rng.shuffle(malicious_texts)
    benign_texts    = benign_texts[:max_benign]
    malicious_texts = malicious_texts[:max_malicious]

    texts  = benign_texts + malicious_texts
    labels = [0] * len(benign_texts) + [1] * len(malicious_texts)

    # Apply normalization
    texts = [normalize(t) for t in texts]

    print(f"[dataset] CSIC 2010 loaded: {len(benign_texts)} benign, "
          f"{len(malicious_texts)} malicious")
    return texts, labels


def load_cicids2018(csv_path: str, seed: int = 42) -> Tuple[List[str], List[int]]:
    """
    Load CICIDS-2018 web attack partition.
    Expects a CSV with columns: 'payload' and 'label' (0/1).
    If the CSV has the original CICIDS format, adapt accordingly.

    Args:
        csv_path: Path to CICIDS-2018 SQL injection CSV
        seed:     Random seed

    Returns:
        (texts, labels)
    """
    df = pd.read_csv(csv_path)

    # Adapt to expected column names
    if 'payload' not in df.columns:
        # Try common column names
        for col in ['Payload', 'request', 'Request', 'http_request', 'feature']:
            if col in df.columns:
                df = df.rename(columns={col: 'payload'})
                break

    if 'label' not in df.columns:
        for col in ['Label', 'class', 'Class', 'attack_type']:
            if col in df.columns:
                df['label'] = (df[col].astype(str).str.lower() != 'benign').astype(int)
                break

    texts  = [normalize(str(t)) for t in df['payload'].tolist()]
    labels = df['label'].astype(int).tolist()

    rng = random.Random(seed)
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    texts, labels = zip(*combined)

    print(f"[dataset] CICIDS-2018 loaded: {sum(l==0 for l in labels)} benign, "
          f"{sum(l==1 for l in labels)} malicious")
    return list(texts), list(labels)


# ── Train / Val / Test split ──────────────────────────────────────────────────

def split_dataset(
    texts: List[str],
    labels: List[int],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed: int = 42,
) -> Tuple:
    """
    Stratified 70 / 15 / 15 split matching paper Section IV.A.

    Returns:
        (train_texts, val_texts, test_texts,
         train_labels, val_labels, test_labels)
    """
    test_ratio = 1.0 - train_ratio - val_ratio

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        texts, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed
    )
    val_frac = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=(1.0 - val_frac),
        stratify=y_tmp,
        random_state=seed
    )

    print(f"[dataset] Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_kfold_splits(
    texts: List[str],
    labels: List[int],
    n_splits: int = 5,
    seed: int = 42,
):
    """
    Stratified k-fold splits for cross-validation (paper Section IV.A).
    Yields (train_texts, train_labels, val_texts, val_labels) per fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    arr_texts  = np.array(texts)
    arr_labels = np.array(labels)
    for fold, (train_idx, val_idx) in enumerate(skf.split(arr_texts, arr_labels)):
        yield (
            fold,
            arr_texts[train_idx].tolist(),
            arr_labels[train_idx].tolist(),
            arr_texts[val_idx].tolist(),
            arr_labels[val_idx].tolist(),
        )


# ── Synthetic demo dataset (no download required) ────────────────────────────

SYNTHETIC_BENIGN = [
    "login=admin&password=secret123",
    "search=laptop&category=electronics&sort=price",
    "user_id=42&action=view&page=1",
    "name=John+Doe&email=john@example.com",
    "product_id=100&quantity=2&color=blue",
    "q=select+all+items&filter=active",
    "comment=where+clause+documentation",
    "filename=report_2024.pdf&download=true",
    "city=New+York&country=US&zip=10001",
    "lang=en&theme=dark&timezone=UTC",
]

SYNTHETIC_MALICIOUS = [
    "id=1' OR '1'='1",
    "username=admin'--",
    "id=1 UNION SELECT username,password FROM users--",
    "id=1; DROP TABLE users--",
    "name=' OR 1=1--",
    "id=1 AND SLEEP(5)--",
    "id=1; EXEC xp_cmdshell('whoami')--",
    "' UNION SELECT 1,2,3,4--",
    "id=1 AND 1=CONVERT(int,(SELECT TOP 1 name FROM sysobjects))--",
    "1' AND BENCHMARK(1000000,MD5(1))--",
    "id=' UNION SELECT null,null,null--",
    "user='; INSERT INTO admin VALUES('hacker','pass')--",
    "id=1 OR 1=1#",
    "pass=') OR ('1'='1",
    "id=1/**/UNION/**/SELECT/**/1,2,3--",
]


def load_synthetic_demo(n_samples: int = 200, seed: int = 42) -> Tuple[List[str], List[int]]:
    """
    Generate a small synthetic dataset for testing without downloading CSIC.
    Expands base samples with augmentation to reach n_samples.
    """
    from augmentation import augment_sample
    rng_aug = random.Random(seed)
    rng_sample = random.Random(seed + 1)

    texts, labels = [], []

    half = n_samples // 2
    # Benign
    while len(texts) < half:
        base = rng_sample.choice(SYNTHETIC_BENIGN)
        texts.append(normalize(base))
        labels.append(0)
        # Simple word perturbations for variety
        texts.append(normalize(base.replace('&', '&amp;')))
        labels.append(0)

    texts = texts[:half]
    labels = labels[:half]

    # Malicious
    mal_texts = []
    while len(mal_texts) < half:
        base = rng_sample.choice(SYNTHETIC_MALICIOUS)
        mal_texts.append(normalize(base))
        mal_texts.append(normalize(augment_sample(base, rng_aug)))
        mal_texts.append(normalize(augment_sample(base, rng_aug)))

    mal_texts = mal_texts[:half]
    texts  = texts  + mal_texts
    labels = labels + [1] * len(mal_texts)

    print(f"[dataset] Synthetic demo: {half} benign, {len(mal_texts)} malicious")
    return texts, labels


if __name__ == '__main__':
    print("=== Synthetic demo dataset ===")
    texts, labels = load_synthetic_demo(n_samples=100, seed=42)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(texts, labels)

    print(f"\nTrain class distribution: "
          f"{sum(y==0 for y in y_train)} benign, {sum(y==1 for y in y_train)} malicious")
    print(f"Val   class distribution: "
          f"{sum(y==0 for y in y_val)} benign, {sum(y==1 for y in y_val)} malicious")
    print(f"Test  class distribution: "
          f"{sum(y==0 for y in y_test)} benign, {sum(y==1 for y in y_test)} malicious")

    print("\n=== 5-fold cross-validation splits ===")
    for fold, tr_t, tr_l, va_t, va_l in get_kfold_splits(texts, labels, n_splits=5):
        pos = sum(l == 1 for l in va_l)
        print(f"  Fold {fold+1}: val={len(va_l)} ({pos} pos / {len(va_l)-pos} neg)")
