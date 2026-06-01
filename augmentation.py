"""
augmentation.py
---------------
Data augmentation pipeline for obfuscation robustness (Section III.E).

For each malicious training sample, generates 3 augmented variants by
randomly composing 2–3 operations from:
  1. Comment insertion   (SEL/**/ECT)
  2. Case mutation       (SeLeCt)
  3. URL encoding        (1–3 chars)
  4. Whitespace expansion (tab, double-space)
  5. Concatenation tricks (CONCAT hex)

Only applied to TRAINING positives — validation and test sets untouched.
"""

import re
import random
import string
from typing import List, Callable


# ── Individual augmentation operations ───────────────────────────────────────

def comment_insertion(text: str, rng: random.Random) -> str:
    """
    Insert SQL inline comment (/**/) at random positions within keywords.
    e.g. SELECT → SEL/**/ECT, UNION → UN/**/ION
    """
    keywords = re.findall(
        r'\b(SELECT|UNION|INSERT|UPDATE|DELETE|DROP|EXEC|EXECUTE|FROM|WHERE|AND|OR)\b',
        text, flags=re.IGNORECASE
    )
    if not keywords:
        return text

    keyword = rng.choice(keywords)
    split_pos = rng.randint(1, len(keyword) - 1)
    replacement = keyword[:split_pos] + '/**/' + keyword[split_pos:]
    # Replace first occurrence only
    return re.sub(re.escape(keyword), replacement, text, count=1, flags=re.IGNORECASE)


def case_mutation(text: str, rng: random.Random) -> str:
    """
    Randomly toggle case of characters within SQL keywords.
    e.g. SELECT → sElEcT, UNION → UnIoN
    """
    def mutate_keyword(m):
        kw = m.group(0)
        return ''.join(
            c.upper() if rng.random() > 0.5 else c.lower()
            for c in kw
        )

    return re.sub(
        r'\b(SELECT|UNION|INSERT|UPDATE|DELETE|DROP|EXEC|FROM|WHERE|AND|OR|HAVING)\b',
        mutate_keyword,
        text,
        flags=re.IGNORECASE
    )


def url_encode_chars(text: str, rng: random.Random, n_chars: int = None) -> str:
    """
    URL-encode a random selection of 1–3 characters in the string.
    e.g. ' → %27, space → %20
    """
    if n_chars is None:
        n_chars = rng.randint(1, 3)

    encodable = [(i, c) for i, c in enumerate(text)
                 if c in "' \"=<>;(){}[]@#%+&/\\"]
    if not encodable:
        return text

    targets = rng.sample(encodable, min(n_chars, len(encodable)))
    result = list(text)
    for idx, ch in sorted(targets, reverse=True):
        result[idx] = f'%{ord(ch):02X}'
    return ''.join(result)


def whitespace_expansion(text: str, rng: random.Random) -> str:
    """
    Replace some spaces with tabs or double-spaces.
    e.g. 'UNION SELECT' → 'UNION\tSELECT' or 'UNION  SELECT'
    """
    options = ['\t', '  ', '\n', '\r\n', '  \t  ']
    return re.sub(r' ', lambda m: rng.choice(options) if rng.random() > 0.4 else ' ', text)


def concat_trick(text: str, rng: random.Random) -> str:
    """
    Replace a random SQL keyword with its CONCAT hex equivalent.
    e.g. SELECT → CONCAT(0x53,0x454c454354)
    Uses MySQL CONCAT syntax.
    """
    keywords = re.findall(
        r'\b(SELECT|UNION|INSERT|FROM|WHERE|AND|OR)\b',
        text, flags=re.IGNORECASE
    )
    if not keywords:
        return text

    keyword = rng.choice(keywords)
    hex_chars = ','.join(f'0x{ord(c):02x}' for c in keyword)
    replacement = f'CONCAT({hex_chars})'
    return re.sub(re.escape(keyword), replacement, text, count=1, flags=re.IGNORECASE)


# ── Augmentation pipeline ─────────────────────────────────────────────────────

AUGMENTATION_OPS: List[Callable] = [
    comment_insertion,
    case_mutation,
    url_encode_chars,
    whitespace_expansion,
    concat_trick,
]


def augment_sample(text: str, rng: random.Random, n_ops: int = None) -> str:
    """
    Generate one augmented variant by composing n_ops random operations.

    Args:
        text:  Original malicious parameter string
        rng:   seeded Random instance for reproducibility
        n_ops: Number of operations to compose (default: random 2–3)

    Returns:
        Augmented string
    """
    if n_ops is None:
        n_ops = rng.randint(2, 3)

    ops = rng.sample(AUGMENTATION_OPS, min(n_ops, len(AUGMENTATION_OPS)))
    result = text
    for op in ops:
        result = op(result, rng)
    return result


def augment_dataset(
    texts: List[str],
    labels: List[int],
    n_variants: int = 3,
    seed: int = 42,
    positive_label: int = 1,
) -> tuple:
    """
    Augment all positive (malicious) samples in a dataset.

    Args:
        texts:         List of input strings
        labels:        Corresponding 0/1 labels
        n_variants:    Number of augmented variants per positive sample
        seed:          Random seed for reproducibility
        positive_label: Label value for malicious class

    Returns:
        (aug_texts, aug_labels) — original + augmented samples combined
    """
    rng = random.Random(seed)
    aug_texts = list(texts)
    aug_labels = list(labels)

    pos_count = 0
    for text, label in zip(texts, labels):
        if label == positive_label:
            pos_count += 1
            for _ in range(n_variants):
                aug_texts.append(augment_sample(text, rng))
                aug_labels.append(positive_label)

    print(f"[augmentation] Original: {len(texts)} samples "
          f"({pos_count} positive)")
    print(f"[augmentation] Augmented: {len(aug_texts)} samples "
          f"(+{pos_count * n_variants} synthetic positives)")
    return aug_texts, aug_labels


if __name__ == '__main__':
    rng = random.Random(42)
    test_payloads = [
        "SELECT * FROM users WHERE id='1' OR '1'='1'--",
        "UNION SELECT username, password FROM admin",
        "'; EXEC xp_cmdshell('whoami')--",
        "1' AND SLEEP(5)--",
    ]

    print("=== Augmentation examples ===\n")
    for payload in test_payloads:
        print(f"Original:  {payload}")
        for i, op in enumerate(AUGMENTATION_OPS):
            result = op(payload, rng)
            print(f"  [{op.__name__:25s}]: {result}")
        print(f"  [composed 2-3 ops    ]: {augment_sample(payload, rng)}")
        print()
