#!/usr/bin/env python3
"""Validate split CSV files before running experiments.

The script writes JSON/CSV reports and, with --strict, fails fast when a
paper-level experiment would silently change sample counts: null text/label,
invalid labels, missing columns, or empty splits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def inspect_split(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    null_text = int(df["text"].isna().sum())
    null_label = int(df["label"].isna().sum())
    df2 = df.dropna(subset=["text", "label"]).copy()
    labels = pd.to_numeric(df2["label"], errors="coerce")
    invalid_label_count = int(labels.isna().sum() + (~labels.dropna().astype(int).isin([0, 1])).sum())
    if invalid_label_count == 0:
        labels = labels.astype(int)
    texts = df2["text"].astype(str)

    return {
        "file": str(path),
        "rows_raw": int(len(df)),
        "rows_usable": int(len(df2) - invalid_label_count),
        "normal": int((labels == 0).sum()) if invalid_label_count == 0 else None,
        "sqli": int((labels == 1).sum()) if invalid_label_count == 0 else None,
        "null_text": null_text,
        "null_label": null_label,
        "invalid_label": invalid_label_count,
        "duplicate_text": int(texts.duplicated().sum()),
        "char_len_p50": float(texts.str.len().quantile(0.50)) if len(texts) else 0.0,
        "char_len_p95": float(texts.str.len().quantile(0.95)) if len(texts) else 0.0,
        "char_len_p99": float(texts.str.len().quantile(0.99)) if len(texts) else 0.0,
        "char_len_max": int(texts.str.len().max()) if len(texts) else 0,
    }


def strict_check(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for r in rows:
        name = r["file"]
        if r["rows_raw"] == 0:
            errors.append(f"{name}: empty split")
        if r["null_text"]:
            errors.append(f"{name}: null_text={r['null_text']}")
        if r["null_label"]:
            errors.append(f"{name}: null_label={r['null_label']}")
        if r["invalid_label"]:
            errors.append(f"{name}: invalid_label={r['invalid_label']}")
        if r["rows_raw"] != r["rows_usable"]:
            errors.append(f"{name}: rows_raw={r['rows_raw']} != rows_usable={r['rows_usable']}")
        if r.get("normal") in (None, 0) or r.get("sqli") in (None, 0):
            errors.append(f"{name}: both classes must be present")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--strict", action="store_true", help="Fail on nulls, invalid labels, empty splits, or raw/usable mismatch.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = [
        data_dir / f"train_dataset{args.suffix}.csv",
        data_dir / f"val_dataset{args.suffix}.csv",
        data_dir / f"test_dataset{args.suffix}.csv",
    ]
    rows = [inspect_split(p) for p in files]
    total = {
        "file": "TOTAL",
        "rows_raw": sum(r["rows_raw"] for r in rows),
        "rows_usable": sum(r["rows_usable"] for r in rows),
        "normal": sum((r.get("normal") or 0) for r in rows),
        "sqli": sum((r.get("sqli") or 0) for r in rows),
        "null_text": sum(r["null_text"] for r in rows),
        "null_label": sum(r["null_label"] for r in rows),
        "invalid_label": sum(r["invalid_label"] for r in rows),
        "duplicate_text": sum(r["duplicate_text"] for r in rows),
    }
    report = {"splits": rows, "total": total}
    out = Path(args.out) if args.out else data_dir / "data_validation_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(rows + [total]).to_csv(out.with_suffix(".csv"), index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.strict:
        errors = strict_check(rows)
        if errors:
            raise SystemExit("Strict data validation failed:\n- " + "\n- ".join(errors))


if __name__ == "__main__":
    main()
