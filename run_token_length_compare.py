#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_token_length_compare.py
--------------------------------------------------------------------------
So do dai chuoi token giua SQL-aware tokenizer va tokenizer mac dinh tren
CUNG mot test set. Muc tieu: dinh luong "SQL-aware tokenization rut ngan
chuoi bao nhieu %", noi truc tiep vao latency da bao cao (it token -> nhanh).

Khong can GPU, khong can checkpoint - chi can 2 tokenizer.

Chay:
  python run_token_length_compare.py --test_csv data/test_dataset.csv ^
      --sql_aware_dir results/best_model_sql_aware ^
      --vanilla_name microsoft/codebert-base ^
      --out_dir results/token_length
"""
import argparse, json, os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

def lengths(tok, texts, max_len):
    # do dai THUC (khong pad, khong truncate) de thay hieu qua nen chuoi
    enc = tok(list(texts), truncation=False, padding=False)
    raw = np.array([len(x) for x in enc["input_ids"]])
    # va do dai sau truncation (anh huong thuc te den compute)
    clipped = np.minimum(raw, max_len)
    return raw, clipped

def stats(a):
    return dict(mean=float(a.mean()), p50=float(np.percentile(a,50)),
                p95=float(np.percentile(a,95)), p99=float(np.percentile(a,99)),
                max=int(a.max()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--sql_aware_dir", required=True,
                    help="thu muc model SQL-aware (da co tokenizer + 139 SQL tokens)")
    ap.add_argument("--vanilla_name", default="microsoft/codebert-base")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--out_dir", default="results/token_length")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.test_csv)
    texts = df[args.text_col].astype(str).tolist()

    tok_sql = AutoTokenizer.from_pretrained(args.sql_aware_dir)
    tok_van = AutoTokenizer.from_pretrained(args.vanilla_name)
    print(f"[vocab] SQL-aware={len(tok_sql)}  vanilla={len(tok_van)}  "
          f"(+{len(tok_sql)-len(tok_van)} tokens)")

    raw_s, clip_s = lengths(tok_sql, texts, args.max_len)
    raw_v, clip_v = lengths(tok_van, texts, args.max_len)

    out = {
        "n_samples": len(texts),
        "vocab_sql_aware": len(tok_sql),
        "vocab_vanilla": len(tok_van),
        "raw_length": {"sql_aware": stats(raw_s), "vanilla": stats(raw_v)},
        "clipped_length": {"sql_aware": stats(clip_s), "vanilla": stats(clip_v)},
        "mean_reduction_pct": float((raw_v.mean()-raw_s.mean())/raw_v.mean()*100),
        "p95_reduction_pct": float((np.percentile(raw_v,95)-np.percentile(raw_s,95))/np.percentile(raw_v,95)*100),
        "pct_samples_shorter": float((raw_s < raw_v).mean()*100),
        "pct_truncated_sql_aware": float((raw_s > args.max_len).mean()*100),
        "pct_truncated_vanilla": float((raw_v > args.max_len).mean()*100),
    }
    with open(os.path.join(args.out_dir,"token_length_summary.json"),"w") as f:
        json.dump(out, f, indent=2)

    print("\n=== TOKEN LENGTH (raw, khong truncation) ===")
    print(f"  SQL-aware : mean={out['raw_length']['sql_aware']['mean']:.1f}  "
          f"p95={out['raw_length']['sql_aware']['p95']:.0f}  "
          f"p99={out['raw_length']['sql_aware']['p99']:.0f}")
    print(f"  Vanilla   : mean={out['raw_length']['vanilla']['mean']:.1f}  "
          f"p95={out['raw_length']['vanilla']['p95']:.0f}  "
          f"p99={out['raw_length']['vanilla']['p99']:.0f}")
    print(f"  => mean giam {out['mean_reduction_pct']:+.1f}%  |  "
          f"p95 giam {out['p95_reduction_pct']:+.1f}%  |  "
          f"{out['pct_samples_shorter']:.1f}% mau ngan hon")
    print(f"  => bi truncate (>{args.max_len}): SQL-aware {out['pct_truncated_sql_aware']:.2f}%  "
          f"vs vanilla {out['pct_truncated_vanilla']:.2f}%")
    print(f"\n[OK] luu: {args.out_dir}/token_length_summary.json")

if __name__ == "__main__":
    main()
