#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_mcnemar.py
--------------------------------------------------------------------------
McNemar paired test giua Proposed (SQL-aware) va Vanilla (identical-recipe),
tren CUNG test set, moi model o THRESHOLD tau* cua chinh no.

Lam 3 viec:
  1. Inference ca 2 checkpoint -> nhan du doan tung mau (LUU ra dia).
  2. McNemar: bang 2x2 concordant/discordant + p-value
       - exact binomial (khuyen dung) + chi-square co hieu chinh lien tuc.
  3. Tach discordant THEO LOP -> cau chuyen "doi FP lay FN" o muc tung mau:
       - trong nhom normal: proposed sai / vanilla dung  vs  nguoc lai
       - trong nhom SQLi  : proposed sai / vanilla dung  vs  nguoc lai

Che do:
  A) MOT cap (mac dinh, seed 42 released):
     python run_mcnemar.py --test_csv data/test_dataset.csv ^
       --proposed_dir results/best_model_sql_aware --proposed_tau 0.25 ^
       --vanilla_dir  results_nosql/seed_42/best_model_sql_aware --vanilla_tau 0.39 ^
       --out_dir results/mcnemar

  B) NHIEU seed (neu co du checkpoint 5 seed): dua vao file JSON --pairs_json
     [{"seed":7,"proposed_dir":"...","proposed_tau":0.xx,
       "vanilla_dir":"...","vanilla_tau":0.28}, ...]
"""
import argparse, json, os, math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------------------------------------------------------- inference
class TxtDS(Dataset):
    def __init__(self, texts, tok, max_len):
        self.enc = tok(list(texts), truncation=True, max_length=max_len,
                       padding=False, return_tensors=None)
    def __len__(self): return len(self.enc["input_ids"])
    def __getitem__(self, i): return {k: self.enc[k][i] for k in self.enc}

def collate(batch, tok):
    maxlen = max(len(b["input_ids"]) for b in batch)
    out = {}
    for k in batch[0]:
        pad = tok.pad_token_id if k == "input_ids" else 0
        out[k] = torch.tensor([b[k] + [pad]*(maxlen-len(b[k])) for b in batch])
    return out

@torch.no_grad()
def predict_proba(model_dir, texts, max_len=256, batch_size=64, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    dl = DataLoader(TxtDS(texts, tok, max_len), batch_size=batch_size,
                    shuffle=False, collate_fn=lambda b: collate(b, tok))
    probs = []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        probs.append(torch.softmax(model(**batch).logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


# ---------------------------------------------------------------- statistics
def exact_binom_two_sided(b, c):
    """McNemar exact: nhi thuc(k=min(b,c), n=b+c, p=0.5), hai phia."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # dung normal approx neu n lon de tranh cham/tran so
    if n > 2000:
        z = (abs(b - c) - 1) / math.sqrt(n)           # continuity-corrected
        return math.erfc(z / math.sqrt(2))            # two-sided normal tail
    # exact: 2 * P(X <= k)
    p = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2 * p)

def chi2_sf_1df(x):
    """Survival function cua chi-square 1 df = erfc(sqrt(x/2))."""
    return math.erfc(math.sqrt(x / 2.0)) if x > 0 else 1.0

def mcnemar(y_true, pred_a, pred_b, name_a="A", name_b="B"):
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    n11 = int(( correct_a &  correct_b).sum())   # ca hai dung
    n00 = int((~correct_a & ~correct_b).sum())   # ca hai sai
    n10 = int(( correct_a & ~correct_b).sum())   # A dung, B sai
    n01 = int((~correct_a &  correct_b).sum())   # A sai, B dung
    b, c = n10, n01
    disc = b + c
    chi2 = ((abs(b - c) - 1) ** 2) / disc if disc else 0.0
    return {
        "both_correct": n11, "both_wrong": n00,
        f"{name_a}_correct_{name_b}_wrong": n10,
        f"{name_a}_wrong_{name_b}_correct": n01,
        "discordant": disc,
        "chi2_cc": round(chi2, 4),
        "p_chi2_cc": chi2_sf_1df(chi2),
        "p_exact_binomial": exact_binom_two_sided(b, c),
    }

def class_breakdown(y_true, pred_a, pred_b, name_a, name_b):
    """Tach discordant theo lop -> lo dien trade-off FP<->FN o muc mau."""
    out = {}
    for cls, lab in [("normal", 0), ("sqli", 1)]:
        m = (y_true == lab)
        ca = (pred_a[m] == y_true[m]); cb = (pred_b[m] == y_true[m])
        out[cls] = {
            f"{name_a}_wrong_{name_b}_correct": int((~ca &  cb).sum()),
            f"{name_a}_correct_{name_b}_wrong": int(( ca & ~cb).sum()),
        }
    return out


# ---------------------------------------------------------------- one pair
def run_pair(test_csv, text_col, label_col, max_len,
             proposed_dir, proposed_tau, vanilla_dir, vanilla_tau,
             out_dir, tag):
    df = pd.read_csv(test_csv)
    texts = df[text_col].astype(str).tolist()
    y = df[label_col].to_numpy().astype(int)

    p_prop = predict_proba(proposed_dir, texts, max_len)
    p_van  = predict_proba(vanilla_dir,  texts, max_len)
    pred_prop = (p_prop >= proposed_tau).astype(int)
    pred_van  = (p_van  >= vanilla_tau ).astype(int)

    # luu prediction per-sample (evidence tu nay co file nay)
    pd.DataFrame({
        "label": y,
        "proba_proposed": p_prop, "pred_proposed": pred_prop,
        "proba_vanilla":  p_van,  "pred_vanilla":  pred_van,
    }).to_csv(os.path.join(out_dir, f"predictions_{tag}.csv"), index=False)

    mc = mcnemar(y, pred_prop, pred_van, "Proposed", "Vanilla")
    cb = class_breakdown(y, pred_prop, pred_van, "Proposed", "Vanilla")
    return {"tag": tag, "n": int(len(y)),
            "proposed_tau": proposed_tau, "vanilla_tau": vanilla_tau,
            "mcnemar": mc, "class_breakdown": cb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--out_dir", default="results/mcnemar")
    # che do A: 1 cap
    ap.add_argument("--proposed_dir"); ap.add_argument("--proposed_tau", type=float)
    ap.add_argument("--vanilla_dir");  ap.add_argument("--vanilla_tau", type=float)
    # che do B: nhieu seed
    ap.add_argument("--pairs_json", default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.pairs_json:
        pairs = json.load(open(args.pairs_json, encoding="utf-8"))
    else:
        assert args.proposed_dir and args.vanilla_dir, "Can --proposed_dir/--vanilla_dir hoac --pairs_json"
        pairs = [{"seed": 42, "proposed_dir": args.proposed_dir, "proposed_tau": args.proposed_tau,
                  "vanilla_dir": args.vanilla_dir, "vanilla_tau": args.vanilla_tau}]

    results = []
    for pr in pairs:
        tag = f"seed_{pr['seed']}"
        print(f"\n===== McNemar {tag} =====")
        r = run_pair(args.test_csv, args.text_col, args.label_col, args.max_len,
                     pr["proposed_dir"], pr["proposed_tau"],
                     pr["vanilla_dir"], pr["vanilla_tau"], args.out_dir, tag)
        results.append(r)
        mc = r["mcnemar"]
        print(f"  both_correct={mc['both_correct']}  both_wrong={mc['both_wrong']}")
        print(f"  Proposed>Vanilla (P dung,V sai) = {mc['Proposed_correct_Vanilla_wrong']}")
        print(f"  Vanilla>Proposed (P sai,V dung) = {mc['Proposed_wrong_Vanilla_correct']}")
        print(f"  discordant={mc['discordant']}  chi2_cc={mc['chi2_cc']}")
        print(f"  p(exact binomial)={mc['p_exact_binomial']:.4g}   p(chi2_cc)={mc['p_chi2_cc']:.4g}")
        cb = r["class_breakdown"]
        print(f"  [normal] P sai/V dung={cb['normal']['Proposed_wrong_Vanilla_correct']}  "
              f"P dung/V sai={cb['normal']['Proposed_correct_Vanilla_wrong']}")
        print(f"  [sqli]   P sai/V dung={cb['sqli']['Proposed_wrong_Vanilla_correct']}  "
              f"P dung/V sai={cb['sqli']['Proposed_correct_Vanilla_wrong']}")

    json.dump(results, open(os.path.join(args.out_dir, "mcnemar_results.json"), "w"), indent=2)

    # bang gom neu nhieu seed
    if len(results) > 1:
        rows = [{"seed": r["tag"], "discordant": r["mcnemar"]["discordant"],
                 "P>V": r["mcnemar"]["Proposed_correct_Vanilla_wrong"],
                 "V>P": r["mcnemar"]["Proposed_wrong_Vanilla_correct"],
                 "p_exact": r["mcnemar"]["p_exact_binomial"]} for r in results]
        sig = sum(1 for x in rows if x["p_exact"] < 0.05)
        print(f"\n=== TONG HOP {len(rows)} seed: {sig}/{len(rows)} seed co p<0.05 ===")
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "mcnemar_multiseed_summary.csv"), index=False)

    print(f"\n[OK] luu: {args.out_dir}/")

if __name__ == "__main__":
    main()
