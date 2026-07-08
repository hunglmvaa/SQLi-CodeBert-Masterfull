#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_obfuscation_compare.py
--------------------------------------------------------------------------
So sanh do ben (robustness) truoc obfuscation cua BA model tren CUNG mot bo
du lieu obfuscated da luu ra dia (tai lap duoc):
    (1) Proposed  : CodeBERT + SQL-aware tokenizer   (results/best_model_sql_aware)
    (2) Vanilla   : CodeBERT-base (khong SQL token)   (checkpoint baseline codebert)
    (3) RoBERTa   : roberta-base                       (checkpoint baseline roberta)

Nguyen tac:
  - Transform TUONG MINH, sinh tu SQLi cua test set, LUU ra CSV de reviewer tai lap.
  - Giu CA normal LAN SQLi -> do duoc ca FP lan FN (khong chi recall).
  - Danh gia moi model o THRESHOLD DA CALIBRATE cua chinh no (khong dat lai tau).
  - Bao cao delta F1 / delta recall giua "clean" va "obfuscated" cho tung model.
    Model ben = delta nho; model mong manh = delta lon.

Chay (Windows, tu thu muc goc project):
  python run_obfuscation_compare.py ^
      --test_csv data/test_dataset.csv ^
      --proposed_dir results/best_model_sql_aware ^
      --vanilla_dir  results/baselines_master/codebert_ckpt ^
      --roberta_dir  results/baselines_master/roberta_ckpt ^
      --proposed_tau 0.25 --vanilla_tau 0.2566 --roberta_tau 0.8267 ^
      --out_dir results/robustness_compare --seed 42

Neu checkpoint baseline khong duoc luu (chi luu metrics json) thi xem phan
"KHONG CO CHECKPOINT" o cuoi file.
"""
import argparse, json, os, random, re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ----------------------------------------------------------------------------
# 1. CAC PHEP OBFUSCATION - tuong minh, moi phep la mot ky thuat WAF-bypass
#    kinh dien. Moi phep BAO TOAN ngu nghia tan cong (payload van la SQLi that).
# ----------------------------------------------------------------------------
SQL_KEYWORDS = ["SELECT", "UNION", "OR", "AND", "WHERE", "FROM", "INSERT",
                "UPDATE", "DELETE", "DROP", "SLEEP", "BENCHMARK", "CONCAT",
                "ORDER", "GROUP", "HAVING", "EXEC", "NULL", "LIKE"]

def obf_inline_comment(s: str) -> str:
    """UNION SELECT -> UN/**/ION SE/**/LECT  (chen comment giua keyword)."""
    def bust(m):
        w = m.group(0)
        i = max(1, len(w) // 2)
        return w[:i] + "/**/" + w[i:]
    pat = re.compile("|".join(SQL_KEYWORDS), re.IGNORECASE)
    return pat.sub(bust, s)

def obf_case_random(s: str) -> str:
    """SELECT -> sElEcT  (dao hoa/thuong ngau nhien tren ky tu chu)."""
    return "".join(c.upper() if (c.isalpha() and random.random() < 0.5) else c.lower()
                   if c.isalpha() else c for c in s)

def obf_url_encode(s: str) -> str:
    """Ma hoa URL mot phan cac ky tu dac trung SQLi ( ' = space ( ) )."""
    table = {"'": "%27", "\"": "%22", " ": "%20", "=": "%3D",
             "(": "%28", ")": "%29", "-": "%2D", ";": "%3B"}
    out = []
    for c in s:
        out.append(table[c] if (c in table and random.random() < 0.6) else c)
    return "".join(out)

def obf_whitespace(s: str) -> str:
    """Thay space bang tab / newline / comment - WAF bo qua whitespace variant."""
    reps = ["\t", "\n", "/**/", "  ", " "]
    return re.sub(r" ", lambda _: random.choice(reps), s)

def obf_keyword_split_plus(s: str) -> str:
    """OR 1=1 -> OR+1=1 ; noi bang '+' thay space (kieu HTTP param)."""
    return s.replace(" ", "+")

TRANSFORMS = {
    "inline_comment":   obf_inline_comment,
    "case_random":      obf_case_random,
    "url_encode":       obf_url_encode,
    "whitespace_sub":   obf_whitespace,
    "plus_concat":      obf_keyword_split_plus,
}

def obfuscate(text: str, n_layers: int = 2) -> str:
    """Ap 1..n phep ngau nhien lien tiep (obfuscation nhieu tang, giong thuc te)."""
    keys = random.sample(list(TRANSFORMS.keys()), k=min(n_layers, len(TRANSFORMS)))
    for k in keys:
        text = TRANSFORMS[k](text)
    return text


# ----------------------------------------------------------------------------
# 2. INFERENCE
# ----------------------------------------------------------------------------
class TxtDS(Dataset):
    def __init__(self, texts, tok, max_len):
        self.enc = tok(list(texts), truncation=True, max_length=max_len,
                       padding=False, return_tensors=None)
    def __len__(self): return len(self.enc["input_ids"])
    def __getitem__(self, i):
        return {k: self.enc[k][i] for k in self.enc}

def collate(batch, tok):
    keys = batch[0].keys()
    out = {}
    maxlen = max(len(b["input_ids"]) for b in batch)
    for k in keys:
        pad = tok.pad_token_id if k == "input_ids" else 0
        out[k] = torch.tensor([b[k] + [pad] * (maxlen - len(b[k])) for b in batch])
    return out

@torch.no_grad()
def predict_proba(model_dir, texts, max_len=256, batch_size=64, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    ds = TxtDS(texts, tok, max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=lambda b: collate(b, tok))
    probs = []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        p = torch.softmax(logits, dim=-1)[:, 1]   # P(class=SQLi)
        probs.append(p.cpu().numpy())
    return np.concatenate(probs)

def metrics_at_tau(y_true, proba, tau):
    y_pred = (proba >= tau).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr  = fp / (fp + tn) if fp + tn else 0.0
    acc  = (tp + tn) / len(y_true)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, precision=prec, recall_tpr=rec,
                f1=f1, fpr=fpr, accuracy=acc)


# ----------------------------------------------------------------------------
# 3. MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--proposed_dir", required=True)
    ap.add_argument("--vanilla_dir", required=True)
    ap.add_argument("--roberta_dir", default=None, help="tuy chon; bo qua neu chua co checkpoint")
    ap.add_argument("--proposed_tau", type=float, required=True)
    ap.add_argument("--vanilla_tau", type=float, required=True)
    ap.add_argument("--roberta_tau", type=float, default=None)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--n_sqli", type=int, default=5000, help="so mau SQLi lay ra de obfuscate")
    ap.add_argument("--n_normal", type=int, default=5000, help="so mau normal giu nguyen")
    ap.add_argument("--n_layers", type=int, default=2, help="so tang obfuscation")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--out_dir", default="results/robustness_compare")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Xay bo danh gia: normal giu nguyen + SQLi (clean & obfuscated song song)
    df = pd.read_csv(args.test_csv)
    sqli = df[df[args.label_col] == 1].sample(n=min(args.n_sqli, (df[args.label_col]==1).sum()),
                                              random_state=args.seed).reset_index(drop=True)
    normal = df[df[args.label_col] == 0].sample(n=min(args.n_normal, (df[args.label_col]==0).sum()),
                                                random_state=args.seed).reset_index(drop=True)

    sqli_clean = sqli[args.text_col].astype(str).tolist()
    sqli_obf   = [obfuscate(t, args.n_layers) for t in sqli_clean]
    normal_txt = normal[args.text_col].astype(str).tolist()

    # Luu bo obfuscated de tai lap (reviewer co the kiem tra tung mau)
    pd.DataFrame({"clean": sqli_clean, "obfuscated": sqli_obf}).to_csv(
        os.path.join(args.out_dir, "obfuscated_pairs.csv"), index=False)

    # Hai tap danh gia dung chung normal:
    #   CLEAN     = normal + sqli_clean
    #   OBFUSCATED= normal + sqli_obf
    y = np.array([0]*len(normal_txt) + [1]*len(sqli_clean))
    X_clean = normal_txt + sqli_clean
    X_obf   = normal_txt + sqli_obf

    models = {
        "Proposed (SQL-aware)": (args.proposed_dir, args.proposed_tau),
        "CodeBERT vanilla":     (args.vanilla_dir,  args.vanilla_tau),
    }
    if args.roberta_dir and args.roberta_tau is not None:
        models["RoBERTa-base"] = (args.roberta_dir, args.roberta_tau)

    rows = []
    for name, (mdir, tau) in models.items():
        print(f"[+] {name}  (tau={tau})")
        p_clean = predict_proba(mdir, X_clean, args.max_len)
        p_obf   = predict_proba(mdir, X_obf,   args.max_len)
        m_clean = metrics_at_tau(y, p_clean, tau)
        m_obf   = metrics_at_tau(y, p_obf,   tau)
        rows.append(dict(model=name, tau=tau, setting="clean", **m_clean))
        rows.append(dict(model=name, tau=tau, setting="obfuscated", **m_obf))
        print(f"    F1 clean={m_clean['f1']:.4f}  obf={m_obf['f1']:.4f}  "
              f"dF1={m_clean['f1']-m_obf['f1']:+.4f}   "
              f"recall clean={m_clean['recall_tpr']:.4f}  obf={m_obf['recall_tpr']:.4f}  "
              f"dRec={m_clean['recall_tpr']-m_obf['recall_tpr']:+.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.out_dir, "robustness_compare_full.csv"), index=False)

    # Bang delta: cot quyet dinh la do SUY GIAM (model ben = suy giam nho)
    piv = out.pivot_table(index="model", columns="setting", values=["f1", "recall_tpr", "fpr"])
    delta = pd.DataFrame({
        "F1_clean":  piv[("f1","clean")],
        "F1_obf":    piv[("f1","obfuscated")],
        "dF1":       piv[("f1","clean")] - piv[("f1","obfuscated")],
        "Rec_clean": piv[("recall_tpr","clean")],
        "Rec_obf":   piv[("recall_tpr","obfuscated")],
        "dRecall":   piv[("recall_tpr","clean")] - piv[("recall_tpr","obfuscated")],
    }).round(4)
    delta.to_csv(os.path.join(args.out_dir, "robustness_delta_summary.csv"))
    print("\n=== DELTA SUMMARY (suy giam cang nho cang ben) ===")
    print(delta.to_string())

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\n[OK] Ket qua luu tai: {args.out_dir}")

if __name__ == "__main__":
    main()
