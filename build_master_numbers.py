#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_master_numbers.py
==========================================================================
Trich xuat TAT CA con so cho manuscript + response letter tu NGUON THO nhat
(per-sample predictions), gop 5 seed, va dong goi evidence co SHA256 cho reviewer.

Nguon (relative to project root):
  results/mcnemar/predictions_seed_{7,42,99,123,2024}.csv   <- nguon chinh (raw)
  results/mcnemar/mcnemar_results.json
  results/robustness_compare/robustness_delta_summary.csv
  results/robustness_compare/robustness_compare_full.csv
  results/benchmark_results.json           (co BOM -> utf-8-sig)
  results/ablation_results.csv
  results/token_length/token_length_summary.json   (neu da chay)
  data/data_validation_report.csv

Xuat ra:
  results/master/MASTER_NUMBERS.md    <- doc tay, chia theo bang/muc, co provenance
  results/master/MASTER_NUMBERS.json  <- may doc
  results/master/evidence_for_reviewers/  <- copy nguon + manifest SHA256 (zip de nop)

Chay:
  python build_master_numbers.py --root . --released_seed 42
"""
import argparse, csv, glob, hashlib, json, os, shutil, math
import numpy as np

SEEDS = [7, 42, 99, 123, 2024]
N_NORMAL_EXPECTED = 68640
N_SQLI_EXPECTED   = 44113

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()

def load_csv_dicts(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def load_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)

# ------------------------------------------------------------ metrics tu raw
def metrics_from_pred(label, pred, proba=None):
    label = np.asarray(label, int); pred = np.asarray(pred, int)
    tp = int(((pred==1)&(label==1)).sum()); tn = int(((pred==0)&(label==0)).sum())
    fp = int(((pred==1)&(label==0)).sum()); fn = int(((pred==0)&(label==1)).sum())
    prec = tp/(tp+fp) if tp+fp else 0.0
    rec  = tp/(tp+fn) if tp+fn else 0.0
    spec = tn/(tn+fp) if tn+fp else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    fpr  = fp/(fp+tn) if fp+tn else 0.0
    acc  = (tp+tn)/len(label)
    ba   = (rec+spec)/2
    den  = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc  = (tp*tn-fp*fn)/den if den else 0.0
    out = dict(tp=tp, tn=tn, fp=fp, fn=fn, precision=prec, recall_tpr=rec,
               specificity=spec, f1=f1, fpr=fpr, accuracy=acc,
               balanced_accuracy=ba, mcc=mcc)
    if proba is not None:
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            out["auc"] = float(roc_auc_score(label, proba))
            out["ap"]  = float(average_precision_score(label, proba))
        except Exception as e:
            out["auc"] = None; out["ap"] = None
    return out

def agg(values):
    a = np.asarray([v for v in values if v is not None], float)
    if len(a)==0: return None
    n = len(a); mean = float(a.mean()); std = float(a.std(ddof=1)) if n>1 else 0.0
    t = 2.776  # t(0.975, df=4) cho n=5
    half = t*std/math.sqrt(n) if n>1 else 0.0
    return dict(mean=mean, std=std, n=n, min=float(a.min()), max=float(a.max()),
                ci95_low=mean-half, ci95_high=mean+half, ci95_half=half)

def fmt(m, keys):
    return {k: (round(m[k],6) if isinstance(m.get(k),(int,float)) else m.get(k)) for k in keys}

# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--released_seed", type=int, default=42)
    args = ap.parse_args()
    R = args.root
    outdir = os.path.join(R, "results", "master")
    evdir = os.path.join(outdir, "evidence_for_reviewers")
    os.makedirs(evdir, exist_ok=True)

    master = {}
    provenance = []   # (number_group, source_relpath, sha256)
    warnings = []

    def add_source(relpath):
        p = os.path.join(R, relpath)
        if os.path.exists(p):
            dst = os.path.join(evdir, relpath.replace("/", "__").replace("\\","__"))
            shutil.copy2(p, dst)
            provenance.append({"source": relpath, "sha256": sha256(p)})
            return p
        warnings.append(f"THIEU: {relpath}")
        return None

    # ---- 1. DATASET (Table II) --------------------------------------------
    p = add_source("data/data_validation_report.csv")
    if p:
        rows = load_csv_dicts(p)
        ds = {r["file"]: r for r in rows}
        master["dataset"] = {
            "total": int(float(ds["TOTAL"]["rows_usable"])),
            "total_normal": int(float(ds["TOTAL"]["normal"])),
            "total_sqli": int(float(ds["TOTAL"]["sqli"])),
            "splits": {k.split("/")[-1].replace("_dataset.csv",""):
                       {"n": int(float(v["rows_usable"])),
                        "normal": int(float(v["normal"])), "sqli": int(float(v["sqli"]))}
                       for k, v in ds.items() if k != "TOTAL"},
            "duplicates": sum(int(float(v["duplicate_text"])) for k,v in ds.items() if k!="TOTAL"),
        }

    # ---- 2+3. DETECTION tu RAW predictions (proposed & vanilla, per seed) --
    per_seed = {"proposed": {}, "vanilla": {}}
    for s in SEEDS:
        p = add_source(f"results/mcnemar/predictions_seed_{s}.csv")
        if not p: continue
        rows = load_csv_dicts(p)
        label = [int(float(r["label"])) for r in rows]
        for who, pref in [("proposed","proposed"), ("vanilla","vanilla")]:
            pred  = [int(float(r[f"pred_{pref}"]))   for r in rows]
            proba = [float(r[f"proba_{pref}"])       for r in rows]
            per_seed[who][s] = metrics_from_pred(label, pred, proba)
        # kiem tra ve sinh du lieu
        n_norm = sum(1 for x in label if x==0); n_sqli = sum(1 for x in label if x==1)
        if n_norm != N_NORMAL_EXPECTED or n_sqli != N_SQLI_EXPECTED:
            warnings.append(f"seed {s}: phan bo test la {n_norm}/{n_sqli}, ky vong {N_NORMAL_EXPECTED}/{N_SQLI_EXPECTED}")

    KEYS = ["f1","auc","ap","recall_tpr","fpr","precision","specificity",
            "balanced_accuracy","mcc","accuracy","tp","tn","fp","fn"]
    for who in ("proposed","vanilla"):
        d = per_seed[who]
        if not d: continue
        master[f"{who}_per_seed"] = {s: fmt(d[s], KEYS) for s in d}
        master[f"{who}_aggregate"] = {k: agg([d[s][k] for s in d]) for k in
                                      ["f1","auc","ap","recall_tpr","fpr","balanced_accuracy","mcc","accuracy"]}
        if args.released_seed in d:
            master[f"{who}_released_seed_{args.released_seed}"] = fmt(d[args.released_seed], KEYS)

    # ---- tau-sweep plateau (released seed, proposed) tu proba --------------
    p = os.path.join(R, f"results/mcnemar/predictions_seed_{args.released_seed}.csv")
    if os.path.exists(p):
        rows = load_csv_dicts(p)
        label = np.array([int(float(r["label"])) for r in rows])
        proba = np.array([float(r["proba_proposed"]) for r in rows])
        sweep = {}
        for tau in [0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.57]:
            m = metrics_from_pred(label, (proba>=tau).astype(int))
            sweep[f"{tau:.2f}"] = round(m["f1"],6)
        f1s = list(sweep.values())
        master["threshold_plateau"] = {
            "released_seed": args.released_seed,
            "f1_by_tau": sweep,
            "f1_max_minus_min": round(max(f1s)-min(f1s),6),
            "note": "F1 tren test khi quet tau; bien do nho => plateau"}

    # ---- 5. McNEMAR -------------------------------------------------------
    p = add_source("results/mcnemar/mcnemar_results.json")
    if p:
        mc = load_json(p)
        summ = []
        for r in mc:
            m = r["mcnemar"]; cb = r["class_breakdown"]
            pv = m["Proposed_correct_Vanilla_wrong"]; vp = m["Proposed_wrong_Vanilla_correct"]
            summ.append({
                "seed": r["tag"].replace("seed_",""),
                "discordant": m["discordant"], "P_gt_V": pv, "V_gt_P": vp,
                "p_exact": m["p_exact_binomial"],
                "favors": "proposed" if pv>vp else ("vanilla" if vp>pv else "tie"),
                "significant_0.05": bool(m["p_exact_binomial"] < 0.05),
                "sqli_net_proposed": cb["sqli"]["Proposed_correct_Vanilla_wrong"] - cb["sqli"]["Proposed_wrong_Vanilla_correct"],
                "normal_net_proposed_fewer_fp": cb["normal"]["Proposed_correct_Vanilla_wrong"] - cb["normal"]["Proposed_wrong_Vanilla_correct"],
            })
        n_sig = sum(1 for x in summ if x["significant_0.05"])
        dirs = set(x["favors"] for x in summ if x["significant_0.05"])
        master["mcnemar"] = {
            "per_seed": summ, "n_significant_of_5": n_sig,
            "significant_directions": sorted(dirs),
            "conclusion": ("Khong nhat quan: %d/5 seed significant nhung huong nguoc nhau (%s) "
                           "=> hai model tuong duong thong ke in-distribution; khac biet phan anh "
                           "phuong sai train, khong phai uu the kien truc." %
                           (n_sig, ", ".join(sorted(dirs)) if dirs else "khong"))}

    # ---- 6. ABLATION ------------------------------------------------------
    p = add_source("results/ablation_results.csv")
    if p:
        rows = load_csv_dicts(p)
        def g(r,*ks):
            for k in ks:
                if k in r: return r[k]
            return None
        master["ablation"] = [{"variant": g(r,"Variant","variant_id"), "desc": g(r,"Description","description"),
                               "tau": float(g(r,"tau")), "f1": float(g(r,"F1","f1")), "auc": float(g(r,"AUC","auc")),
                               "fpr": float(g(r,"FPR","fpr"))} for r in rows]
        try:
            vid = lambda r: g(r,"Variant","variant_id") or ""
            f1v0 = next(float(g(r,"F1","f1")) for r in rows if vid(r).startswith("v0"))
            f1v1 = next(float(g(r,"F1","f1")) for r in rows if "sql_tokens" in vid(r))
            master["ablation_note"] = ("SQL tokenizer don le: F1 %.5f -> %.5f (delta %+.5f). "
                "Loi ich den tu label smoothing + calibration, KHONG tu tokenizer."
                % (f1v0, f1v1, f1v1-f1v0))
        except StopIteration:
            pass

    # ---- 7. ROBUSTNESS (honest) ------------------------------------------
    p = add_source("results/robustness_compare/robustness_delta_summary.csv")
    add_source("results/robustness_compare/robustness_compare_full.csv")
    if p:
        rows = load_csv_dicts(p)
        master["robustness"] = [{"model": r["model"],
            "F1_clean": float(r["F1_clean"]), "F1_obf": float(r["F1_obf"]), "dF1": float(r["dF1"]),
            "Rec_clean": float(r["Rec_clean"]), "Rec_obf": float(r["Rec_obf"]), "dRecall": float(r["dRecall"])}
            for r in rows]
        master["robustness_note"] = ("Ca 2 model suy giam ~6-7pt recall duoi obfuscation 2 tang; "
            "SQL-aware KHONG ben hon. Dua vao limitations, THAY bang robustness cu (qua lac quan).")

    # ---- 8. LATENCY / DEPLOYMENT (Table VII) ------------------------------
    p = add_source("results/benchmark_results.json")
    if p:
        b = load_json(p)
        rows = b.get("rows", b if isinstance(b, list) else [])
        tbl = []
        for r in rows:
            tbl.append({"config": r["Deployment Configuration"], "format": r.get("Data Format"),
                        "avg_ms": r["Avg. Latency (ms)"], "p95_ms": r["P95 Latency (ms)"],
                        "size_mb": r["Size (MB)"], "speedup": r["Speedup"]})
        master["latency"] = {"rows": tbl, "n_measured": rows[0].get("Measured Samples") if rows else None,
            "note": "Dung bo nay (7.97ms INT8, full 112,753 mau). Sua 'sub-7ms' -> 'sub-8ms' / 'within sub-10ms budget'."}

    # ---- 9. TOKEN LENGTH (neu co) -----------------------------------------
    p = add_source("results/token_length/token_length_summary.json")
    if p:
        tl = load_json(p)
        red = tl.get("p95_reduction_pct", 0) or 0
        tl["_verdict_meaningful"] = bool(red >= 5.0)
        tl["_verdict_note"] = ("Tokenizer rut ngan chuoi >=5%% (p95) -> co the gop vao latency."
                               if red >= 5.0 else
                               "Tokenizer KHONG rut ngan chuoi (p95 giam %.1f%%) -> KHONG co loi ich latency; "
                               "chi con interpretability." % red)
        master["token_length"] = tl
    else:
        warnings.append("token_length CHUA chay -> chay run_token_length_compare.py (luan diem latency cho tokenizer)")

    # ---- 10. LOSO check ---------------------------------------------------
    loso_found = any(os.path.exists(os.path.join(R,x)) for x in
                     ["results/loso", "results/cross_source", "paper_assets/table_loso.csv"])
    if not loso_found:
        warnings.append("LOSO/cross-source KHONG TIM THAY -> neu bai claim cross-source generalization, "
                        "PHAI tim lai run cu hoac chay lai (day la dong gop cot loi, chua co evidence).")

    # ---- ghi output -------------------------------------------------------
    master["_warnings"] = warnings
    json.dump(master, open(os.path.join(outdir,"MASTER_NUMBERS.json"),"w"), indent=2, ensure_ascii=False)

    # manifest SHA256
    manifest = {"generated_from_root": os.path.abspath(R), "files": provenance,
                "note": "Moi con so trong MASTER_NUMBERS.* truy xuat tu cac file duoi day."}
    json.dump(manifest, open(os.path.join(evdir,"MANIFEST_sha256.json"),"w"), indent=2)

    # ---- MARKDOWN doc tay -------------------------------------------------
    write_markdown(os.path.join(outdir,"MASTER_NUMBERS.md"), master, args.released_seed)

    print("\n[OK] Da tao:")
    print("  ", os.path.join(outdir,"MASTER_NUMBERS.md"))
    print("  ", os.path.join(outdir,"MASTER_NUMBERS.json"))
    print("  ", evdir, "(+ MANIFEST_sha256.json)")
    if warnings:
        print("\n[!] CANH BAO:")
        for w in warnings: print("   -", w)

def write_markdown(path, m, rs):
    L = []
    def w(s=""): L.append(s)
    w("# MASTER NUMBERS — ETASR #20371 (batch-32, thong nhat)")
    w("\n> Moi con so tinh tu per-sample predictions (nguon thô, reviewer tu kiem duoc).")
    w("> Released checkpoint = seed %d. Aggregate = 5 seed (mean +/- std, CI95 t-based).\n" % rs)

    if "dataset" in m:
        d = m["dataset"]
        w("## 1. Dataset (Table II)")
        w(f"- Tong: {d['total']:,} (normal {d['total_normal']:,} / sqli {d['total_sqli']:,}), duplicate={d['duplicates']}")
        for k,v in d["splits"].items():
            w(f"- {k}: {v['n']:,} (normal {v['normal']:,} / sqli {v['sqli']:,})")
        w("")

    def block(title, mkey):
        if mkey not in m: return
        r = m[mkey]
        w(f"### {title}")
        w(f"- F1={r['f1']:.4f}  AUC={r.get('auc') and round(r['auc'],4)}  AP={r.get('ap') and round(r['ap'],4)}")
        w(f"- TPR={r['recall_tpr']*100:.2f}%  FPR={r['fpr']*100:.3f}%  Precision={r['precision']*100:.3f}%")
        w(f"- BA={r['balanced_accuracy']:.4f}  MCC={r['mcc']:.4f}  Acc={r['accuracy']:.4f}")
        w(f"- Confusion: TP={r['tp']} TN={r['tn']} FP={r['fp']} FN={r['fn']}")
        w("")

    w("## 2. Released checkpoint (seed %d) — so headline single-run" % rs)
    block(f"Proposed (seed {rs})", f"proposed_released_seed_{rs}")
    block(f"Vanilla identical-recipe (seed {rs})", f"vanilla_released_seed_{rs}")

    def aggblock(title, mkey):
        if mkey not in m: return
        a = m[mkey]
        w(f"### {title}")
        for k,lab in [("f1","F1"),("auc","AUC"),("ap","AP"),("recall_tpr","TPR"),
                      ("fpr","FPR"),("balanced_accuracy","BA"),("mcc","MCC")]:
            g = a.get(k)
            if g: w(f"- {lab} = {g['mean']:.4f} +/- {g['std']:.4f}  (CI95 [{g['ci95_low']:.4f}, {g['ci95_high']:.4f}], min {g['min']:.4f}, max {g['max']:.4f})")
        w("")

    w("## 3. Multi-seed aggregate (Table R1)")
    aggblock("Proposed — 5 seed", "proposed_aggregate")
    aggblock("Vanilla identical-recipe — 5 seed", "vanilla_aggregate")

    if "threshold_plateau" in m:
        tp = m["threshold_plateau"]
        w("## 4. Threshold plateau (deployment-relevant)")
        w(f"- F1 theo tau (seed {tp['released_seed']}): " + ", ".join(f"{k}:{v:.4f}" for k,v in tp["f1_by_tau"].items()))
        w(f"- Bien do F1 (max-min) = {tp['f1_max_minus_min']:.4f}  => plateau, robust voi lua chon threshold\n")

    if "mcnemar" in m:
        mc = m["mcnemar"]
        w("## 5. McNemar significance (5 seed) — Reviewer M")
        w("| seed | discordant | P>V | V>P | p(exact) | favors | sig? |")
        w("|---|---|---|---|---|---|---|")
        for x in mc["per_seed"]:
            w(f"| {x['seed']} | {x['discordant']} | {x['P_gt_V']} | {x['V_gt_P']} | {x['p_exact']:.3g} | {x['favors']} | {'YES' if x['significant_0.05'] else 'no'} |")
        w(f"\n**Ket luan:** {mc['conclusion']}\n")

    if "ablation" in m:
        w("## 6. Ablation")
        w("| variant | tau | F1 | AUC | FPR |")
        w("|---|---|---|---|---|")
        for a in m["ablation"]:
            w(f"| {a['variant']} | {a['tau']:.2f} | {a['f1']:.5f} | {a['auc']:.5f} | {a['fpr']:.5f} |")
        if "ablation_note" in m: w(f"\n> {m['ablation_note']}\n")

    if "robustness" in m:
        w("## 7. Robustness — obfuscation (honest, Reviewer Q)")
        w("| model | F1 clean | F1 obf | dF1 | Rec clean | Rec obf | dRecall |")
        w("|---|---|---|---|---|---|---|")
        for r in m["robustness"]:
            w(f"| {r['model']} | {r['F1_clean']:.4f} | {r['F1_obf']:.4f} | {r['dF1']:.4f} | {r['Rec_clean']:.4f} | {r['Rec_obf']:.4f} | {r['dRecall']:.4f} |")
        w(f"\n> {m['robustness_note']}\n")

    if "latency" in m:
        w("## 8. Latency / deployment (Table VII)")
        w("| config | format | avg ms | p95 ms | size MB | speedup |")
        w("|---|---|---|---|---|---|")
        for r in m["latency"]["rows"]:
            w(f"| {r['config']} | {r['format']} | {r['avg_ms']:.2f} | {r['p95_ms']:.2f} | {r['size_mb']:.1f} | {r['speedup']:.2f}x |")
        w(f"\n> {m['latency']['note']}\n")

    if "token_length" in m:
        t = m["token_length"]
        w("## 9. Token length (tokenizer -> latency)")
        w(f"- mean giam {t.get('mean_reduction_pct',0):.1f}%, p95 giam {t.get('p95_reduction_pct',0):.1f}%, "
          f"{t.get('pct_samples_shorter',0):.1f}% mau ngan hon")
        w(f"- **{t.get('_verdict_note','')}**")
        w("")

    w("## 10. CLAIM DUOC / KHONG DUOC — chot")
    w("**DUOC dua vao bai (co evidence, trung thuc):**")
    w("- Multi-seed stability: F1 ~0.992 +/- 0.0008 over 5 seeds (Reviewer J).")
    w("- Threshold calibration + plateau (deployment-relevant).")
    w("- INT8 deployment: ~8ms CPU, ~4.6x speedup, 120MB, within sub-10ms budget.")
    w("- Leakage-filtered multi-source evaluation protocol.")
    w("- Honest significance testing: no in-dist superiority claimed (integrity, disarms Reviewer M).")
    tl = m.get("token_length")
    if tl and tl.get("_verdict_meaningful"):
        w("- Tokenizer rut ngan chuoi -> gop vao latency (token-length confirmed).")
    w("\n**KHONG duoc claim (chet qua multi-seed / ablation):**")
    w("- In-distribution superiority over baselines (RoBERTa/vanilla ngang hoac hon).")
    w("- 'Best F1/AUC among baselines'.")
    w("- FPR reduction vs vanilla (doi dau qua seed).")
    w("- Error-profile shift 'doi FP lay FN' (chi dung seed 42, doi dau seed khac).")
    w("- Robustness to obfuscation as an advantage (ca 2 suy giam nhu nhau).")
    w("- SQL tokenizer cai thien accuracy (ablation: don le lam GIAM F1).")
    if tl and not tl.get("_verdict_meaningful"):
        w("- SQL tokenizer rut ngan chuoi / giam latency (token-length: giong het vanilla).")

    if m.get("_warnings"):
        w("\n## CANH BAO can xu ly")
        for x in m["_warnings"]: w(f"- {x}")

    open(path,"w",encoding="utf-8").write("\n".join(L))

if __name__ == "__main__":
    main()
