"""
run_multiseed.py
----------------
Run the proposed CodeBERT + SQL-aware tokenizer training across multiple random
seeds and report the mean +/- std of the main test-set metrics, so the paper can
report robustness instead of a single-seed point estimate.

For each seed S the proposed model is trained in a *fresh* subprocess (clean CUDA
context, no state leakage) into its own directory:

  results/seed_<S>/best_model_sql_aware/   checkpoint + tokenizer
  results/seed_<S>/history.json            per-epoch losses
  results/seed_<S>/threshold.json          calibrated tau* + test metrics
  results/seed_<S>/{val,test}_predictions.npz

After all seeds finish, this script aggregates the per-seed threshold.json files
and writes:

  results/multiseed/summary.json           per-seed values + mean/std for each metric
  results/multiseed/summary.csv            tidy per-seed table (one row per seed)
  results/multiseed/summary_table.tex      LaTeX-ready "mean +/- std" row for the paper

Usage
-----
  # Default seeds from config.SEEDS (7, 42, 99, 123, 2024), default 5 epochs / batch 32
  python run_multiseed.py

  # Custom seeds and Kaggle-style settings
  python run_multiseed.py --seeds 42,123,2024,7,99 --epochs 5 --batch_size 32 --num_workers 0

  # Ablation: disable SQL-aware tokenizer for all seeds
  python run_multiseed.py --no_sql_tokens

  # Skip seeds that already produced a threshold.json (resume after a crash)
  python run_multiseed.py --skip_existing

  # Only re-aggregate from existing per-seed results, do not train
  python run_multiseed.py --aggregate_only
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from typing import Dict, List, Optional

import config

# Metrics pulled from each seed's threshold.json. Keys are the names used in the
# summary; values describe where to read them from the threshold.json structure.
METRIC_KEYS = ["f1", "auc", "ap", "tpr", "fpr", "balanced_accuracy", "mcc", "tau_star", "f1_at_0_5"]
PRETTY = {
    "f1": "F1 @ tau*",
    "auc": "AUC",
    "ap": "AP",
    "tpr": "TPR",
    "fpr": "FPR",
    "balanced_accuracy": "BA",
    "mcc": "MCC",
    "tau_star": "tau*",
    "f1_at_0_5": "F1 @ 0.5",
}


# ── Extract metrics from one seed's threshold.json ──────────────────────────

def read_seed_metrics(threshold_path: str) -> Dict[str, float]:
    """Parse a single threshold.json into the flat metric dict used downstream."""
    with open(threshold_path) as f:
        d = json.load(f)

    calib = d.get("test_metrics_at_calibrated_tau", {})
    default = d.get("test_metrics_at_default_tau_0_5", {})
    cm = d.get("test_confusion_matrix_calibrated", {})
    tn = float(cm.get("TN", 0))
    fp = float(cm.get("FP", 0))
    fn = float(cm.get("FN", 0))
    tp = float(cm.get("TP", 0))
    tpr = tp / (tp + fn) if (tp + fn) else float(cm.get("TPR", float("nan")))
    fpr = fp / (fp + tn) if (fp + tn) else float(cm.get("FPR", float("nan")))
    balanced_accuracy = (tpr + (1.0 - fpr)) / 2.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else float("nan")

    return {
        "f1": float(calib.get("f1")),
        "auc": float(calib.get("auc")),
        "ap": float(calib.get("ap")),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "balanced_accuracy": float(balanced_accuracy),
        "mcc": float(mcc),
        "tau_star": float(d.get("tau_star_from_val_F1")),
        "f1_at_0_5": float(default.get("f1")) if default.get("f1") is not None else float("nan"),
    }


# ── Aggregation (pure function, unit-testable without a GPU) ─────────────────

def aggregate(per_seed: Dict[int, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Compute mean / sample-std (ddof=1) / min / max for each metric across seeds.

    `per_seed` maps seed -> {metric: value}. Returns {metric: {mean, std, ...}}.
    With a single seed, std is reported as 0.0 (undefined sample std).
    """
    out: Dict[str, Dict[str, float]] = {}
    for m in METRIC_KEYS:
        vals = [per_seed[s][m] for s in per_seed
                if m in per_seed[s] and not math.isnan(per_seed[s][m])]
        if not vals:
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out[m] = {
            "mean": mean,
            "std": std,
            "min": min(vals),
            "max": max(vals),
            "n": len(vals),
        }
    return out


# ── Output writers ───────────────────────────────────────────────────────────

def write_summary(per_seed: Dict[int, Dict[str, float]],
                  agg: Dict[str, Dict[str, float]],
                  out_dir: str,
                  run_meta: Dict) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1) JSON: machine-readable, full detail
    summary = {
        "run_meta": run_meta,
        "seeds": sorted(per_seed.keys()),
        "per_seed": per_seed,
        "aggregate": agg,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 2) CSV: one row per seed, plus mean/std rows
    csv_path = os.path.join(out_dir, "summary.csv")
    cols = ["seed"] + METRIC_KEYS
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for s in sorted(per_seed.keys()):
            row = [str(s)] + [f"{per_seed[s].get(m, float('nan')):.6f}" for m in METRIC_KEYS]
            f.write(",".join(row) + "\n")
        f.write(",".join(["mean"] + [f"{agg.get(m, {}).get('mean', float('nan')):.6f}"
                                     for m in METRIC_KEYS]) + "\n")
        f.write(",".join(["std"] + [f"{agg.get(m, {}).get('std', float('nan')):.6f}"
                                    for m in METRIC_KEYS]) + "\n")

    # 3) LaTeX: a "mean +/- std" row ready to paste into the paper.
    #    Percentages for TPR/FPR, 4 decimals for F1/AUC/AP.
    def fmt(metric: str, scale_pct: bool = False, dec: int = 4) -> str:
        a = agg.get(metric)
        if not a:
            return "-"
        mean, std = a["mean"], a["std"]
        if scale_pct:
            return f"{mean*100:.2f} $\\pm$ {std*100:.2f}"
        return f"{mean:.{dec}f} $\\pm$ {std:.{dec}f}"

    tex = (
        "% Multi-seed result row for the proposed model (mean $\\pm$ std over "
        f"{len(per_seed)} seeds: {sorted(per_seed.keys())})\n"
        "CodeBERT + SQL-aware tok. (ours) & "
        f"{fmt('f1')} & {fmt('auc')} & {fmt('fpr', scale_pct=True, dec=2)} \\\\\n"
        "% Extended columns: TPR(\\%) = " + fmt('tpr', scale_pct=True, dec=2)
        + ", AP = " + fmt('ap')
        + ", tau* = " + fmt('tau_star', dec=3) + "\n"
    )
    with open(os.path.join(out_dir, "summary_table.tex"), "w") as f:
        f.write(tex)


def print_table(per_seed: Dict[int, Dict[str, float]],
                agg: Dict[str, Dict[str, float]]) -> None:
    seeds = sorted(per_seed.keys())
    headers = ["seed"] + [PRETTY[m] for m in METRIC_KEYS]
    widths = [max(len(h), 12) for h in headers]

    def fmt_row(label, values):
        cells = [str(label).ljust(widths[0])]
        for v, w in zip(values, widths[1:]):
            cells.append((f"{v:.4f}" if not math.isnan(v) else "nan").rjust(w))
        return "  ".join(cells)

    print("\n" + "=" * 90)
    print("MULTI-SEED SUMMARY (proposed model)")
    print("=" * 90)
    print("  ".join(h.rjust(w) if i else h.ljust(w)
                    for i, (h, w) in enumerate(zip(headers, widths))))
    print("-" * 90)
    for s in seeds:
        print(fmt_row(s, [per_seed[s].get(m, float("nan")) for m in METRIC_KEYS]))
    print("-" * 90)
    print(fmt_row("mean", [agg.get(m, {}).get("mean", float("nan")) for m in METRIC_KEYS]))
    print(fmt_row("std",  [agg.get(m, {}).get("std", float("nan")) for m in METRIC_KEYS]))
    print("=" * 90)
    # Headline line in the same phrasing as the paper.
    if all(m in agg for m in ("f1", "auc", "fpr")):
        f1, auc, fpr = agg["f1"], agg["auc"], agg["fpr"]
        print(f"\nProposed model over {len(seeds)} seeds: "
              f"F1 = {f1['mean']:.4f} +/- {f1['std']:.4f}, "
              f"AUC = {auc['mean']:.4f} +/- {auc['std']:.4f}, "
              f"FPR = {fpr['mean']*100:.2f}% +/- {fpr['std']*100:.2f}%")


# ── Training driver ──────────────────────────────────────────────────────────

def run_one_seed(seed: int, args) -> str:
    """Launch train_codebert.py for one seed in a fresh subprocess.

    Returns the path to that seed's threshold.json.
    """
    results_dir = config.seed_results_dir(seed)
    model_dir = config.seed_model_dir(seed)
    os.makedirs(results_dir, exist_ok=True)
    threshold_path = os.path.join(results_dir, "threshold.json")

    if args.skip_existing and os.path.exists(threshold_path):
        print(f"[seed {seed}] skip_existing: found {threshold_path}, skipping training.")
        return threshold_path

    cmd = [
        sys.executable, os.path.join(config.PROJECT_ROOT, "train_codebert.py"),
        "--seed", str(seed),
        "--results_dir", results_dir,
        "--output_dir", model_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--max_len", str(args.max_len),
        "--num_workers", str(args.num_workers),
        "--lr", str(args.lr),
        "--model_name", args.model_name,
    ]
    if args.no_sql_tokens:
        cmd.append("--no_sql_tokens")

    print("\n" + "#" * 90)
    print(f"# SEED {seed}  ->  {results_dir}")
    print("# " + " ".join(cmd))
    print("#" * 90, flush=True)

    subprocess.run(cmd, check=True)

    if not os.path.exists(threshold_path):
        raise RuntimeError(
            f"[seed {seed}] training finished but {threshold_path} was not created."
        )
    return threshold_path


def parse_seeds(raw: Optional[str]) -> List[int]:
    if not raw:
        return list(config.SEEDS)
    return [int(x) for x in raw.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Multi-seed runner for the proposed SQLi model.")
    ap.add_argument("--seeds", default=None,
                    help="Comma-separated seeds. Default: config.SEEDS "
                         f"({','.join(map(str, config.SEEDS))}).")
    ap.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--max_len", type=int, default=config.MAX_SEQ_LEN)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--model_name", default=config.BASE_MODEL_NAME)
    ap.add_argument("--no_sql_tokens", action="store_true",
                    help="Ablation: disable the SQL-aware vocabulary for every seed.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip seeds whose threshold.json already exists.")
    ap.add_argument("--aggregate_only", action="store_true",
                    help="Do not train; only aggregate existing per-seed results.")
    ap.add_argument("--out_dir", default=os.path.join(config.RESULTS_DIR, "multiseed"),
                    help="Where to write the aggregated summary files.")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    config.ensure_dirs()
    print(f"[multiseed] seeds = {seeds}")
    print(f"[multiseed] epochs={args.epochs} batch_size={args.batch_size} "
          f"max_len={args.max_len} sql_tokens={not args.no_sql_tokens}")

    per_seed: Dict[int, Dict[str, float]] = {}
    for seed in seeds:
        if args.aggregate_only:
            tp = os.path.join(config.seed_results_dir(seed), "threshold.json")
            if not os.path.exists(tp):
                print(f"[seed {seed}] aggregate_only: {tp} missing, skipping.")
                continue
        else:
            tp = run_one_seed(seed, args)
        per_seed[seed] = read_seed_metrics(tp)

    if not per_seed:
        print("[multiseed] No seed produced metrics. Nothing to aggregate.")
        sys.exit(1)

    agg = aggregate(per_seed)
    run_meta = {
        "seeds": sorted(per_seed.keys()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.batch_size,
        "max_len": args.max_len,
        "lr": args.lr,
        "model_name": args.model_name,
        "sql_tokens_enabled": not args.no_sql_tokens,
    }
    write_summary(per_seed, agg, args.out_dir, run_meta)
    print_table(per_seed, agg)
    print(f"\n[multiseed] wrote summary.json / summary.csv / summary_table.tex to {args.out_dir}")


if __name__ == "__main__":
    main()
