#!/usr/bin/env python3
"""
postprocess_multiseed.py
------------------------
Create reviewer-facing statistics from results/multiseed/summary.json.

Outputs:
  results/multiseed/summary_extended.csv
  results/multiseed/summary_extended.json
  results/multiseed/summary_extended.tex
  paper_assets/table_multiseed_extended_metrics.csv/.md/.tex
  paper_assets/fig_multiseed_error_bars.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    "f1",
    "auc",
    "ap",
    "tpr",
    "fpr",
    "balanced_accuracy",
    "mcc",
    "tau_star",
    "f1_at_0_5",
]

LABELS = {
    "f1": "F1",
    "auc": "ROC-AUC",
    "ap": "AP",
    "tpr": "TPR",
    "fpr": "FPR",
    "balanced_accuracy": "Balanced Accuracy",
    "mcc": "MCC",
    "tau_star": "tau*",
    "f1_at_0_5": "F1 @ 0.5",
}


def t_critical_95(n: int) -> float:
    if n <= 1:
        return float("nan")
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df=n - 1))
    except Exception:
        # Common values; fallback is conservative for n not listed.
        return {
            2: 12.706,
            3: 4.303,
            4: 3.182,
            5: 2.776,
            6: 2.571,
            7: 2.447,
            8: 2.365,
            9: 2.306,
            10: 2.262,
        }.get(n, 1.96)


def mcc_from_counts(tn: float, fp: float, fn: float, tp: float) -> float:
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn) - (fp * fn)) / denom if denom else float("nan")


def read_threshold_counts(results_dir: Path, seed: int) -> dict[str, float]:
    p = results_dir / f"seed_{seed}" / "threshold.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    cm = d.get("test_confusion_matrix_calibrated", {})
    if not cm:
        return {}
    tn, fp, fn, tp = [float(cm.get(k, 0)) for k in ("TN", "FP", "FN", "TP")]
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tpr": tpr,
        "fpr": fpr,
        "balanced_accuracy": (tpr + (1.0 - fpr)) / 2.0,
        "mcc": mcc_from_counts(tn, fp, fn, tp),
    }


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        vals = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        crit = t_critical_95(len(vals))
        half = float(crit * std / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        rows.append({
            "metric": metric,
            "label": LABELS.get(metric, metric),
            "n": int(len(vals)),
            "mean": mean,
            "std": std,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "ci95_low": mean - half,
            "ci95_high": mean + half,
            "ci95_half_width": half,
        })
    return pd.DataFrame(rows)


def write_outputs(df_seed: pd.DataFrame, df_agg: pd.DataFrame, out_dir: Path, assets_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    df_seed.to_csv(out_dir / "summary_extended.csv", index=False, float_format="%.9f")
    df_agg.to_csv(out_dir / "summary_extended_aggregate.csv", index=False, float_format="%.9f")
    payload: dict[str, Any] = {
        "per_seed": df_seed.to_dict(orient="records"),
        "aggregate": df_agg.to_dict(orient="records"),
    }
    (out_dir / "summary_extended.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "summary_extended.tex").write_text(
        df_agg.to_latex(index=False, float_format="%.6f"),
        encoding="utf-8",
    )

    df_agg.to_csv(assets_dir / "table_multiseed_extended_metrics.csv", index=False, float_format="%.9f")
    df_agg.to_markdown(assets_dir / "table_multiseed_extended_metrics.md", index=False)
    (assets_dir / "table_multiseed_extended_metrics.tex").write_text(
        df_agg.to_latex(index=False, float_format="%.6f"),
        encoding="utf-8",
    )

    plot_metrics = ["f1", "auc", "ap", "balanced_accuracy", "mcc", "fpr"]
    plot_df = df_agg[df_agg["metric"].isin(plot_metrics)].copy()
    plot_df["label"] = pd.Categorical(plot_df["label"], [LABELS[m] for m in plot_metrics], ordered=True)
    plot_df = plot_df.sort_values("label")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df["mean"], yerr=plot_df["std"], capsize=5, color="#1f4e79", alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=20, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Five-seed stability with standard-deviation error bars")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(assets_dir / "fig_multiseed_error_bars.png", dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/multiseed/summary.json")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="results/multiseed")
    ap.add_argument("--assets_dir", default="paper_assets")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    results_dir = Path(args.results_dir)
    d = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for seed_raw, values in d.get("per_seed", {}).items():
        seed = int(seed_raw)
        row = {"seed": seed, **values}
        counts = read_threshold_counts(results_dir, seed)
        row.update({k: v for k, v in counts.items() if k in {"tn", "fp", "fn", "tp"}})
        for k in ("tpr", "fpr", "balanced_accuracy", "mcc"):
            if k in counts:
                row[k] = counts[k]
        rows.append(row)
    df_seed = pd.DataFrame(rows).sort_values("seed")
    for metric in METRICS:
        if metric not in df_seed.columns:
            df_seed[metric] = np.nan
    df_agg = aggregate(df_seed)
    write_outputs(df_seed, df_agg, Path(args.out_dir), Path(args.assets_dir))
    print(df_agg.to_string(index=False))


if __name__ == "__main__":
    main()
