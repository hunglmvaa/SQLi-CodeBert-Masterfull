#!/usr/bin/env python3
"""
paper_assets_builder.py
-----------------------
Generate all paper tables and figures from REAL experiment artifacts.

This script intentionally avoids hard-coded paper numbers. If an artifact is
missing, the corresponding table/figure is skipped and reported in
paper_assets/missing_artifacts.json.

Expected inputs:
  data/train_dataset.csv, data/val_dataset.csv, data/test_dataset.csv
  results/history.json
  results/val_predictions.npz
  results/test_predictions.npz
  results/threshold.json
  results/ablation_results.csv or results/ablation_results.json
  results/benchmark_results.json
  results/baselines_master/all_results.json          optional
  results/baselines_csic2010/all_results.json        optional

Outputs:
  paper_assets/*.csv, *.md, *.tex, *.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

import config

BLUE = "#1f4e79"
GOLD = "#c79100"
GREEN = "#7f9c8a"
ORANGE = "#bf6f4a"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_table(df: pd.DataFrame, out_dir: Path, stem: str, caption: str | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False, float_format="%.6f")
    df.to_markdown(out_dir / f"{stem}.md", index=False)
    # Escape only the most common LaTeX special chars in object columns.
    tex_df = df.copy()
    for col in tex_df.columns:
        if tex_df[col].dtype == object:
            tex_df[col] = tex_df[col].astype(str).str.replace("&", r"\&", regex=False)
    tex = tex_df.to_latex(index=False, escape=False, float_format="%.6f")
    if caption:
        tex = "% " + caption + "\n" + tex
    (out_dir / f"{stem}.tex").write_text(tex, encoding="utf-8")


def split_stats(csv_path: Path) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"])
    labels = df["label"].astype(int)
    return {
        "Samples": int(len(df)),
        "Normal": int((labels == 0).sum()),
        "SQLi": int((labels == 1).sum()),
    }


def make_dataset_table(out_dir: Path) -> pd.DataFrame:
    rows = []
    specs = [
        ("Training", Path(config.TRAIN_CSV), "Model training"),
        ("Validation", Path(config.VAL_CSV), "Model selection and threshold calibration"),
        ("Independent Test", Path(config.TEST_CSV), "Final independent evaluation"),
    ]
    for split, path, purpose in specs:
        s = split_stats(path)
        rows.append({"Split": split, **s, "Purpose": purpose})
    total_samples = sum(r["Samples"] for r in rows)
    for r in rows:
        r["Proportion"] = f"{100 * r['Samples'] / total_samples:.1f}%"
    rows.append({
        "Split": "Total",
        "Samples": sum(r["Samples"] for r in rows),
        "Normal": sum(r["Normal"] for r in rows),
        "SQLi": sum(r["SQLi"] for r in rows),
        "Proportion": "100.0%",
        "Purpose": "Full corpus",
    })
    df = pd.DataFrame(rows)[["Split", "Samples", "Normal", "SQLi", "Proportion", "Purpose"]]
    save_table(df, out_dir, "table_ii_dataset_split_class_distribution", "Dataset split and class distribution")
    return df


def baseline_json_to_rows(path: Path, proposed_metrics: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    data = read_json(path)
    rows: list[dict[str, Any]] = []
    name_map = {
        "modsecurity_crs": "ModSecurity + OWASP CRS-style rules",
        "svm": "TF-IDF + SVM",
        "tfidf_svm": "TF-IDF + Linear SVM calibrated",
        "tfidf_nb": "TF-IDF + ComplementNB",
        "tfidf_logreg": "TF-IDF + Logistic Regression",
        "tfidf_sgd": "TF-IDF + SGDClassifier",
        "bilstm": "Character BiLSTM",
        "cnn": "Character 1D-CNN",
        "codebert": "CodeBERT baseline without SQL-aware tokenizer",
        "bert": "BERT-base",
        "distilbert": "DistilBERT-base",
        "roberta": "RoBERTa-base",
    }
    model_order = [
        "modsecurity_crs", "tfidf_nb", "tfidf_logreg", "tfidf_sgd", "svm", "tfidf_svm",
        "bilstm", "cnn", "bert", "distilbert", "roberta", "codebert",
    ]
    ordered_items = []
    for key in model_order:
        if key in data:
            ordered_items.append((key, data[key]))
    for key, payload in data.items():
        if key not in model_order:
            ordered_items.append((key, payload))
    for key, payload in ordered_items:
        metrics = payload.get("metrics", payload)
        cfg = payload.get("config", {})
        name = cfg.get("baseline") or name_map.get(key, key)
        rows.append({
            "Model": name,
            "Accuracy": metrics.get("accuracy"),
            "Precision": metrics.get("precision"),
            "Recall": metrics.get("recall"),
            "F1": metrics.get("f1"),
            "ROC-AUC": metrics.get("roc_auc", metrics.get("auc")),
            "AP": metrics.get("average_precision", metrics.get("ap")),
            "FPR": metrics.get("fpr"),
            "Latency_ms_per_sample": metrics.get("latency_ms_per_sample"),
        })
    if proposed_metrics:
        rows.append({
            "Model": "Proposed CodeBERT + SQL-aware tokenizer",
            "Accuracy": proposed_metrics.get("accuracy"),
            "Precision": proposed_metrics.get("precision"),
            "Recall": proposed_metrics.get("recall"),
            "F1": proposed_metrics.get("f1"),
            "ROC-AUC": proposed_metrics.get("auc", proposed_metrics.get("roc_auc")),
            "AP": proposed_metrics.get("ap", proposed_metrics.get("average_precision")),
            "FPR": proposed_metrics.get("fpr"),
            "Latency_ms_per_sample": None,
        })
    return rows


def load_threshold(results_dir: Path) -> dict[str, Any] | None:
    p = results_dir / "threshold.json"
    return read_json(p) if p.exists() else None


def proposed_metrics_from_threshold(threshold: dict[str, Any] | None) -> dict[str, Any] | None:
    if not threshold:
        return None
    m = dict(threshold.get("test_metrics_at_calibrated_tau", {}))
    cm = threshold.get("test_confusion_matrix_calibrated", {})
    if cm:
        m["fpr"] = cm.get("FPR")
    return m


def make_baseline_tables(out_dir: Path, results_dir: Path, threshold: dict[str, Any] | None) -> list[str]:
    missing = []
    proposed = proposed_metrics_from_threshold(threshold)
    for stem, path in [
        ("table_iii_baseline_comparison_master", results_dir / "baselines_master" / "all_results.json"),
        ("table_iv_baseline_comparison_csic2010", results_dir / "baselines_csic2010" / "all_results.json"),
    ]:
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.DataFrame(baseline_json_to_rows(path, proposed))
        numeric_cols = [c for c in df.columns if c != "Model"]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        save_table(df, out_dir, stem, stem.replace("_", " ").title())
    return missing


def make_training_history(out_dir: Path, results_dir: Path) -> str | None:
    p = results_dir / "history.json"
    if not p.exists():
        return str(p)
    hist = read_json(p)
    rows = []
    for h in hist:
        vm = h.get("val_metrics_at_0_5", {})
        rows.append({
            "Epoch": h.get("epoch"),
            "Train Loss": h.get("train_loss"),
            "Validation Loss": h.get("val_loss"),
            "Val Accuracy": vm.get("accuracy"),
            "Val Precision": vm.get("precision"),
            "Val Recall": vm.get("recall"),
            "Val F1": vm.get("f1"),
            "Val AUC": vm.get("auc"),
            "Val AP": vm.get("ap"),
            "Seconds": h.get("sec"),
        })
    df = pd.DataFrame(rows)
    save_table(df, out_dir, "table_training_history", "Training and validation metrics by epoch")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(df["Epoch"], df["Train Loss"], "o-", color=BLUE, label="Training loss")
    ax.plot(df["Epoch"], df["Validation Loss"], "s--", color=GOLD, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Cross-entropy loss (label-smoothed, $\epsilon=0.1$)")
    ax.set_title("Training and validation loss during CodeBERT fine-tuning")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_training_curves.png")
    plt.close(fig)
    return None


def make_threshold_assets(out_dir: Path, results_dir: Path, threshold: dict[str, Any] | None) -> list[str]:
    missing = []
    if threshold is None:
        missing.append(str(results_dir / "threshold.json"))
        return missing
    tau = threshold.get("tau_star_from_val_F1")
    default_m = threshold.get("test_metrics_at_default_tau_0_5", {})
    calib_m = threshold.get("test_metrics_at_calibrated_tau", {})
    rows = [
        {"Threshold": "Default tau=0.5", "tau": 0.5, **default_m},
        {"Threshold": "Calibrated tau*", "tau": tau, **calib_m},
    ]
    df = pd.DataFrame(rows)
    save_table(df, out_dir, "table_v_threshold_calibration", "Default and calibrated threshold results")

    cm = threshold.get("test_confusion_matrix_calibrated")
    if cm:
        tn, fp, fn, tp = cm["TN"], cm["FP"], cm["FN"], cm["TP"]
        specificity = tn / (tn + fp) if tn + fp else np.nan
        fpr = fp / (tn + fp) if tn + fp else np.nan
        tpr = tp / (tp + fn) if tp + fn else np.nan
        fnr = fn / (tp + fn) if tp + fn else np.nan
        df_cm = pd.DataFrame([
            {"Metric": "True Negative (TN)", "Value": tn},
            {"Metric": "False Positive (FP)", "Value": fp},
            {"Metric": "False Negative (FN)", "Value": fn},
            {"Metric": "True Positive (TP)", "Value": tp},
            {"Metric": "Specificity", "Value": specificity},
            {"Metric": "False Positive Rate (FPR)", "Value": fpr},
            {"Metric": "True Positive Rate (TPR/Recall)", "Value": tpr},
            {"Metric": "False Negative Rate (FNR)", "Value": fnr},
        ])
        save_table(df_cm, out_dir, "table_vi_confusion_error_rates", "Confusion matrix counts and WAF-oriented rates")

        cm_norm = np.array([[specificity, fpr], [fnr, tpr]], dtype=float)
        fig, ax = plt.subplots(figsize=(5.4, 4.5))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Benign", "SQLi"]); ax.set_yticklabels(["Benign", "SQLi"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(rf"Normalized confusion matrix at $\tau^*={tau:.4f}$")
        for i in range(2):
            for j in range(2):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm_norm[i, j]*100:.2f}%", ha="center", va="center", color=color, fontweight="bold", fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / "fig5_normalized_confusion_matrix.png")
        plt.close(fig)

    curve = threshold.get("val_F1_curve", {})
    if curve:
        th = np.array(curve.get("thresholds", []), dtype=float)
        f1s = np.array(curve.get("F1", []), dtype=float)
        df_curve = pd.DataFrame({"tau": th, "Validation F1": f1s})
        save_table(df_curve, out_dir, "table_threshold_f1_curve", "Validation F1 by decision threshold")
        fig, ax = plt.subplots(figsize=(7, 4.3))
        ax.plot(th, f1s, color=BLUE, lw=2)
        ax.axvline(tau, color=GOLD, ls="--", label=rf"$\tau^*={tau:.4f}$")
        ax.axvline(0.5, color="gray", ls=":", label=r"default $\tau=0.5$")
        ax.set_xlabel(r"Decision threshold $\tau$")
        ax.set_ylabel("F1 score (validation set)")
        ax.set_title("F1 score across decision thresholds")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        ax.set_xlim(0, 1)
        ymin = max(0, min(f1s) - 0.05) if len(f1s) else 0
        ax.set_ylim(ymin, 1.005)
        fig.tight_layout()
        fig.savefig(out_dir / "fig4_f1_vs_threshold.png")
        plt.close(fig)
    return missing


def normalize_benchmark_rows(raw: Any) -> list[dict[str, Any]]:
    entries = raw.get("rows", raw) if isinstance(raw, dict) else raw
    rows = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        name = e.get("Deployment Configuration") or e.get("name") or e.get("configuration")
        dtype = e.get("Data Format") or e.get("dtype") or e.get("format") or ""
        latency = e.get("Avg. Latency (ms)", e.get("latency_ms", e.get("avg_latency_ms")))
        size = e.get("Size (MB)", e.get("size_mb", e.get("model_size_mb")))
        speedup = e.get("Speedup", e.get("speedup"))
        if name is None or latency is None or size is None:
            continue
        rows.append({
            "Deployment Configuration": name,
            "Data Format": dtype,
            "Avg. Latency (ms/sample)": float(latency),
            "Size (MB)": float(size),
            "Speedup": float(str(speedup).replace("×", "").replace("x", "")) if speedup is not None else np.nan,
        })
    if rows and any(pd.isna(r["Speedup"]) for r in rows):
        base = rows[0]["Avg. Latency (ms/sample)"]
        for r in rows:
            if pd.isna(r["Speedup"]):
                r["Speedup"] = base / r["Avg. Latency (ms/sample)"]
    return rows


def make_benchmark_assets(out_dir: Path, results_dir: Path) -> str | None:
    p = results_dir / "benchmark_results.json"
    if not p.exists():
        return str(p)
    rows = normalize_benchmark_rows(read_json(p))
    df = pd.DataFrame(rows)
    save_table(df, out_dir, "table_vii_inference_performance", "Inference performance across deployment configurations")
    names = [str(x).replace(" (", "\n(") for x in df["Deployment Configuration"]]
    lat = df["Avg. Latency (ms/sample)"].astype(float).to_numpy()
    size = df["Size (MB)"].astype(float).to_numpy()
    x = np.arange(len(names)); w = 0.35
    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    b1 = ax1.bar(x - w/2, lat, w, color=BLUE, label="Latency (ms)")
    ax1.set_ylabel("Inference latency (ms/sample)", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.set_xticks(x); ax1.set_xticklabels(names)
    ax1.set_ylim(0, max(lat) * 1.30)
    ax2 = ax1.twinx()
    b2 = ax2.bar(x + w/2, size, w, color=GOLD, label="Size (MB)")
    ax2.set_ylabel("Model size (MB)", color=GOLD)
    ax2.tick_params(axis="y", labelcolor=GOLD)
    ax2.set_ylim(0, max(size) * 1.25)
    for b in b1:
        ax1.annotate(f"{b.get_height():.1f}", xy=(b.get_x()+b.get_width()/2, b.get_height()), xytext=(0,3), textcoords="offset points", ha="center", color=BLUE, fontsize=9)
    for b in b2:
        ax2.annotate(f"{b.get_height():.1f}", xy=(b.get_x()+b.get_width()/2, b.get_height()), xytext=(0,3), textcoords="offset points", ha="center", color=GOLD, fontsize=9)
    fig.legend(handles=[b1, b2], loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    ax1.set_title("Inference latency and model size across deployment configurations")
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(out_dir / "fig6_latency_vs_size.png")
    plt.close(fig)
    return None


def make_ablation_table(out_dir: Path, results_dir: Path) -> str | None:
    csv_p = results_dir / "ablation_results.csv"
    json_p = results_dir / "ablation_results.json"
    if csv_p.exists():
        df = pd.read_csv(csv_p)
    elif json_p.exists():
        df = pd.DataFrame(read_json(json_p))
    else:
        return str(csv_p)
    rename = {
        "variant_id": "Variant",
        "description": "Description",
        "tau": "tau",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "auc": "AUC",
        "ap": "AP",
        "fpr": "FPR",
    }
    df = df.rename(columns=rename)
    keep = [c for c in ["Variant", "Description", "tau", "Accuracy", "Precision", "Recall", "F1", "AUC", "AP", "FPR", "epochs", "batch_size", "max_len"] if c in df.columns]
    df = df[keep]
    save_table(df, out_dir, "table_ablation_study", "Ablation study of proposed components")
    return None


def make_roc_pr_assets(out_dir: Path, results_dir: Path) -> str | None:
    p = results_dir / "test_predictions.npz"
    if not p.exists():
        return str(p)
    d = np.load(p)
    labels = d["labels"]
    probs = d["probs"]
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)
    precision, recall, _ = precision_recall_curve(labels, probs)

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"CodeBERT + SQL-aware (AUC={auc:.4f})")
    ax.set_xscale("log")
    ax.set_xlim(max(1e-6, np.min(fpr[fpr > 0]) if np.any(fpr > 0) else 1e-6), 1.0)
    ax.set_ylim(0.90, 1.005)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve on the independent test set")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig7_roc_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    ax.plot(recall, precision, color=BLUE, lw=2, label=f"AP={ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve on the independent test set")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "fig8_precision_recall_curve.png")
    plt.close(fig)

    df = pd.DataFrame([{"ROC-AUC": auc, "Average Precision": ap}])
    save_table(df, out_dir, "table_auc_ap_summary", "Threshold-independent metrics")
    return None


def make_token_length_fig(out_dir: Path, skip_tokenize: bool = False, sample_size: int = 20000) -> dict[str, Any]:
    df = pd.read_csv(config.TRAIN_CSV).dropna(subset=["text"])
    df = df.sample(n=min(sample_size, len(df)), random_state=getattr(config, "SEED", 42))
    texts = df["text"].astype(str).tolist()
    used_tokens = False
    lengths: Iterable[int]
    if not skip_tokenize:
        try:
            from transformers import AutoTokenizer
            from tokenizer import ALL_SQL_TOKENS_DEDUPED
            tok = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
            tok.add_tokens(ALL_SQL_TOKENS_DEDUPED, special_tokens=False)
            lengths = [len(tok.tokenize(t)) for t in texts]
            used_tokens = True
        except Exception:
            lengths = [len(t) for t in texts]
    else:
        lengths = [len(t) for t in texts]
    lengths = np.array(list(lengths))
    p50, p95, p99 = np.percentile(lengths, [50, 95, 99])
    unit = "tokens" if used_tokens else "characters"
    df_stats = pd.DataFrame([{"p50": p50, "p95": p95, "p99": p99, "unit": unit, "sample_size": len(lengths)}])
    save_table(df_stats, out_dir, "table_token_length_statistics", "Token length statistics")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    upper = max(np.percentile(lengths, 99.5) * 1.2, p99 * 1.2)
    ax.hist(lengths, bins=80, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(p99, color=GOLD, ls="--", lw=2, label=f"p99 = {p99:.0f} {unit}")
    ax.set_xlim(0, upper)
    ax.set_xlabel(f"Sequence length ({unit})")
    ax.set_ylabel("Sample count")
    ax.set_title("Payload sequence-length distribution")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_token_length_distribution.png")
    plt.close(fig)
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99), "unit": unit}


def make_architecture_fig(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.2); ax.axis("off")
    def box(x, y, w, h, text, fc=BLUE, tc="white", fontsize=9):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="black", linewidth=1.0))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", color=tc, fontsize=fontsize, fontweight="bold")
    def arrow(x1, y1, x2, y2, color="black"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.4, color=color))
    box(0.2, 3.7, 1.7, 0.9, "HTTP Request\n(query/form/URL)", fc="#e0e0e0", tc="black")
    box(2.2, 3.7, 1.9, 0.9, "Input\nPreprocessing")
    box(4.5, 3.7, 2.1, 0.9, "SQL-Aware\nTokenization\n(+139 tokens)")
    box(7.0, 3.7, 1.9, 0.9, "CodeBERT\nEncoder\n(12 layers, d=768)")
    box(9.3, 3.7, 1.5, 0.9, "Threshold\nDecision (tau*)", fc=GOLD, tc="black")
    for x1, x2 in [(1.9,2.2),(4.1,4.5),(6.6,7.0),(8.9,9.3)]: arrow(x1,4.15,x2,4.15)
    ax.text(10.05, 3.45, "Benign / SQLi", ha="center", va="top", fontsize=10, fontstyle="italic")
    box(2.2, 2.2, 1.9, 0.9, "Multi-Source\nMaster Dataset\n(CSIC, SR-BH,\nFWAF, HttpParams)", fc=GREEN, fontsize=8)
    box(4.5, 2.2, 2.1, 0.9, "Label-Smoothed\nFine-tuning\n(eps=0.1, AdamW)", fc=GREEN, fontsize=9)
    box(7.0, 2.2, 1.9, 0.9, "Validation-Loss\nEarly Stop", fc=GREEN, fontsize=9)
    arrow(4.1,2.65,4.5,2.65,"gray"); arrow(6.6,2.65,7.0,2.65,"gray"); arrow(7.95,3.1,7.95,3.7,"gray")
    box(2.2, 0.7, 1.9, 0.9, "PyTorch\nCheckpoint", fc=ORANGE)
    box(4.5, 0.7, 2.1, 0.9, "ONNX Export\n+ O3 Graph Opt", fc=ORANGE)
    box(7.0, 0.7, 1.9, 0.9, "INT8 Dynamic\nQuantization", fc=ORANGE)
    box(9.3, 0.7, 1.5, 0.9, "Production\nWAF", fc=GOLD, tc="black")
    arrow(4.1,1.15,4.5,1.15,"gray"); arrow(6.6,1.15,7.0,1.15,"gray"); arrow(8.9,1.15,9.3,1.15,"gray"); arrow(7.95,2.2,7.95,1.6,"gray")
    ax.text(0.1, 4.85, "Online inference", fontsize=9, color=BLUE, fontweight="bold")
    ax.text(0.1, 3.35, "Training (offline)", fontsize=9, color="#5a6f60", fontweight="bold")
    ax.text(0.1, 1.85, "Deployment optimization (offline)", fontsize=9, color="#7a4530", fontweight="bold")
    ax.set_title("Proposed SQL Injection detection workflow", fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_architecture.png")
    plt.close(fig)



def make_csic2010_transfer_table(out_dir: Path) -> str | None:
    """Include MasterFull -> CSIC2010 fine-tuning summary if available."""
    transfer_summary = Path(config.PROJECT_ROOT) / "fine_tune2010" / "results" / "master_to_csic" / "summary_transfer.csv"
    if not transfer_summary.exists():
        return str(transfer_summary)
    try:
        df = pd.read_csv(transfer_summary)
        if df.empty:
            return str(transfer_summary)
        save_table(df, out_dir, "table_csic2010_transfer_finetune", "Master-pretrained CodeBERT fine-tuned on CSIC2010")
        return None
    except Exception:
        return str(transfer_summary)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=config.RESULTS_DIR)
    ap.add_argument("--out_dir", default=config.ASSETS_DIR)
    ap.add_argument("--skip_tokenize", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []

    make_architecture_fig(out_dir)
    make_dataset_table(out_dir)
    token_stats = make_token_length_fig(out_dir, skip_tokenize=args.skip_tokenize)
    threshold = load_threshold(results_dir)
    missing.extend(make_baseline_tables(out_dir, results_dir, threshold))
    m = make_training_history(out_dir, results_dir)
    if m: missing.append(m)
    missing.extend(make_threshold_assets(out_dir, results_dir, threshold))
    m = make_ablation_table(out_dir, results_dir)
    if m: missing.append(m)
    m = make_benchmark_assets(out_dir, results_dir)
    if m: missing.append(m)
    m = make_roc_pr_assets(out_dir, results_dir)
    if m: missing.append(m)
    m = make_csic2010_transfer_table(out_dir)
    if m: missing.append(m)

    summary = {
        "output_dir": str(out_dir),
        "results_dir": str(results_dir),
        "token_stats": token_stats,
        "missing_artifacts": sorted(set(missing)),
        "files": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
    }
    (out_dir / "paper_assets_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "missing_artifacts.json").write_text(json.dumps(summary["missing_artifacts"], indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
