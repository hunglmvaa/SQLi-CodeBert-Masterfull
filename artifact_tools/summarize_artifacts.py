#!/usr/bin/env python3
"""Print the key values from the revision artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def pct(value: float) -> str:
    return f"{100 * value:.3f}%"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def print_baseline_table() -> None:
    path = ARTIFACTS / "baselines_master" / "transformer_summary.csv"
    print("\nTransformer baselines")
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        print(
            f"- {row['baseline']}: F1={float(row['f1']):.4f}, "
            f"AUC={float(row['roc_auc']):.4f}, FPR={pct(float(row['fpr']))}"
        )


def main() -> None:
    master = load_json(ARTIFACTS / "MASTER_NUMBERS.json")

    dataset = master["dataset"]
    print("Dataset")
    print(f"- total={dataset['total']:,}")
    for name, split in dataset["splits"].items():
        print(
            f"- {name}: n={split['n']:,}, "
            f"normal={split['normal']:,}, sqli={split['sqli']:,}"
        )

    proposed = master["proposed_released_seed_42"]
    print("\nReleased proposed run, seed 42")
    print(
        f"- F1={proposed['f1']:.4f}, AUC={proposed['auc']:.4f}, "
        f"AP={proposed['ap']:.4f}, TPR={pct(proposed['recall_tpr'])}, "
        f"FPR={pct(proposed['fpr'])}"
    )
    print(
        f"- Confusion: TP={proposed['tp']}, TN={proposed['tn']}, "
        f"FP={proposed['fp']}, FN={proposed['fn']}"
    )

    agg = master["proposed_aggregate"]["f1"]
    vanilla = master["vanilla_aggregate"]["f1"]
    print("\nFive-seed F1")
    print(
        f"- Proposed: mean={agg['mean']:.4f}, std={agg['std']:.4f}, "
        f"CI95=[{agg['ci95_low']:.4f}, {agg['ci95_high']:.4f}]"
    )
    print(
        f"- Vanilla CodeBERT: mean={vanilla['mean']:.4f}, "
        f"std={vanilla['std']:.4f}, "
        f"CI95=[{vanilla['ci95_low']:.4f}, {vanilla['ci95_high']:.4f}]"
    )

    print("\nMcNemar")
    for row in master["mcnemar"]["per_seed"]:
        proposed_gt_vanilla = row.get("proposed_only_correct", row.get("P_gt_V"))
        vanilla_gt_proposed = row.get("vanilla_only_correct", row.get("V_gt_P"))
        significant = row.get("significant_0_05", row.get("significant_0.05"))
        sig = "yes" if significant else "no"
        print(
            f"- seed {row['seed']}: P>V={proposed_gt_vanilla}, "
            f"V>P={vanilla_gt_proposed}, p={row['p_exact']:.3g}, "
            f"favors={row['favors']}, sig={sig}"
        )

    print("\nLatency")
    for row in master["latency"]["rows"]:
        print(
            f"- {row['config']}: avg={row['avg_ms']:.2f} ms, "
            f"p95={row['p95_ms']:.2f} ms, size={row['size_mb']:.1f} MB, "
            f"speedup={row['speedup']:.2f}x"
        )

    print_baseline_table()

    warnings = master.get("_warnings", [])
    if warnings:
        print("\nWarnings")
        for item in warnings:
            print(f"- {item}")


if __name__ == "__main__":
    main()
