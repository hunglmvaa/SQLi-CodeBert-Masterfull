#!/usr/bin/env python3
"""
select_representative_seed.py
-----------------------------
Copy one seed's artifacts to results/ so paper_assets_builder.py and ONNX export
can work with the multi-seed layout.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def choose_seed(summary_path: Path, mode: str, explicit_seed: int | None) -> int:
    d = json.loads(summary_path.read_text(encoding="utf-8"))
    per_seed = {int(k): v for k, v in d.get("per_seed", {}).items()}
    if explicit_seed is not None:
        if explicit_seed not in per_seed:
            raise ValueError(f"Seed {explicit_seed} is not in {summary_path}")
        return explicit_seed
    if not per_seed:
        raise ValueError(f"No per-seed metrics found in {summary_path}")
    if mode == "best_f1":
        return max(per_seed, key=lambda s: per_seed[s].get("f1", float("-inf")))
    ranked = sorted(per_seed, key=lambda s: per_seed[s].get("f1", float("nan")))
    return ranked[len(ranked) // 2]


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/multiseed/summary.json")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["median_f1", "best_f1"], default="median_f1")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    seed = choose_seed(Path(args.summary), args.mode, args.seed)
    seed_dir = results_dir / f"seed_{seed}"
    model_src = seed_dir / "best_model_sql_aware"
    model_dst = results_dir / "best_model_sql_aware"

    if model_dst.exists():
        shutil.rmtree(model_dst)
    shutil.copytree(model_src, model_dst)

    for name in ["history.json", "threshold.json", "val_predictions.npz", "test_predictions.npz", "run_config.json"]:
        copy_file(seed_dir / name, results_dir / name)

    meta = {
        "representative_seed": seed,
        "source_dir": str(seed_dir),
        "model_dir": str(model_dst),
        "selection_note": "Representative artifacts copied for figures, ONNX export, and paper assets. Main reported metrics should use results/multiseed/summary.*",
    }
    (results_dir / "representative_seed.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
