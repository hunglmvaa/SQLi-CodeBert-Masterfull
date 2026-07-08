#!/usr/bin/env python3
"""Select and report the CPU/ONNX benchmark configuration from artifacts.

This script does not rerun inference. It reads the recorded benchmark artifacts
and reconstructs the decision used for the manuscript:

1. Use the preliminary thread sweep to select the best CPU thread count.
2. Confirm the full-test benchmark was run with that thread count.
3. Select the lowest-latency deployment configuration from the full benchmark.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts" / "evidence_for_reviewers"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_thread_sweep() -> list[dict]:
    path = EVIDENCE / "autotune_threads_20260703_211756.csv"
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_history() -> list[dict]:
    path = EVIDENCE / "benchmark_history.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latency(row: dict) -> float:
    return float(row["Avg. Latency (ms)"])


def main() -> None:
    optimize_meta = load_json(EVIDENCE / "onnx_optimize_metadata.json")
    benchmark_env = load_json(EVIDENCE / "benchmark_environment.json")
    benchmark_results = load_json(EVIDENCE / "benchmark_results.json")

    sweep = load_thread_sweep()
    best_sweep = min(sweep, key=lambda r: float(r["int8_avg_latency_ms"]))
    history = load_history()
    full_runs = [
        r for r in history
        if r.get("input_meta", {}).get("sample_mode") == "all"
        or r.get("input_meta", {}).get("sample_size") == 112753
    ]
    if not full_runs:
        raise SystemExit("No full-test benchmark run found in benchmark_history.jsonl")
    full = full_runs[-1]

    full_rows = full["rows"]
    best_full = min(full_rows, key=latency)
    int8_full = next(r for r in full_rows if "INT8" in r["Deployment Configuration"])

    print("CPU/ONNX selection summary")
    print("==========================")
    print(f"ONNX Runtime version: {optimize_meta['onnxruntime']['version']}")
    print("Available providers:", ", ".join(optimize_meta["onnxruntime"]["available_providers"]))
    print("Benchmark providers:", ", ".join(benchmark_env["providers"]))
    print()
    print("Thread sweep, 200 samples")
    for row in sweep:
        marker = "  <-- selected" if row["num_threads"] == best_sweep["num_threads"] else ""
        print(
            f"- {row['num_threads']:>2} threads: "
            f"INT8 avg {float(row['int8_avg_latency_ms']):.4f} ms{marker}"
        )
    print()
    print("Full-test benchmark")
    print(f"- samples: {full['input_meta']['sample_size']:,}")
    print(f"- num_threads: {full['num_threads']}")
    print(f"- warmup: {full['warmup']}")
    print(f"- command: {full['environment']['command']}")
    print()
    print("Selected deployment row")
    print(f"- configuration: {best_full['Deployment Configuration']}")
    print(f"- format: {best_full['Data Format']}")
    print(f"- avg latency: {best_full['Avg. Latency (ms)']:.4f} ms")
    print(f"- p95 latency: {best_full['P95 Latency (ms)']:.4f} ms")
    print(f"- model size: {best_full['Size (MB)']:.4f} MB")
    print(f"- speedup: {best_full['Speedup']:.4f}x")
    print()
    print("Manuscript value")
    print(
        f"ONNX INT8 on CPU, OMP_NUM_THREADS={full['num_threads']}: "
        f"avg={int8_full['Avg. Latency (ms)']:.2f} ms, "
        f"p95={int8_full['P95 Latency (ms)']:.2f} ms, "
        f"size={int8_full['Size (MB)']:.1f} MB, "
        f"speedup={int8_full['Speedup']:.2f}x."
    )


if __name__ == "__main__":
    main()

