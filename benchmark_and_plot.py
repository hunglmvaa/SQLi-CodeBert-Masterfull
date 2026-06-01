
"""
benchmark_and_plot.py
---------------------
CPU-oriented dynamic-shape benchmark for PyTorch FP32, ONNX FP32 O3,
ONNX FP16, and ONNX INT8. Use --sample_size all for paper-mode benchmark;
use --sample_size 500 for a quick Kaggle sanity check.
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config


def parse_sample_size(value):
    if value is None:
        return config.BENCH_SAMPLES
    s = str(value).strip().lower()
    if s in {"all", "none", "full", "-1", "0"}:
        return None
    n = int(s)
    if n <= 0:
        return None
    return n


def apply_thread_env(num_threads: int):
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    torch.set_num_threads(num_threads)


def get_dir_size(directory: str | Path, ext: str = ".onnx") -> float:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    total_bytes = 0
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.name.endswith(ext):
            total_bytes += file_path.stat().st_size
    return total_bytes / (1024 ** 2)


def build_ort_session_options(num_threads: int) -> ort.SessionOptions:
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = int(num_threads)
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_profiling = False
    return sess_options


def load_dynamic_inputs(tokenizer: AutoTokenizer, sample_size, max_length: int, data_path: Path) -> tuple[list[dict], dict]:
    print("2. Preparing dynamic-shape benchmark inputs...")
    if not data_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {data_path}")
    df = pd.read_csv(data_path).dropna(subset=["text"]).reset_index(drop=True)
    if "text" not in df.columns:
        raise ValueError(f"{data_path} must contain a 'text' column.")
    n_total = len(df)
    if sample_size is not None and sample_size < n_total:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    sample_texts = df["text"].astype(str).tolist()
    dynamic_inputs = []
    for text in sample_texts:
        encoded = tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=max_length)
        dynamic_inputs.append(encoded)
    token_lengths = np.array([int(sample["input_ids"].shape[1]) for sample in dynamic_inputs])
    meta = {
        "data_path": str(data_path),
        "raw_usable_rows": int(n_total),
        "sample_size": int(len(dynamic_inputs)),
        "sample_mode": "all" if len(dynamic_inputs) == n_total else "sampled",
        "token_len_min": int(token_lengths.min()),
        "token_len_mean": float(token_lengths.mean()),
        "token_len_p50": float(np.percentile(token_lengths, 50)),
        "token_len_p95": float(np.percentile(token_lengths, 95)),
        "token_len_p99": float(np.percentile(token_lengths, 99)),
        "token_len_max": int(token_lengths.max()),
    }
    print(f"[data] {meta['sample_size']:,}/{n_total:,} samples from {data_path} ({meta['sample_mode']})")
    print(f"[length] min={meta['token_len_min']}, mean={meta['token_len_mean']:.2f}, p50={meta['token_len_p50']:.1f}, p95={meta['token_len_p95']:.1f}, p99={meta['token_len_p99']:.1f}, max={meta['token_len_max']}")
    return dynamic_inputs, meta


def measure_latency_dynamic(model, dynamic_inputs: list[dict], is_ort: bool, warmup: int) -> float:
    warmup_inputs = dynamic_inputs[: min(warmup, len(dynamic_inputs))]
    for sample in warmup_inputs:
        if is_ort:
            model(**sample)
        else:
            with torch.no_grad():
                model(**sample)
    start = time.perf_counter()
    for sample in dynamic_inputs:
        if is_ort:
            model(**sample)
        else:
            with torch.no_grad():
                model(**sample)
    return ((time.perf_counter() - start) * 1000) / len(dynamic_inputs)


def run_benchmark(args):
    apply_thread_env(args.num_threads)
    sess_options = build_ort_session_options(args.num_threads)
    print("ONNX Runtime providers:", ort.get_available_providers())
    print("Configured CPU threads:", args.num_threads)
    print("Benchmark mode: CPU dynamic shape; padding=False")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_DIR)
    sample_size = parse_sample_size(args.sample_size)
    dynamic_inputs, input_meta = load_dynamic_inputs(tokenizer, sample_size, args.max_length, Path(args.data_path))

    print("3. Loading models...")
    model_pt = AutoModelForSequenceClassification.from_pretrained(config.MODEL_DIR).eval()
    model_fp32 = ORTModelForSequenceClassification.from_pretrained(config.ONNX_FP32_DIR, file_name="model_optimized.onnx", session_options=sess_options)
    model_fp16 = ORTModelForSequenceClassification.from_pretrained(config.ONNX_FP16_DIR, file_name="model_optimized.onnx", session_options=sess_options)
    model_int8 = ORTModelForSequenceClassification.from_pretrained(config.ONNX_INT8_DIR, file_name="model_quantized.onnx", session_options=sess_options)

    print("4. Benchmarking...")
    lat_pt = measure_latency_dynamic(model_pt, dynamic_inputs, False, args.warmup)
    time.sleep(args.cooldown)
    lat_fp32 = measure_latency_dynamic(model_fp32, dynamic_inputs, True, args.warmup)
    time.sleep(args.cooldown)
    lat_fp16 = measure_latency_dynamic(model_fp16, dynamic_inputs, True, args.warmup)
    time.sleep(args.cooldown)
    lat_int8 = measure_latency_dynamic(model_int8, dynamic_inputs, True, args.warmup)

    size_pt = sum(param.numel() for param in model_pt.parameters()) * 4 / (1024 ** 2)
    size_fp32 = get_dir_size(config.ONNX_FP32_DIR)
    size_fp16 = get_dir_size(config.ONNX_FP16_DIR)
    size_int8 = get_dir_size(config.ONNX_INT8_DIR)

    labels = ["PyTorch (Baseline)", "ONNX (FP32 O3)", "ONNX (FP16)", "ONNX (INT8)"]
    formats = ["32-bit (FP32)", "32-bit (FP32)", "16-bit (FP16)", "8-bit (INT8)"]
    latencies = [lat_pt, lat_fp32, lat_fp16, lat_int8]
    sizes = [size_pt, size_fp32, size_fp16, size_int8]
    rows = []
    for label, data_format, latency, size in zip(labels, formats, latencies, sizes):
        rows.append({
            "Deployment Configuration": label,
            "Data Format": data_format,
            "Avg. Latency (ms)": round(float(latency), 4),
            "Size (MB)": round(float(size), 4),
            "Speedup": round(float(lat_pt / latency), 4),
        })
    return rows, input_meta


def save_benchmark_artifacts(rows, input_meta, args):
    results_dir = Path(config.RESULTS_DIR)
    paper_assets_dir = Path(config.ASSETS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    paper_assets_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_type": "dynamic_shape_cpu",
        "runtime": "CPU-oriented ONNX Runtime benchmark",
        "unit": "ms/sample",
        "input_meta": input_meta,
        "tokenization": {"padding": False, "truncation": True, "max_length": args.max_length},
        "onnxruntime_available_providers": ort.get_available_providers(),
        "num_threads": int(args.num_threads),
        "warmup": int(args.warmup),
        "rows": rows,
    }
    (results_dir / "benchmark_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    table_df = pd.DataFrame(rows)
    table_df.to_csv(paper_assets_dir / "table_vii_inference_performance.csv", index=False)
    table_df.to_csv(paper_assets_dir / "table_iii_inference_perf.csv", index=False)  # legacy compatibility
    with open(paper_assets_dir / "table_vii_inference_performance.tex", "w", encoding="utf-8") as f:
        f.write("% Inference performance across optimization strategies\n")
        f.write(table_df.to_latex(index=False, float_format="%.4f"))

    # one chart only, no color specialization beyond matplotlib defaults for reproducibility
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(rows))
    ax1.bar(x - 0.2, [r["Avg. Latency (ms)"] for r in rows], width=0.4, label="Avg latency (ms)")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, [r["Size (MB)"] for r in rows], width=0.4, label="Size (MB)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([r["Deployment Configuration"] for r in rows], rotation=15, ha="right")
    ax1.set_ylabel("Latency (ms/sample)")
    ax2.set_ylabel("Model size (MB)")
    ax1.set_title("Dynamic-shape CPU inference benchmark")
    fig.tight_layout()
    fig.savefig(paper_assets_dir / "fig_inference_performance.png", dpi=300)
    plt.close(fig)
    print("\n--- BENCHMARK RESULTS ---")
    print(table_df.to_string(index=False))
    print(f"Saved: {results_dir/'benchmark_results.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_size", default=str(config.BENCH_SAMPLES), help="Number of samples or 'all'. Use 500 for quick check, all for paper-mode.")
    ap.add_argument("--num_threads", type=int, default=int(config.NUM_THREADS))
    ap.add_argument("--warmup", type=int, default=int(config.BENCH_WARMUP))
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--max_length", type=int, default=int(config.TRUNCATE_AT))
    ap.add_argument("--data_path", default=config.TEST_CSV)
    args = ap.parse_args()
    rows, input_meta = run_benchmark(args)
    save_benchmark_artifacts(rows, input_meta, args)


if __name__ == "__main__":
    main()
