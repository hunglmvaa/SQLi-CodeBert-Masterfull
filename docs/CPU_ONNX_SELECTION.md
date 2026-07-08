# CPU/ONNX Selection Evidence

This full repository includes the Python source files used for ONNX export and CPU benchmarking:

- `optimize_model.py`
- `benchmark_and_plot.py`

The repository does not include trained checkpoints or ONNX model binaries because these are generated outputs and are not promised by the manuscript. Therefore, reviewers can verify the reported CPU/ONNX selection directly from the released artifacts, and can rerun ONNX export/benchmarking after reproducing or providing the corresponding trained checkpoint.

Recorded evidence files:

- `artifacts/evidence_for_reviewers/onnx_optimize_metadata.json`
- `artifacts/evidence_for_reviewers/benchmark_environment.json`
- `artifacts/evidence_for_reviewers/benchmark_history.jsonl`
- `artifacts/evidence_for_reviewers/autotune_threads_20260703_211756.csv`
- `artifacts/evidence_for_reviewers/benchmark_results.json`
- `artifacts/evidence_for_reviewers/benchmark_latency_raw.csv`

Use the helper below to reconstruct the reported CPU/ONNX choice from the recorded artifacts:

```bash
python artifact_tools/select_cpu_onnx_result.py
```

The decision represented by the artifacts is:

- The preliminary 200-sample sweep compared 1, 2, 4, 6, 8, 12, 16, and 20 CPU threads.
- 12 and 16 threads were effectively tied for ONNX INT8 latency, with 16 threads slightly faster in the sweep.
- The full 112,753-sample benchmark therefore used `OMP_NUM_THREADS=16`.
- The full-test benchmark used ONNX Runtime 1.24.4 with CPU execution provider available and reports model-inference latency only.
- The lowest-latency deployment row is ONNX INT8: 7.97 ms average latency, p95 = 15.39 ms, model size = 120.2 MB, speedup = 4.63x over PyTorch FP32.
