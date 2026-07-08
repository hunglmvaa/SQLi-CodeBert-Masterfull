# Artifact Inventory

## Consolidated Numbers

- `artifacts/MASTER_NUMBERS.json`: machine-readable source of the final numbers used in the response/manuscript.
- `artifacts/MASTER_NUMBERS.md`: human-readable audit notes and table-ready values.

## Reviewer Evidence

Directory: `artifacts/evidence_for_reviewers/`

- `ablation_results.csv`: v0-v3 ablation metrics.
- `autotune_threads_20260703_211756.csv`: preliminary ONNX INT8 thread sweep on a 200-sample subset.
- `benchmark_environment.json`: full-test CPU benchmark environment.
- `benchmark_history.jsonl`: thread-sweep benchmark history.
- `benchmark_latency_raw.csv`: raw per-sample benchmark latency values for the full benchmark.
- `benchmark_results.json`: summarized benchmark results.
- `data_validation_report.csv`: row counts and split checks.
- `mcnemar_results.json`: paired McNemar results by seed.
- `onnx_optimize_metadata.json`: ONNX export/optimization metadata.
- `predictions_seed_*.csv`: paired per-sample predictions for proposed vs. identical-recipe vanilla CodeBERT.
- `robustness_compare_full.csv`: clean vs. obfuscated comparison.
- `robustness_delta_summary.csv`: robustness deltas.
- `token_length_summary.json`: tokenizer length summary.

## Helper Scripts

- `artifact_tools/summarize_artifacts.py`: prints the key dataset, accuracy, McNemar,
  latency, and baseline values.
- `artifact_tools/select_cpu_onnx_result.py`: reconstructs the CPU/ONNX selection from
  the recorded thread sweep and full benchmark artifacts.
- `artifact_tools/verify_checksums.py`: verifies all files against `MANIFEST_SHA256.json`.

## Baseline Results

Directory: `artifacts/baselines_master/`

- `summary.csv`: non-transformer selected baseline summary.
- `classical_summary.csv`: TF-IDF classical model metrics.
- `transformer_summary.csv`: BERT, DistilBERT, RoBERTa, and CodeBERT baseline metrics.
- `*_metrics.json`: per-model metric files.
- `all_results.json`: combined JSON for the subset of baselines available in that source file.
