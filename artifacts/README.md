# Revision evidence artifacts

This directory contains the numerical evidence package used for the ETASR #20371 revision. In this full repository, the artifacts complement the runnable source code, released split CSV files, token list, and evaluation scripts in the repository root.

Included evidence:

- `MASTER_NUMBERS.json` / `.md`: consolidated values used to audit manuscript tables and reviewer responses.
- `baselines_master/`: baseline metric summaries.
- `evidence_for_reviewers/`: ablation outputs, paired prediction CSV files for McNemar tests, CPU/ONNX benchmark logs, token-length summary, and exploratory robustness stress-test files.

Important scope notes:

- The prediction CSV files intentionally omit payload text, sample IDs, source IDs, and payload hashes.
- The exploratory obfuscation files do not establish a demonstrated robustness advantage; the manuscript treats dedicated adversarial-obfuscation evaluation as future work.
- Trained checkpoints and ONNX binaries are not included because the manuscript does not promise model weights and these files are generated outputs.
