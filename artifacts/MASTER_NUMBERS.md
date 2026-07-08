# Master numbers for ETASR #20371

This file consolidates the reviewer-facing numerical evidence included in `artifacts/`. It is intended for audit and consistency checking, not as a source of additional claims beyond the manuscript.

## Dataset

- Total: 299,434 samples (Normal 180,649 / SQLi 118,785).
- train: n=149,344, Normal=89,607, SQLi=59,737.
- val: n=37,337, Normal=22,402, SQLi=14,935.
- test: n=112,753, Normal=68,640, SQLi=44,113.

## Released proposed run, seed 42

- F1=0.9923, AUC=0.9992, AP=0.9990, TPR=98.82%, FPR=0.229%.
- Balanced Accuracy=0.9930, MCC=0.9874, Accuracy=0.9940.
- Confusion matrix: TP=43594, TN=68483, FP=157, FN=519.

## Identical-recipe default-tokenizer CodeBERT, seed 42

- F1=0.9920, AUC=0.9989, AP=0.9989, TPR=98.73%, FPR=0.210%.
- Confusion matrix: TP=43553, TN=68496, FP=144, FN=560.

## Five-seed aggregate

### Proposed

- F1: mean=0.9922, std=0.0008, CI95=[0.9913, 0.9931].
- AUC: mean=0.9991, std=0.0002, CI95=[0.9988, 0.9993].
- AP: mean=0.9989, std=0.0001, CI95=[0.9987, 0.9991].
- FPR: mean=0.21%, std=0.02%, CI95=[0.18%, 0.24%].
- BALANCED_ACCURACY: mean=0.9928, std=0.0008, CI95=[0.9918, 0.9938].
- MCC: mean=0.9873, std=0.0012, CI95=[0.9857, 0.9888].
### Default-tokenizer CodeBERT

- F1: mean=0.9920, std=0.0012, CI95=[0.9905, 0.9935].
- AUC: mean=0.9990, std=0.0004, CI95=[0.9985, 0.9995].
- AP: mean=0.9989, std=0.0004, CI95=[0.9985, 0.9993].
- FPR: mean=0.22%, std=0.06%, CI95=[0.14%, 0.30%].
- BALANCED_ACCURACY: mean=0.9927, std=0.0013, CI95=[0.9911, 0.9942].
- MCC: mean=0.9870, std=0.0020, CI95=[0.9845, 0.9894].

## Threshold plateau

- F1 by threshold for seed 42: tau=0.25: 0.9923, tau=0.30: 0.9921, tau=0.35: 0.9921, tau=0.40: 0.9921, tau=0.45: 0.9921, tau=0.50: 0.9920, tau=0.55: 0.9921, tau=0.57: 0.9921.
- Max-min F1 over the shown threshold range: 0.000267.

## McNemar paired tests

| seed | discordant | proposed-only correct | baseline-only correct | p exact | favors | significant at 0.05 |
|---:|---:|---:|---:|---:|---|---|
| 7 | 499 | 273 | 226 | 0.0394 | proposed | yes |
| 42 | 344 | 186 | 158 | 0.145 | proposed | no |
| 99 | 476 | 155 | 321 | 2.16e-14 | vanilla | yes |
| 123 | 546 | 351 | 195 | 2.42e-11 | proposed | yes |
| 2024 | 388 | 202 | 186 | 0.446 | proposed | no |

The paired tests are mixed across seeds and point in opposing directions; no systematic in-distribution accuracy advantage over the same-backbone baseline is claimed.

## Ablation

| variant | tau | F1 | AUC | FPR |
|---|---:|---:|---:|---:|
| v0_baseline | 0.50 | 0.98771 | 0.99853 | 0.00272 |
| v1_sql_tokens | 0.50 | 0.98680 | 0.99809 | 0.00326 |
| v2_label_smooth | 0.50 | 0.99157 | 0.99894 | 0.00198 |
| v3_full | 0.32 | 0.99174 | 0.99894 | 0.00216 |

Under the controlled ablation protocol, label smoothing accounts for most of the F1 gain; threshold calibration selects the final recall-oriented operating point. SQL-aware tokenization alone does not improve F1 at tau=0.5.

## Exploratory obfuscation stress test

| model | F1 clean | F1 obfuscated | delta F1 | recall clean | recall obfuscated | delta recall |
|---|---:|---:|---:|---:|---:|---:|
| CodeBERT vanilla | 0.9928 | 0.9599 | 0.0328 | 0.9874 | 0.9246 | 0.0628 |
| Proposed (SQL-aware) | 0.9936 | 0.9572 | 0.0363 | 0.9894 | 0.9200 | 0.0694 |

These files are retained as exploratory reviewer evidence. They are not used to claim a demonstrated adversarial-obfuscation advantage; the manuscript lists such evaluation as future work.

## Latency

| config | format | avg ms | p95 ms | size MB | speedup |
|---|---|---:|---:|---:|---:|
| PyTorch (Baseline) | 32-bit (FP32) | 36.91 | 58.13 | 475.8 | 1.00x |
| ONNX (FP32 O3) | 32-bit (FP32) | 10.63 | 19.30 | 475.9 | 3.47x |
| ONNX (FP16) | 16-bit (FP16) | 11.04 | 20.17 | 237.9 | 3.34x |
| ONNX (INT8) | 8-bit (INT8) | 7.97 | 15.39 | 120.2 | 4.63x |

## Token length audit

- Test samples: 112,753.
- Mean reduction: -0.141%; p95 reduction: 0.000%; samples shorter under SQL-aware tokenization: 0.250%.
- SQL-aware tokenization does not materially shorten sequences in this audit; latency gains are attributed to ONNX graph optimization and INT8 quantization rather than shorter tokenized inputs.

## Scope notes

- Full leave-one-source-out generalization is not claimed in the manuscript; LOSO retraining is left for future work.
- The robustness files are exploratory stress-test evidence and do not establish adversarial-obfuscation robustness as a demonstrated advantage.
- The artifact prediction CSVs intentionally exclude payload text, sample IDs, source IDs, and payload hashes.
