# SQLi-CodeBert-Masterfull: gói tái lập thực nghiệm multi-seed

Kho này là gói mã nguồn **multi-seed** cho bài báo:

**SQL Injection Detection with SQL-Aware CodeBERT and ONNX INT8 Deployment**

Kho này cung cấp các script, bộ dữ liệu đã chia, và tài liệu cần thiết để tái lập thực nghiệm multi-seed và tạo bảng/hình cho bài báo.

## Nội dung 

- Huấn luyện mô hình đề xuất CodeBERT + SQL-aware tokenizer theo nhiều seed.
- Hiệu chỉnh ngưỡng trên validation cho từng seed.
- Tổng hợp mean và standard deviation của các metric chính.
- Script baseline, ablation, ONNX benchmark và tạo bảng/hình phục vụ bài báo.
- Bộ split MasterFull đã xử lý: `train_dataset.csv`, `val_dataset.csv`,
  `test_dataset.csv`.

## Cấu trúc chính

```text
run_multiseed.py          # entry point chính
run_multiseed.sh          # shell wrapper
train_codebert.py         # worker nội bộ, được run_multiseed.py gọi theo từng seed
validate_inputs.py        # kiểm tra dữ liệu
run_ablation_multigpu_resumable.py
run_classical_ml_baselines.py
run_baseline_exact_split_v2.py
run_transformer_baselines.py
optimize_model.py
benchmark_and_plot.py
paper_assets_builder.py
data/
docs/MULTI_SEED_GUIDE.md
```

## Cài môi trường

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

## Kiểm tra dữ liệu

```bash
python validate_inputs.py --data_dir data --suffix "" --strict
```

Số mẫu kỳ vọng:

| Split | Samples | Normal | SQLi |
|---|---:|---:|---:|
| Training | 149,344 | 89,607 | 59,737 |
| Validation | 37,337 | 22,402 | 14,935 |
| Test | 112,753 | 68,640 | 44,113 |

## Chạy multi-seed

```bash
python run_multiseed.py \
  --seeds 42,123,2024 \
  --epochs 5 \
  --batch_size 32 \
  --max_len 256 \
  --num_workers 2 \
  --skip_existing
```

Hoặc:

```bash
bash run_multiseed.sh
```

Output chính:

```text
results/seed_42/
results/seed_123/
results/seed_2024/
results/multiseed/summary.csv
results/multiseed/summary.json
results/multiseed/summary_table.tex
```

Lưu ý: `train_codebert.py` vẫn được giữ vì `run_multiseed.py` cần gọi nó cho
từng seed.  Khi báo cáo bài báo, không dùng `train_codebert.py` như một kết quả
single-seed mới; entry point chính là `run_multiseed.py`.

## Các script phụ phục vụ bài báo

- `run_ablation_multigpu_resumable.py`: chạy ablation.
- `run_classical_ml_baselines.py`: baseline TF-IDF/classical ML.
- `run_baseline_exact_split_v2.py`: Character BiLSTM và 1D-CNN.
- `run_transformer_baselines.py`: BERT, DistilBERT, RoBERTa, CodeBERT baseline.
- `optimize_model.py`: export ONNX FP32/FP16/INT8.
- `benchmark_and_plot.py`: benchmark CPU.
- `paper_assets_builder.py`: tạo bảng/hình từ artifact thực nghiệm.

