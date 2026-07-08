param(
    [string]$Seeds = "7,42,99,123,2024",
    [int]$Epochs = 5,
    [int]$BatchSize = 32,
    [int]$MaxLen = 256,
    [int]$NumWorkers = 0,
    [int]$RepresentativeSeed = 42,
    [switch]$RunAblation,
    [switch]$RunTransformerBaselines,
    [switch]$RunClassicalBaselines,
    [switch]$RunCharBaselines,
    [switch]$RunTsne,
    [switch]$RunRobustness,
    [switch]$BuildPaperAssets
)

$ErrorActionPreference = "Stop"

Write-Host "=== SQLi CodeBERT full rerun ==="
Write-Host "Seeds=$Seeds Epochs=$Epochs BatchSize=$BatchSize MaxLen=$MaxLen NumWorkers=$NumWorkers"

python validate_inputs.py --data_dir data --suffix "" --strict

python run_multiseed.py `
  --seeds $Seeds `
  --epochs $Epochs `
  --batch_size $BatchSize `
  --max_len $MaxLen `
  --num_workers $NumWorkers `
  --skip_existing

python postprocess_multiseed.py `
  --summary results/multiseed/summary.json `
  --results_dir results `
  --out_dir results/multiseed `
  --assets_dir paper_assets

python select_representative_seed.py `
  --summary results/multiseed/summary.json `
  --results_dir results `
  --seed $RepresentativeSeed

if ($RunAblation) {
  python run_ablation_multigpu_resumable.py `
    --epochs 3 `
    --batch_size $BatchSize `
    --max_len $MaxLen `
    --num_workers $NumWorkers `
    --seed $RepresentativeSeed `
    --resume
}

if ($RunTransformerBaselines) {
  python run_transformer_baselines.py `
    --train data/train_dataset.csv `
    --val data/val_dataset.csv `
    --test data/test_dataset.csv `
    --output_dir results/baselines_master `
    --models bert,distilbert,roberta,codebert `
    --epochs 3 `
    --batch_size $BatchSize `
    --eval_batch_size $BatchSize `
    --max_len $MaxLen `
    --num_workers $NumWorkers `
    --seed $RepresentativeSeed
}

if ($RunClassicalBaselines) {
  python run_classical_ml_baselines.py `
    --train data/train_dataset.csv `
    --val data/val_dataset.csv `
    --test data/test_dataset.csv `
    --output_dir results/baselines_master `
    --models all `
    --max_features 50000 `
    --seed $RepresentativeSeed
}

if ($RunCharBaselines) {
  python run_baseline_exact_split_v2.py `
    --train data/train_dataset.csv `
    --val data/val_dataset.csv `
    --test data/test_dataset.csv `
    --baseline all `
    --output_dir results/baselines_master `
    --device cuda `
    --epochs 3 `
    --batch_size $BatchSize `
    --seed $RepresentativeSeed
}

if ($RunTsne) {
  python make_tsne_embeddings.py `
    --model_dir results/best_model_sql_aware `
    --data data/test_dataset.csv `
    --out_dir paper_assets `
    --batch_size $BatchSize `
    --max_len $MaxLen `
    --seed $RepresentativeSeed
}

if ($RunRobustness) {
  python evaluate_robustness_obfuscation.py `
    --model_dir results/best_model_sql_aware `
    --threshold_json results/threshold.json `
    --data data/test_dataset.csv `
    --out_dir paper_assets `
    --batch_size $BatchSize `
    --max_len $MaxLen `
    --seed $RepresentativeSeed
}

if ($BuildPaperAssets) {
  python paper_assets_builder.py --results_dir results --out_dir paper_assets
}

Write-Host "=== Done ==="
Write-Host "Main outputs:"
Write-Host "  results/multiseed/summary.json"
Write-Host "  results/multiseed/summary_extended.csv"
Write-Host "  paper_assets/table_multiseed_extended_metrics.csv"
Write-Host "  paper_assets/fig_multiseed_error_bars.png"
