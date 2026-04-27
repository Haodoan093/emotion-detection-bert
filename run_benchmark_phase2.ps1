$ErrorActionPreference = "Stop"

Write-Host "=========================================================="
Write-Host "PHASE 2 BENCHMARK: 3 EPOCHS + FOCAL LOSS"
Write-Host "Models: DistilBERT vs mBERT"
Write-Host "=========================================================="

Write-Host "`n[1/4] Starting Job: DistilBERT on Kaggle-8k (English)"
python scripts/train_emotion_model.py `
    --model distilbert-base-multilingual-cased `
    --train-file data/emotion/splits/kaggle-8k/train.jsonl `
    --val-file data/emotion/splits/kaggle-8k/val.jsonl `
    --epochs 3 `
    --batch-size 8 `
    --output-dir models/emotion/bench/distil_kaggle8k_e3 `
    --best-dir models/emotion/bench/distil_kaggle8k_e3/best `
    --summary-file reports/emotion/benchmark_20260423/train_distil_kaggle8k_e3.json

Write-Host "`n[2/4] Starting Job: DistilBERT on UIT-VSMEC (Vietnamese) + Focal Loss"
python scripts/train_emotion_model.py `
    --model distilbert-base-multilingual-cased `
    --train-file data/emotion/splits/uit-vsmec/train.jsonl `
    --val-file data/emotion/splits/uit-vsmec/val.jsonl `
    --epochs 3 `
    --batch-size 8 `
    --use-focal-loss `
    --output-dir models/emotion/bench/distil_uit_e3_focal `
    --best-dir models/emotion/bench/distil_uit_e3_focal/best `
    --summary-file reports/emotion/benchmark_20260423/train_distil_uit_e3_focal.json

Write-Host "`n[3/4] Starting Job: mBERT on Kaggle-8k (English)"
python scripts/train_emotion_model.py `
    --model bert-base-multilingual-cased `
    --train-file data/emotion/splits/kaggle-8k/train.jsonl `
    --val-file data/emotion/splits/kaggle-8k/val.jsonl `
    --epochs 3 `
    --batch-size 8 `
    --output-dir models/emotion/bench/mbert_kaggle8k_e3 `
    --best-dir models/emotion/bench/mbert_kaggle8k_e3/best `
    --summary-file reports/emotion/benchmark_20260423/train_mbert_kaggle8k_e3.json

Write-Host "`n[4/4] Starting Job: mBERT on UIT-VSMEC (Vietnamese) + Focal Loss"
python scripts/train_emotion_model.py `
    --model bert-base-multilingual-cased `
    --train-file data/emotion/splits/uit-vsmec/train.jsonl `
    --val-file data/emotion/splits/uit-vsmec/val.jsonl `
    --epochs 3 `
    --batch-size 8 `
    --use-focal-loss `
    --output-dir models/emotion/bench/mbert_uit_e3_focal `
    --best-dir models/emotion/bench/mbert_uit_e3_focal/best `
    --summary-file reports/emotion/benchmark_20260423/train_mbert_uit_e3_focal.json

Write-Host "`n=========================================================="
Write-Host "PHASE 2 TRAINING COMPLETED SUCCESSFULLY!"
Write-Host "=========================================================="
