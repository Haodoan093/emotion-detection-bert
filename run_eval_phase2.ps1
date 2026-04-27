$ErrorActionPreference = "Stop"

Write-Host "Running Evaluations for Phase 2 Models..."

# 1. DistilBERT Kaggle E3
python scripts/evaluate_emotion_model.py --model models/emotion/bench/distil_kaggle8k_e3/best --test-file data/emotion/splits/kaggle-8k/test.jsonl --output reports/emotion/benchmark_20260423/eval_distil_kaggle8k_e3_on_kaggle8k.json --error-md reports/emotion/benchmark_20260423/error_distil_kaggle8k_e3_on_kaggle8k.md
python scripts/evaluate_emotion_model.py --model models/emotion/bench/distil_kaggle8k_e3/best --test-file data/emotion/splits/uit-vsmec/test.jsonl --output reports/emotion/benchmark_20260423/eval_distil_kaggle8k_e3_on_uit.json --error-md reports/emotion/benchmark_20260423/error_distil_kaggle8k_e3_on_uit.md

# 2. DistilBERT UIT E3 Focal
python scripts/evaluate_emotion_model.py --model models/emotion/bench/distil_uit_e3_focal/best --test-file data/emotion/splits/kaggle-8k/test.jsonl --output reports/emotion/benchmark_20260423/eval_distil_uit_e3_focal_on_kaggle8k.json --error-md reports/emotion/benchmark_20260423/error_distil_uit_e3_focal_on_kaggle8k.md
python scripts/evaluate_emotion_model.py --model models/emotion/bench/distil_uit_e3_focal/best --test-file data/emotion/splits/uit-vsmec/test.jsonl --output reports/emotion/benchmark_20260423/eval_distil_uit_e3_focal_on_uit.json --error-md reports/emotion/benchmark_20260423/error_distil_uit_e3_focal_on_uit.md

# 3. mBERT Kaggle E3
python scripts/evaluate_emotion_model.py --model models/emotion/bench/mbert_kaggle8k_e3/best --test-file data/emotion/splits/kaggle-8k/test.jsonl --output reports/emotion/benchmark_20260423/eval_mbert_kaggle8k_e3_on_kaggle8k.json --error-md reports/emotion/benchmark_20260423/error_mbert_kaggle8k_e3_on_kaggle8k.md
python scripts/evaluate_emotion_model.py --model models/emotion/bench/mbert_kaggle8k_e3/best --test-file data/emotion/splits/uit-vsmec/test.jsonl --output reports/emotion/benchmark_20260423/eval_mbert_kaggle8k_e3_on_uit.json --error-md reports/emotion/benchmark_20260423/error_mbert_kaggle8k_e3_on_uit.md

# 4. mBERT UIT E3 Focal
python scripts/evaluate_emotion_model.py --model models/emotion/bench/mbert_uit_e3_focal/best --test-file data/emotion/splits/kaggle-8k/test.jsonl --output reports/emotion/benchmark_20260423/eval_mbert_uit_e3_focal_on_kaggle8k.json --error-md reports/emotion/benchmark_20260423/error_mbert_uit_e3_focal_on_kaggle8k.md
python scripts/evaluate_emotion_model.py --model models/emotion/bench/mbert_uit_e3_focal/best --test-file data/emotion/splits/uit-vsmec/test.jsonl --output reports/emotion/benchmark_20260423/eval_mbert_uit_e3_focal_on_uit.json --error-md reports/emotion/benchmark_20260423/error_mbert_uit_e3_focal_on_uit.md

Write-Host "EVALUATIONS COMPLETED"
