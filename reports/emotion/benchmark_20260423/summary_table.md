# Benchmark Summary — 2026-04-23

## Cross-Evaluation Results (2 models × 2 datasets)

| Checkpoint | Test Set | Samples | Accuracy | Macro F1 |
|---|---|---|---|---|
| distil_kaggle8k | kaggle-8k | 806 | 0.7084 | **0.6924** |
| distil_kaggle8k | uit-vsmec | 697 | 0.2956 | 0.0963 |
| distil_uit | kaggle-8k | 806 | 0.3226 | 0.1127 |
| distil_uit | uit-vsmec | 697 | 0.4491 | **0.4092** |
| xlm_kaggle8k | kaggle-8k | 806 | 0.5062 | 0.2922 |
| xlm_kaggle8k | uit-vsmec | 697 | 0.3486 | 0.1553 |
| xlm_uit | kaggle-8k | 806 | 0.4715 | 0.1633 |
| xlm_uit | uit-vsmec | 697 | 0.4419 | 0.3174 |

## In-Domain Best (macro_f1)

| Domain | Best Checkpoint | Macro F1 |
|---|---|---|
| Kaggle 8k | distil_kaggle8k | 0.6924 |
| UIT-VSMEC | distil_uit | 0.4092 |

## Cross-Domain Min(macro_f1) Ranking

Dùng cho quyết định deploy chung cho cả 2 domain:

| Checkpoint | F1 on Kaggle 8k | F1 on UIT | min(F1) |
|---|---|---|---|
| distil_kaggle8k | 0.6924 | 0.0963 | 0.0963 |
| distil_uit | 0.1127 | 0.4092 | 0.1127 |
| xlm_kaggle8k | 0.2922 | 0.1553 | 0.1553 |
| xlm_uit | 0.1633 | 0.3174 | 0.1633 |

## Model Selection Rule (theo huong_dan_benchmark_2_models.md)

1. **Metric chính**: macro_f1
2. **In-domain winner**: `distilbert-base-multilingual-cased` trained on Kaggle 8k → macro_f1 = 0.6924 (cao nhất)
3. **Cross-domain winner (min rule)**: `xlm_uit` có min(F1)=0.1633 (cao nhất trong 4 checkpoint), nhưng vẫn rất thấp cross-domain
4. **Nhận xét**: Tất cả checkpoint đều có cross-domain rất yếu (epoch=1, CPU only). Nếu cần dùng chung 2 domain, cần train thêm epochs hoặc merge data.

## Khuyến nghị Deploy

- **Nếu chỉ dùng cho English (Kaggle domain)**: `models/emotion/bench/distil_kaggle8k/best`
- **Nếu chỉ dùng cho Vietnamese (UIT domain)**: `models/emotion/bench/distil_uit/best`
- **Model nhẹ hơn và inference nhanh hơn**: DistilBERT (~14.57 samples/s) vs XLM-R (~7.0 samples/s) → DistilBERT nhanh gấp ~2x

### EMOTION_MODEL_NAME đề xuất

```dotenv
# Nếu ưu tiên English:
EMOTION_MODEL_NAME=models/emotion/bench/distil_kaggle8k/best

# Nếu ưu tiên Vietnamese:
EMOTION_MODEL_NAME=models/emotion/bench/distil_uit/best
```

## Training Hyperparameters

| Model | Dataset | Epochs | Batch | LR | Val Accuracy | Val Macro F1 |
|---|---|---|---|---|---|---|
| distilbert-base-multilingual-cased | kaggle-8k | 1 | 8 | 2e-5 | 0.6876 | 0.6826 |
| distilbert-base-multilingual-cased | uit-vsmec | 1 | 8 | 2e-5 | 0.4253 | 0.3870 |
| xlm-roberta-base | kaggle-8k | 1 | 6 | 2e-5 | 0.5069 | 0.2882 |
| xlm-roberta-base | uit-vsmec | 1 | 6 | 2e-5 | 0.4253 | 0.3210 |
