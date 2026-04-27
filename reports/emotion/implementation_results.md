# Emotion Implementation and Results

Ngay cap nhat: 2026-04-22

## 1. Muc tieu

Tai lieu nay tong hop:

1. Cac buoc da trien khai cho emotion pipeline.
2. Cau hinh train da dung.
3. Ket qua truoc/sau de danh gia nhanh.
4. Cach chay lai de tai lap ket qua.

## 2. Artifact map

1. Baseline ban dau (sample nho): reports/emotion/baseline.json
2. Baseline on dinh hon (500 mau): reports/emotion/baseline_500.json
3. Data profile sau prepare: reports/emotion/data_profile.json
4. Train summary v1: reports/emotion/train_summary.json
5. Eval test v1: reports/emotion/finetuned.json
6. Error analysis v1: reports/emotion/error_analysis.md
7. Train summary v2 weighted: reports/emotion/train_summary_v2_weighted.json
8. Eval test v2 weighted: reports/emotion/finetuned_v2_weighted.json
9. Error analysis v2 weighted: reports/emotion/error_analysis_v2_weighted.md

## 3. Cac buoc da trien khai

### B1) Prepare data

Script: scripts/prepare_emotion_dataset.py

Viec da lam:

1. Download du lieu emotion tu Kaggle handle.
2. Chuan hoa text + label.
3. Loai bo mau khong hop le.
4. Stratified split train/val/test.
5. Luu profile phan bo nhan.

Ket qua profile:

1. raw: 2000
2. train: 1598
3. val: 197
4. test: 205

### B2) Train v1 (moc dau)

Model: models/emotion/distilbert-v1/best

Dac diem:

1. Chay epoch thap, chua xu ly imbalance hieu qua.
2. Dan den macro F1 thap tren test.

### B3) Cai thien train v2 weighted

Script: scripts/train_emotion_model.py

Cai thien da ap dung:

1. WeightedLossTrainer de xu ly lech lop.
2. Class weights tinh tu tan suat train labels.
3. Them warmup_ratio va gradient_accumulation_steps.
4. Chon best model theo macro_f1.

Model moi:

1. output_dir: models/emotion/distilbert-v2_weighted
2. best_dir: models/emotion/distilbert-v2_weighted/best

## 4. Cau hinh train v2 weighted (thuc te da chay)

Nguon: reports/emotion/train_summary_v2_weighted.json

1. model_name: distilbert-base-uncased
2. train_samples: 1598
3. val_samples: 197
4. batch_size: 8
5. epochs: 3.0
6. learning_rate: 2e-05
7. max_length: 128
8. weight_decay: 0.01
9. warmup_ratio: 0.1
10. gradient_accumulation_steps: 1
11. use_class_weights: true
12. eval_macro_f1 (val): 0.8476314983304231

Class weights da dung:

1. anger: 0.6975
2. fear: 0.8332
3. joy: 0.2697
4. love: 1.1997
5. sadness: 0.3218
6. surprise: 2.6780

## 5. Ket qua so sanh

Nguon:

1. reports/emotion/finetuned.json
2. reports/emotion/finetuned_v2_weighted.json
3. reports/emotion/baseline_500.json

### 5.1 Test set (205 mau): v1 vs v2 weighted

1. v1 accuracy: 0.5853658536585366
2. v1 macro_f1: 0.24110259455180105
3. v2 accuracy: 0.8292682926829268
4. v2 macro_f1: 0.7820419809716227

Chenhlech (v2 - v1):

1. accuracy: +0.2439024390243902
2. macro_f1: +0.5409393864198217

### 5.2 Baseline 500 vs v2 weighted

1. baseline_500 accuracy: 0.842
2. baseline_500 macro_f1: 0.6954329164585288
3. v2 test accuracy: 0.8292682926829268
4. v2 test macro_f1: 0.7820419809716227

Chenhlech (v2 - baseline_500):

1. accuracy: -0.0127317073170732
2. macro_f1: +0.0866090645130939

Nhan xet:

1. Accuracy giam nhe so voi baseline_500, nhung macro_f1 tang ro.
2. Macro_f1 tang la dau hieu model can bang lop tot hon, phu hop bai toan lech lop.

## 6. Cach tai lap nhanh

1. Baseline 500:
   python scripts/evaluate_emotion_datasets.py --limit 500 --report-path reports/emotion/baseline_500.json

2. Prepare data:
   python scripts/prepare_emotion_dataset.py --limit 2000 --profile-output reports/emotion/data_profile.json

3. Train weighted v2:
   python scripts/train_emotion_model.py --model distilbert-base-uncased --epochs 3 --batch-size 8 --learning-rate 2e-5 --output-dir models/emotion/distilbert-v2_weighted --best-dir models/emotion/distilbert-v2_weighted/best --summary-file reports/emotion/train_summary_v2_weighted.json

4. Evaluate v2:
   python scripts/evaluate_emotion_model.py --model models/emotion/distilbert-v2_weighted/best --test-file data/emotion/splits/test.jsonl --output-json reports/emotion/finetuned_v2_weighted.json --error-md reports/emotion/error_analysis_v2_weighted.md

5. Runtime config:
   EMOTION_MODEL_NAME=models/emotion/distilbert-v2_weighted/best

## 7. Tieu chi danh gia de quyet dinh

De xuat gate tam thoi cho emotion text model:

1. macro_f1 >= 0.75
2. khong co lop nao F1 = 0
3. support thap (vi du surprise) van co recall > 0.5
4. ket qua on dinh qua 2 lan train voi seed khac nhau

## 8. Ke hoach tiep theo

1. Chay sweep 6 cau hinh (lr, batch, epochs) de chon best theo macro_f1 test.
2. Thu model multilingual cho data tieng Viet/mixed.
3. Mo rong theo huong multimodal (audio + text) nhu paper tham khao.
