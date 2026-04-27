# Huong Dan Chi Tiet Benchmark 2 Model Emotion

Cap nhat: 2026-04-23

## 1) Muc tieu

Tai lieu nay mo ta day du quy trinh benchmark 2 model:

1. distilbert-base-multilingual-cased
2. xlm-roberta-base

tren 2 bo du lieu dang co trong du an:

1. Kaggle subset 8k
2. UIT-VSMEC

Quy trinh gom: train tuan tu, evaluate, tong hop ket qua, chon model deploy.

## 2) Du lieu dau vao

### 2.1 Kaggle 8k (da duoc tao)

- File processed: data/emotion/processed/kaggle-emotion.cleaned.8k.jsonl
- Splits:
  - data/emotion/splits/kaggle-8k/train.jsonl
  - data/emotion/splits/kaggle-8k/val.jsonl
  - data/emotion/splits/kaggle-8k/test.jsonl
- Profile: reports/emotion/data_profile_kaggle_8k.json

### 2.2 UIT-VSMEC

- File processed: data/emotion/processed/uit-vsmec.cleaned.jsonl
- Splits:
  - data/emotion/splits/uit-vsmec/train.jsonl
  - data/emotion/splits/uit-vsmec/val.jsonl
  - data/emotion/splits/uit-vsmec/test.jsonl
- Profile: reports/emotion/data_profile_uit_vsmec.json

## 3) Nguyen tac chay

1. Chay tuan tu tung job, khong chay song song.
2. Tren may CPU, uu tien epoch nho (1 truoc, sau do moi nang len 2 neu on).
3. Neu job bi ngat, rerun lai dung lenh voi output_dir moi hoac xoa output_dir cu.

## 4) Chuan bi thu muc benchmark

Chay 1 lan:

```powershell
$bench = "reports/emotion/benchmark_20260423"
New-Item -ItemType Directory -Force -Path $bench | Out-Null
```

## 5) Train 4 job (2 model x 2 dataset)

Luu y: nen chay epoch=1 truoc de co ket qua nhanh va tranh job qua lau tren CPU.

### 5.1 Distil tren Kaggle 8k

```powershell
python scripts/train_emotion_model.py `
  --model distilbert-base-multilingual-cased `
  --train-file data/emotion/splits/kaggle-8k/train.jsonl `
  --val-file data/emotion/splits/kaggle-8k/val.jsonl `
  --epochs 1 `
  --batch-size 8 `
  --learning-rate 2e-5 `
  --output-dir models/emotion/bench/distil_kaggle8k `
  --best-dir models/emotion/bench/distil_kaggle8k/best `
  --summary-file reports/emotion/benchmark_20260423/train_distil_kaggle8k.json
```

### 5.2 Distil tren UIT-VSMEC

```powershell
python scripts/train_emotion_model.py `
  --model distilbert-base-multilingual-cased `
  --train-file data/emotion/splits/uit-vsmec/train.jsonl `
  --val-file data/emotion/splits/uit-vsmec/val.jsonl `
  --epochs 1 `
  --batch-size 8 `
  --learning-rate 2e-5 `
  --output-dir models/emotion/bench/distil_uit `
  --best-dir models/emotion/bench/distil_uit/best `
  --summary-file reports/emotion/benchmark_20260423/train_distil_uit.json
```

### 5.3 XLM-R tren Kaggle 8k

```powershell
python scripts/train_emotion_model.py `
  --model xlm-roberta-base `
  --train-file data/emotion/splits/kaggle-8k/train.jsonl `
  --val-file data/emotion/splits/kaggle-8k/val.jsonl `
  --epochs 1 `
  --batch-size 6 `
  --learning-rate 2e-5 `
  --output-dir models/emotion/bench/xlm_kaggle8k `
  --best-dir models/emotion/bench/xlm_kaggle8k/best `
  --summary-file reports/emotion/benchmark_20260423/train_xlm_kaggle8k.json
```

### 5.4 XLM-R tren UIT-VSMEC

```powershell
python scripts/train_emotion_model.py `
  --model xlm-roberta-base `
  --train-file data/emotion/splits/uit-vsmec/train.jsonl `
  --val-file data/emotion/splits/uit-vsmec/val.jsonl `
  --epochs 1 `
  --batch-size 6 `
  --learning-rate 2e-5 `
  --output-dir models/emotion/bench/xlm_uit `
  --best-dir models/emotion/bench/xlm_uit/best `
  --summary-file reports/emotion/benchmark_20260423/train_xlm_uit.json
```

## 6) Evaluate 8 bai test (evaluate cheo)

Ly do: xem kha nang in-domain va cross-domain.

### 6.1 Distil-Kaggle8k checkpoint

```powershell
python scripts/evaluate_emotion_model.py --model models/emotion/bench/distil_kaggle8k/best --test-file data/emotion/splits/kaggle-8k/test.jsonl --batch-size 64 --output-json reports/emotion/benchmark_20260423/eval_distil_kaggle8k_on_kaggle8k.json --error-md reports/emotion/benchmark_20260423/error_distil_kaggle8k_on_kaggle8k.md
python scripts/evaluate_emotion_model.py --model models/emotion/bench/distil_kaggle8k/best --test-file data/emotion/splits/uit-vsmec/test.jsonl --batch-size 64 --output-json reports/emotion/benchmark_20260423/eval_distil_kaggle8k_on_uit.json --error-md reports/emotion/benchmark_20260423/error_distil_kaggle8k_on_uit.md
```

### 6.2 Distil-UIT checkpoint

```powershell
python scripts/evaluate_emotion_model.py --model models/emotion/bench/distil_uit/best --test-file data/emotion/splits/kaggle-8k/test.jsonl --batch-size 64 --output-json reports/emotion/benchmark_20260423/eval_distil_uit_on_kaggle8k.json --error-md reports/emotion/benchmark_20260423/error_distil_uit_on_kaggle8k.md
python scripts/evaluate_emotion_model.py --model models/emotion/bench/distil_uit/best --test-file data/emotion/splits/uit-vsmec/test.jsonl --batch-size 64 --output-json reports/emotion/benchmark_20260423/eval_distil_uit_on_uit.json --error-md reports/emotion/benchmark_20260423/error_distil_uit_on_uit.md
```

### 6.3 XLM-Kaggle8k checkpoint

```powershell
python scripts/evaluate_emotion_model.py --model models/emotion/bench/xlm_kaggle8k/best --test-file data/emotion/splits/kaggle-8k/test.jsonl --batch-size 64 --output-json reports/emotion/benchmark_20260423/eval_xlm_kaggle8k_on_kaggle8k.json --error-md reports/emotion/benchmark_20260423/error_xlm_kaggle8k_on_kaggle8k.md
python scripts/evaluate_emotion_model.py --model models/emotion/bench/xlm_kaggle8k/best --test-file data/emotion/splits/uit-vsmec/test.jsonl --batch-size 64 --output-json reports/emotion/benchmark_20260423/eval_xlm_kaggle8k_on_uit.json --error-md reports/emotion/benchmark_20260423/error_xlm_kaggle8k_on_uit.md
```

### 6.4 XLM-UIT checkpoint

```powershell
python scripts/evaluate_emotion_model.py --model models/emotion/bench/xlm_uit/best --test-file data/emotion/splits/kaggle-8k/test.jsonl --batch-size 64 --output-json reports/emotion/benchmark_20260423/eval_xlm_uit_on_kaggle8k.json --error-md reports/emotion/benchmark_20260423/error_xlm_uit_on_kaggle8k.md
python scripts/evaluate_emotion_model.py --model models/emotion/bench/xlm_uit/best --test-file data/emotion/splits/uit-vsmec/test.jsonl --batch-size 64 --output-json reports/emotion/benchmark_20260423/eval_xlm_uit_on_uit.json --error-md reports/emotion/benchmark_20260423/error_xlm_uit_on_uit.md
```

## 7) Tong hop ket qua tu dong

Chay doan lenh sau de tao bang tong hop:

```powershell
$script = @'
from __future__ import annotations
import json
from pathlib import Path

bench = Path("reports/emotion/benchmark_20260423")
rows = []
for p in sorted(bench.glob("eval_*.json")):
    obj = json.loads(p.read_text(encoding="utf-8"))
    metrics = obj.get("metrics", {})
    rows.append({
        "file": p.name,
        "model": obj.get("model"),
        "samples": obj.get("samples"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
    })

out = bench / "summary_table.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"saved={out}")
for r in rows:
    print(f"{r['file']}: acc={r['accuracy']:.4f} macro_f1={r['macro_f1']:.4f}")
'@
$script | python -
```

## 8) Rule chon model cuoi

De xuat rule:

1. Metric chinh: macro_f1.
2. Uu tien model co macro_f1 cao nhat tren test target chinh.
3. Neu can model dung chung cho 2 domain, chon model co min(macro_f1_kaggle, macro_f1_uit) cao nhat.
4. Neu chenh lech nho hon 0.01, uu tien model nhe hon/latency tot hon.

## 9) Cap nhat runtime sau khi chot

Sua bien moi truong:

```dotenv
EMOTION_ENABLED=true
EMOTION_MODEL_NAME=models/emotion/bench/<ten_checkpoint>/best
EMOTION_DEVICE=auto
```

Vi du:

```dotenv
EMOTION_MODEL_NAME=models/emotion/bench/xlm_kaggle8k/best
```

## 10) Checklist hoan thanh

1. Co 4 file train summary trong reports/emotion/benchmark_20260423.
2. Co 8 file eval_*.json va 8 file error_*.md trong reports/emotion/benchmark_20260423.
3. Co summary_table.json de so sanh nhanh.
4. Da cap nhat EMOTION_MODEL_NAME theo checkpoint thang.
