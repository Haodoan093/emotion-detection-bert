# Emotion Plan (Detailed)

Ngay lap ke hoach: 2026-04-21

## 1) Muc tieu

1. Xay dung pipeline Emotion end-to-end trong chinh du an de co the hoc va tai lap.
2. Fine-tune model local (khong phu thuoc service inference ben ngoai).
3. Tich hop model da train vao API hien tai.
4. Co bao cao so sanh baseline truoc/sau train bang chi so ro rang.

## 2) Trang thai hien tai

1. Endpoint Emotion da co san: app/adapters/http/emotion_routes.py.
2. Emotion service dang dung Transformers pipeline local: app/infrastructure/hf_emotion.py.
3. Default model dang la j-hartmann/emotion-english-distilroberta-base trong app/core/config.py.
4. Script evaluate hien tai chu yeu danh gia 1 nguon emotion dataset.
5. Workspace chua co thu muc data/ va models/ de luu artifact noi bo.
6. .env chua co nhom bien EMOTION\_\*.

## 3) Pham vi cong viec

### Trong pham vi

1. Baseline model hien tai.
2. Thu thap + chuan hoa du lieu emotion.
3. Tao script train/evaluate local.
4. Tich hop model local vao API.
5. Viet tai lieu runbook va cap nhat README.

### Ngoai pham vi (dot nay)

1. Toi uu LLM/RAG.
2. Huan luyen ASR.
3. Trien khai production cloud.

## 4) Kien truc muc tieu cho Emotion

1. Data layer

- data/emotion/raw
- data/emotion/processed
- data/emotion/splits

2. Training layer

- scripts/prepare_emotion_dataset.py
- scripts/train_emotion_model.py
- scripts/evaluate_emotion_model.py

3. Model artifacts

- models/emotion/distilbert-v1
- models/emotion/distilbert-v1/best

4. Reporting

- reports/emotion/baseline.json
- reports/emotion/finetuned.json
- reports/emotion/error_analysis.md

5. Serving

- app/infrastructure/hf_emotion.py nhan duong dan local model
- .env tro EMOTION_MODEL_NAME ve models/emotion/distilbert-v1/best

## 5) Ke hoach chi tiet theo giai doan

### Giai doan A - Baseline va xac lap moc (0.5 ngay)

Muc tieu: Co ket qua tham chieu truoc khi train.

Cong viec:

1. Chay baseline voi model hien tai.
2. Luu ket qua vao reports/emotion/baseline.json.
3. Ghi lai thong tin moi truong chay (CPU/GPU, thoi gian infer).

Lenh du kien:

1. python scripts/evaluate_emotion_datasets.py --limit 500

Dau ra bat buoc:

1. baseline.json co Accuracy, Macro F1, F1 theo label.
2. Ghi chu nhung label bi nham nhieu nhat.

### Giai doan B - Chuan hoa du lieu (1 ngay)

Muc tieu: Tao bo du lieu train/val/test sach, co the tai lap.

Cong viec:

1. Tao script scripts/prepare_emotion_dataset.py.
2. Doc cac nguon du lieu emotion, map ve schema chung: text, label.
3. Chuan hoa label ve bo nhan muc tieu.
4. Loai bo ban ghi rong, qua ngan, trung lap tho.
5. Tach train/val/test theo stratified split (80/10/10).

Dau ra bat buoc:

1. data/emotion/splits/train.jsonl
2. data/emotion/splits/val.jsonl
3. data/emotion/splits/test.jsonl
4. reports/emotion/data_profile.json (so luong mau tung label)

Tieu chi dat:

1. Tat ca split co du cac label chinh.
2. Khong co dong text rong.

### Giai doan C - Train model local v1 (1 ngay)

Muc tieu: Co model local checkpoint chay duoc.

Cong viec:

1. Tao script scripts/train_emotion_model.py bang Hugging Face Trainer.
2. Chon backbone:

- Neu uu tien da ngon ngu: distilbert-base-multilingual-cased
- Neu uu tien tieng Anh: distilbert-base-uncased

3. Cau hinh train de xuat:

- max_length: 128
- learning_rate: 2e-5
- batch_size: 16
- num_train_epochs: 3 den 5
- weight_decay: 0.01
- early_stopping_patience: 2

4. Luu best checkpoint vao models/emotion/distilbert-v1/best.

Dau ra bat buoc:

1. Co folder checkpoint day du (config, tokenizer, model).
2. Co train log theo epoch.

### Giai doan D - Danh gia sau train va phan tich loi (0.5 ngay)

Muc tieu: Chung minh duoc loi ich cua model da fine-tune.

Cong viec:

1. Tao script scripts/evaluate_emotion_model.py.
2. Chay tren test split co dinh.
3. Tinh chi so:

- Accuracy
- Macro F1
- F1 theo tung label
- Confusion matrix

4. Xuat 20 truong hop du doan sai de phan tich.

Dau ra bat buoc:

1. reports/emotion/finetuned.json
2. reports/emotion/error_analysis.md
3. So sanh baseline vs finetuned.

Tieu chi dat:

1. Macro F1 cai thien hoac giu on dinh nhung latency tot hon.
2. Co giai thich duoc vi sao cac case sai xay ra.

### Giai doan E - Tich hop API voi model local (0.5 ngay)

Muc tieu: Endpoint Emotion su dung model ban vua train.

Cong viec:

1. Cap nhat .env voi nhom bien:

- EMOTION_ENABLED=true
- EMOTION_MODEL_NAME=models/emotion/distilbert-v1/best
- EMOTION_DEVICE=auto (hoac cuda:0)

2. Dam bao startup tao duoc DetectEmotionUseCase.
3. Test endpoint POST /emotion/detect.
4. Tao smoke test nho cho route emotion.

Dau ra bat buoc:

1. API tra ve label/score/predictions tu model local.
2. Log startup thong bao model local da duoc nap.

### Giai doan F - Hoan thien tai lieu va dong goi quy trinh (0.5 ngay)

Muc tieu: Nguoi moi vao co the chay lai toan bo trong 1 lan.

Cong viec:

1. Cap nhat README phan Emotion theo workflow moi.
2. Them file .env.example co cac bien EMOTION\_\*.
3. Ghi runbook ngan:

- Chuan bi data
- Train
- Evaluate
- Serve API

Dau ra bat buoc:

1. README co muc Emotion end-to-end.
2. Co checklist troubleshooting.

## 6) Danh sach file du kien tao/sua

### Tao moi

1. scripts/prepare_emotion_dataset.py
2. scripts/train_emotion_model.py
3. scripts/evaluate_emotion_model.py
4. tests/test_emotion_service.py
5. tests/test_emotion_route.py
6. reports/emotion/.gitkeep
7. data/emotion/.gitkeep
8. models/emotion/.gitkeep

### Chinh sua

1. app/core/config.py
2. app/main.py (neu can toi uu startup theo mode)
3. README.md
4. .env.example
5. scripts/evaluate_emotion_datasets.py

## 7) Ke hoach thoi gian de xuat (4 ngay lam viec)

1. Ngay 1

- Giai doan A + B (baseline + data prep)

2. Ngay 2

- Giai doan C (train v1)

3. Ngay 3

- Giai doan D + E (evaluate + integrate API)

4. Ngay 4

- Giai doan F (docs + cleanup + demo)

## 8) Tieu chi danh gia thanh cong (Definition of Done)

1. Co bo split train/val/test local, tai lap duoc.
2. Co model checkpoint local duoc train trong du an.
3. Endpoint Emotion tra ket qua tu model local.
4. Co bao cao baseline vs finetuned va error analysis.
5. Co huong dan run end-to-end cho nguoi khac.

## 9) Rui ro chinh va cach giam thieu

1. Mat can bang label

- Giam thieu: class weight, oversampling, macro F1 la metric chinh.

2. Label map khong dong nhat giua dataset

- Giam thieu: duy tri bang mapping trung tam va unit test cho mapping.

3. Khong co GPU

- Giam thieu: train batch nho hon, dung gradient accumulation, giam epochs.

4. Tai du lieu online that bai

- Giam thieu: cache local vao data/emotion/raw va cho phep chay lai offline.

5. Overfitting

- Giam thieu: early stopping, monitor val macro F1, giam learning rate.

## 10) Checklist thuc thi nhanh

1. Tao thu muc data, models, reports.
2. Chay baseline va luu ket qua.
3. Viet script prepare data va tao split.
4. Viet script train va train v1.
5. Viet script evaluate + error analysis.
6. Tro model local trong .env.
7. Test endpoint emotion.
8. Cap nhat README va .env.example.
