import json
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')

def main():
    bench_dir = Path("reports/emotion/benchmark_20260423")
    
    models = [
        ("distil_kaggle8k_e3", "Kaggle 8k"),
        ("distil_uit_e3_focal", "UIT-VSMEC"),
        ("mbert_kaggle8k_e3", "Kaggle 8k"),
        ("mbert_uit_e3_focal", "UIT-VSMEC")
    ]
    
    test_sets = ["kaggle8k", "uit"]
    
    print("### Kết quả Phase 2: Epoch=3, Focal Loss")
    print("| Model | Domain Huấn luyện | Test Kaggle 8k (EN) (F1/Acc) | Test UIT-VSMEC (VI) (F1/Acc) |")
    print("|-------|------------------|------------------------------|-----------------------------|")
    
    for model_id, train_domain in models:
        row = [model_id, train_domain]
        for test_domain in test_sets:
            file_path = bench_dir / f"eval_{model_id}_on_{test_domain}.json"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    metrics = data.get("metrics", {})
                    f1 = float(metrics.get("macro_f1", 0.0))
                    acc = float(metrics.get("accuracy", 0.0))
                    row.append(f"{f1:.4f} / {acc:.4f}")
            else:
                row.append("N/A")
        
        print(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

if __name__ == '__main__':
    main()
