from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
from pathlib import Path
import argparse
import sys

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
except ImportError:
    print("Vui lòng cài đặt matplotlib, seaborn và scikit-learn để chạy script này.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="reports/emotion/benchmark_20260423/eval_xlm_uit_on_uit.json")
    parser.add_argument("--output", "-o", type=str, default="reports/emotion/confusion_matrix_uit.png")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    data = json.loads(input_path.read_text(encoding="utf-8"))
    
    metrics = data.get("metrics", {})
    cm = metrics.get("confusion_matrix")
    labels = metrics.get("labels", [])

    if not cm or not labels:
        print("Không tìm thấy confusion_matrix hoặc labels trong file JSON.")
        return

    # Normalize confusion matrix
    import numpy as np
    cm_array = np.array(cm, dtype=np.float32)
    # Normalize by row (true label)
    row_sums = cm_array.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    cm_norm = cm_array / row_sums

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Normalized Confusion Matrix ({input_path.name})")
    plt.xlabel("Predicted")
    plt.ylabel("Gold Label")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Đã lưu Confusion Matrix tại: {output_path}")

if __name__ == "__main__":
    main()
