from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
from pathlib import Path
from collections import defaultdict
import argparse

ERROR_CATEGORIES = {
    "sarcasm": ["mỉa", "kiểu", "haha", "perfect"],
    "negation": ["không", "chẳng", "chả", "đâu có"],
    "intensifier": ["rất", "cực", "quá", "siêu"],
    "slang": ["vcl", "vl", "bff", "crush"],
}

# Các cặp từ dễ gây nhầm lẫn
OVERLAP_PAIRS = [("sadness", "fear"), ("joy", "surprise"), ("anger", "sadness")]

def categorize_error(text: str, true_label: str, pred_label: str) -> str:
    text_lower = text.lower()
    for category, keywords in ERROR_CATEGORIES.items():
        if any(kw in text_lower for kw in keywords):
            return category
            
    if (true_label, pred_label) in OVERLAP_PAIRS or (pred_label, true_label) in OVERLAP_PAIRS:
        return "semantic_overlap"
        
    return "other"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="reports/emotion/benchmark_20260423/eval_xlm_uit_on_uit.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    data = json.loads(input_path.read_text(encoding="utf-8"))
    errors = data.get("errors", [])
    
    if not errors:
        print("Không có lỗi nào để phân tích.")
        return

    category_counts = defaultdict(int)
    examples = {}

    for error in errors:
        text = error["text"]
        cat = categorize_error(text, error["gold"], error["pred"])
        category_counts[cat] += 1
        if cat not in examples:
            examples[cat] = {
                "text": text,
                "gold": error["gold"],
                "pred": error["pred"]
            }

    total_errors = len(errors)
    
    print(f"### Error Analysis for {input_path.name}")
    print(f"Total errors analyzed: {total_errors}\n")
    print("| Loại lỗi | Số lượng | % | Ví dụ |")
    print("|---|---|---|---|")
    
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_errors) * 100
        ex = examples[cat]
        # Shorten text for table
        short_text = ex["text"][:50] + "..." if len(ex["text"]) > 50 else ex["text"]
        print(f"| {cat} | {count} | {pct:.1f}% | \"{short_text}\" (Gold: {ex['gold']}, Pred: {ex['pred']}) |")

if __name__ == "__main__":
    main()
