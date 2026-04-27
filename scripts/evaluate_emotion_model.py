from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from app.infrastructure.hf_emotion import HfEmotionService
from app.utils.emotion_dataset import read_jsonl
from app.utils.emotion_metrics import summary_metrics


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_error_markdown(path: Path, errors: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Emotion Error Analysis",
        "",
        "| # | Gold | Pred | Score | Text |",
        "|---|------|------|-------|------|",
    ]
    for idx, error in enumerate(errors, start=1):
        text = str(error["text"]).replace("|", " ").replace("\n", " ")
        if len(text) > 160:
            text = text[:157] + "..."
        lines.append(
            f"| {idx} | {error['gold']} | {error['pred']} | {error['score']:.4f} | {text} |"
        )
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a local emotion model on test split.")
    parser.add_argument("--model", default="models/emotion/distilbert-v1/best")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0")
    parser.add_argument("--test-file", default="data/emotion/splits/test.jsonl")
    parser.add_argument("--output-json", default="reports/emotion/finetuned.json")
    parser.add_argument("--error-md", default="reports/emotion/error_analysis.md")
    parser.add_argument("--limit", type=int, default=0, help="Max samples to evaluate; 0 means all")
    parser.add_argument("--max-errors", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.test_file))
    samples = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        label = str(row.get("label", "")).strip()
        if not text or not label:
            continue
        samples.append({"text": text, "label": label})

    if not samples:
        raise RuntimeError("No valid test samples found.")

    if args.limit > 0:
        samples = samples[: args.limit]

    service = HfEmotionService(model_name=args.model, device=args.device)

    y_true: list[str] = []
    y_pred: list[str] = []
    errors: list[dict[str, Any]] = []

    step = max(1, args.batch_size)
    for idx in range(0, len(samples), step):
        batch_samples = samples[idx : idx + step]
        batch_texts = [sample["text"] for sample in batch_samples]
        batch_results = service.detect_batch(batch_texts, top_k=1, batch_size=step)

        if len(batch_results) != len(batch_samples):
            raise RuntimeError("Batch prediction size mismatch.")

        for sample, result in zip(batch_samples, batch_results, strict=False):
            y_true.append(sample["label"])
            y_pred.append(result.label)
            if result.label != sample["label"]:
                errors.append(
                    {
                        "text": sample["text"],
                        "gold": sample["label"],
                        "pred": result.label,
                        "score": result.score,
                    }
                )

    metrics = summary_metrics(y_true=y_true, y_pred=y_pred)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "device": args.device,
        "samples": len(samples),
        "metrics": metrics,
        "errors": errors[: max(1, args.max_errors)],
    }

    write_json(Path(args.output_json), report)
    write_error_markdown(Path(args.error_md), report["errors"])

    print("Evaluation completed")
    print(f"  samples={len(samples)}")
    print(f"  accuracy={metrics['accuracy']:.4f}")
    print(f"  macro_f1={metrics['macro_f1']:.4f}")
    print(f"  batch_size={step}")
    print(f"  report={args.output_json}")
    print(f"  error_md={args.error_md}")


if __name__ == "__main__":
    main()
