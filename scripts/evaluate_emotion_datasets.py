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

import kagglehub

from app.infrastructure.hf_emotion import HfEmotionService
from app.utils.emotion_dataset import (
    DEFAULT_EMOTION_LABEL_MAP,
    extract_sample,
    iter_records,
)
from app.utils.emotion_metrics import summary_metrics

DEFAULT_KAGGLE_EMOTION_HANDLE = "bhavikjikadara/emotions-dataset"


def evaluate_dataset(
    service: HfEmotionService,
    dataset_handle: str,
    dataset_name: str,
    label_map: dict[str, str],
    limit: int,
) -> dict[str, Any]:
    dataset_path = Path(kagglehub.dataset_download(dataset_handle))
    rows = iter_records(dataset_path)

    samples: list[dict[str, str]] = []
    inspected_keys: set[str] = set()
    raw_label_samples: set[str] = set()

    for row in rows:
        inspected_keys.update(str(key) for key in row.keys())
        if len(raw_label_samples) < 10:
            for key in row.keys():
                if "label" in str(key).lower() or "emotion" in str(key).lower():
                    value = row.get(key)
                    if value is not None:
                        raw_label_samples.add(str(value))

        sample = extract_sample(row, label_map)
        if sample is None:
            continue
        samples.append({"text": sample[0], "label": sample[1]})

        if limit and len(samples) >= limit:
            break

    if not samples:
        preview_keys = sorted(inspected_keys)[:20]
        preview_labels = sorted(raw_label_samples)
        raise RuntimeError(
            f"No labeled text samples detected for {dataset_name}. "
            f"Detected columns={preview_keys}, label_samples={preview_labels}. "
            "Check text/label columns or provide another dataset handle."
        )

    y_true: list[str] = []
    y_pred: list[str] = []
    errors: list[dict[str, Any]] = []
    for sample in samples:
        result = service.detect(text=sample["text"], top_k=1)
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
    return {
        "dataset_name": dataset_name,
        "dataset_handle": dataset_handle,
        "samples": len(samples),
        "metrics": metrics,
        "errors": errors,
    }


def _write_baseline_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline emotion model on Kaggle-style datasets.")
    parser.add_argument("--model", default="j-hartmann/emotion-english-distilroberta-base")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0")
    parser.add_argument("--limit", type=int, default=500, help="Max labeled samples per dataset")
    parser.add_argument("--kaggle-emotion-handle", default=DEFAULT_KAGGLE_EMOTION_HANDLE)
    parser.add_argument(
        "--report-path",
        default="reports/emotion/baseline.json",
        help="Where to save the baseline report JSON",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="Max misclassified examples to keep in report",
    )
    args = parser.parse_args()

    service = HfEmotionService(model_name=args.model, device=args.device)

    datasets_to_eval = [
        (args.kaggle_emotion_handle, "kaggle-emotion", DEFAULT_EMOTION_LABEL_MAP),
    ]

    runs: list[dict[str, Any]] = []
    for handle, name, label_map in datasets_to_eval:
        report = evaluate_dataset(
            service=service,
            dataset_handle=handle,
            dataset_name=name,
            label_map=label_map,
            limit=max(1, args.limit),
        )
        runs.append(
            {
                **report,
                "errors": report["errors"][: max(1, args.max_errors)],
            }
        )

        metrics = report["metrics"]
        print(f"[{name}] handle={handle}")
        print(f"  samples={report['samples']}")
        print(f"  accuracy={metrics['accuracy']:.4f}")
        print(f"  macro_f1={metrics['macro_f1']:.4f}")

    baseline_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "device": args.device,
        "limit": args.limit,
        "runs": runs,
    }
    _write_baseline_report(Path(args.report_path), baseline_report)
    print(f"Saved baseline report to: {args.report_path}")


if __name__ == "__main__":
    main()

