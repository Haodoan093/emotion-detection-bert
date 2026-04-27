from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import kagglehub
from huggingface_hub import snapshot_download

from app.utils.emotion_dataset import (
    DEFAULT_EMOTION_LABEL_MAP,
    deduplicate_samples,
    extract_sample,
    iter_records,
    label_distribution,
    stratified_split,
    write_jsonl,
)

DEFAULT_KAGGLE_EMOTION_HANDLE = "bhavikjikadara/emotions-dataset"
DEFAULT_UIT_VSMEC_DATASET_ID = "duwuonline/UIT-VSMEC"


def collect_samples(
    dataset_source: str,
    dataset_handle: str,
    label_map: dict[str, str],
    limit: int,
    min_text_length: int,
) -> list[dict[str, str]]:
    if dataset_source == "kaggle":
        dataset_path = Path(kagglehub.dataset_download(dataset_handle))
    elif dataset_source == "huggingface":
        dataset_path = Path(
            snapshot_download(
                repo_id=dataset_handle,
                repo_type="dataset",
                allow_patterns=["*.csv", "*.json", "*.jsonl"],
            )
        )
    else:
        raise ValueError(f"Unsupported dataset source: {dataset_source}")

    records = iter_records(dataset_path)

    samples: list[dict[str, str]] = []
    for row in records:
        sample = extract_sample(row, label_map)
        if sample is None:
            continue
        text, label = sample
        if len(text.strip()) < min_text_length:
            continue
        samples.append(
            {
                "text": text,
                "label": label,
                "source": dataset_handle,
            }
        )
        if limit > 0 and len(samples) >= limit:
            break

    if not samples:
        raise RuntimeError("No valid emotion samples were collected from dataset.")

    return samples


def write_profile(
    path: Path,
    source_handle: str,
    raw_samples: list[dict[str, str]],
    processed_samples: list[dict[str, str]],
    train_samples: list[dict[str, str]],
    val_samples: list[dict[str, str]],
    test_samples: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_handle": source_handle,
        "counts": {
            "raw": len(raw_samples),
            "processed": len(processed_samples),
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "distribution": {
            "raw": label_distribution(raw_samples),
            "processed": label_distribution(processed_samples),
            "train": label_distribution(train_samples),
            "val": label_distribution(val_samples),
            "test": label_distribution(test_samples),
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare emotion dataset splits for local training.")
    parser.add_argument(
        "--source",
        choices=["kaggle-emotion", "uit-vsmec"],
        default="kaggle-emotion",
        help="Dataset source to ingest",
    )
    parser.add_argument("--kaggle-handle", default=DEFAULT_KAGGLE_EMOTION_HANDLE)
    parser.add_argument("--hf-dataset-id", default=DEFAULT_UIT_VSMEC_DATASET_ID)
    parser.add_argument("--limit", type=int, default=0, help="Max samples to collect; 0 means all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--min-text-length", type=int, default=3)
    parser.add_argument("--raw-output", default="data/emotion/raw/kaggle-emotion.jsonl")
    parser.add_argument("--processed-output", default="data/emotion/processed/kaggle-emotion.cleaned.jsonl")
    parser.add_argument("--splits-dir", default="data/emotion/splits")
    parser.add_argument("--profile-output", default="reports/emotion/data_profile.json")
    args = parser.parse_args()

    if args.source == "kaggle-emotion":
        dataset_source = "kaggle"
        dataset_handle = args.kaggle_handle
    else:
        dataset_source = "huggingface"
        dataset_handle = args.hf_dataset_id

    raw_samples = collect_samples(
        dataset_source=dataset_source,
        dataset_handle=dataset_handle,
        label_map=DEFAULT_EMOTION_LABEL_MAP,
        limit=max(0, args.limit),
        min_text_length=max(1, args.min_text_length),
    )
    processed_samples = deduplicate_samples(raw_samples)

    train_samples, val_samples, test_samples = stratified_split(
        samples=processed_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    write_jsonl(Path(args.raw_output), raw_samples)
    write_jsonl(Path(args.processed_output), processed_samples)

    splits_dir = Path(args.splits_dir)
    write_jsonl(splits_dir / "train.jsonl", train_samples)
    write_jsonl(splits_dir / "val.jsonl", val_samples)
    write_jsonl(splits_dir / "test.jsonl", test_samples)

    write_profile(
        path=Path(args.profile_output),
        source_handle=f"{args.source}:{dataset_handle}",
        raw_samples=raw_samples,
        processed_samples=processed_samples,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
    )

    print("Prepared emotion dataset splits")
    print(f"  raw={len(raw_samples)}")
    print(f"  processed={len(processed_samples)}")
    print(f"  train={len(train_samples)}")
    print(f"  val={len(val_samples)}")
    print(f"  test={len(test_samples)}")
    print(f"  source={args.source}:{dataset_handle}")
    print(f"  profile={args.profile_output}")


if __name__ == "__main__":
    main()
