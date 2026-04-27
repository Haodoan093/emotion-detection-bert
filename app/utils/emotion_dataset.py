from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

TEXT_KEYS = (
    "text",
    "sentence",
    "content",
    "utterance",
    "transcript",
    "review",
    "message",
)
LABEL_KEYS = ("label", "emotion", "sentiment", "tag", "class")

EMOTION_ID_MAP = {
    "0": "sadness",
    "1": "joy",
    "2": "love",
    "3": "anger",
    "4": "fear",
    "5": "surprise",
}

DEFAULT_EMOTION_LABEL_MAP = {
    "sad": "sadness",
    "sadness": "sadness",
    "joy": "joy",
    "enjoyment": "joy",
    "happy": "joy",
    "happiness": "joy",
    "love": "love",
    "anger": "anger",
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "surprise": "surprise",
    "other": "neutral",
    "neutral": "neutral",
}


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).strip().lower())


def canonicalize_row(row: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for key, value in row.items():
        canonical[normalize_key(key)] = value
    return canonical


def normalize_label(raw: Any, mapping: dict[str, str]) -> str | None:
    if raw is None:
        return None

    if isinstance(raw, int):
        mapped = EMOTION_ID_MAP.get(str(raw))
        if mapped:
            return mapped

    text = str(raw).strip().lower()
    if not text:
        return None

    compact = re.sub(r"\s+", "", text)
    if compact in EMOTION_ID_MAP:
        return EMOTION_ID_MAP[compact]

    if text in mapping:
        return mapping[text]
    return mapping.get(compact)


def extract_sample(row: dict[str, Any], label_map: dict[str, str]) -> tuple[str, str] | None:
    canonical_row = canonicalize_row(row)

    text_value: str | None = None
    for key in TEXT_KEYS:
        value = row.get(key)
        if value is None:
            value = canonical_row.get(normalize_key(key))
        if isinstance(value, str) and value.strip():
            text_value = value.strip()
            break

    label_value: str | None = None
    for key in LABEL_KEYS:
        value = row.get(key)
        if value is None:
            value = canonical_row.get(normalize_key(key))
        normalized = normalize_label(value, label_map)
        if normalized:
            label_value = normalized
            break

    if not text_value or not label_value:
        return None
    return text_value, label_value


def iter_records(dataset_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for csv_path in dataset_path.rglob("*.csv"):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                records.append(dict(row))

    for jsonl_path in dataset_path.rglob("*.jsonl"):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = line.strip()
                if not item:
                    continue
                try:
                    payload = json.loads(item)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)

    for json_path in dataset_path.rglob("*.json"):
        with json_path.open("r", encoding="utf-8") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError:
                continue
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    records.append(item)
        elif isinstance(payload, dict):
            records.append(payload)

    return records


def deduplicate_samples(samples: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for sample in samples:
        key = (sample["text"].strip().lower(), sample["label"].strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sample)
    return deduped


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = line.strip()
            if not item:
                continue
            payload = json.loads(item)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def label_distribution(samples: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        label = sample["label"]
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def stratified_split(
    samples: list[dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if not 0 <= test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.")

    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for sample in samples:
        by_label[sample["label"]].append(sample)

    rng = random.Random(seed)
    train: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    test: list[dict[str, str]] = []

    for bucket in by_label.values():
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        if n >= 3:
            if n_train == 0:
                n_train = 1
            if n_val == 0:
                n_val = 1
            n_test = n - n_train - n_val
            if n_test == 0:
                n_test = 1
                if n_train > n_val:
                    n_train -= 1
                else:
                    n_val -= 1

        train.extend(bucket[:n_train])
        val.extend(bucket[n_train : n_train + n_val])
        test.extend(bucket[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test
