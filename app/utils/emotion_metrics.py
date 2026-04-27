from __future__ import annotations

from collections import defaultdict
from typing import Any


def accuracy_score(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for expected, predicted in zip(y_true, y_pred, strict=False) if expected == predicted)
    return correct / len(y_true)


def confusion_matrix(labels: list[str], y_true: list[str], y_pred: list[str]) -> list[list[int]]:
    index_by_label = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for expected, predicted in zip(y_true, y_pred, strict=False):
        expected_idx = index_by_label.get(expected)
        predicted_idx = index_by_label.get(predicted)
        if expected_idx is None or predicted_idx is None:
            continue
        matrix[expected_idx][predicted_idx] += 1
    return matrix


def per_label_scores(labels: list[str], y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, float]]:
    matrix = confusion_matrix(labels=labels, y_true=y_true, y_pred=y_pred)
    scores: dict[str, dict[str, float]] = {}

    for i, label in enumerate(labels):
        tp = float(matrix[i][i])
        fp = float(sum(matrix[row][i] for row in range(len(labels)) if row != i))
        fn = float(sum(matrix[i][col] for col in range(len(labels)) if col != i))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        scores[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(sum(matrix[i])),
        }

    return scores


def macro_f1_score(labels: list[str], y_true: list[str], y_pred: list[str]) -> float:
    if not labels:
        return 0.0
    scores = per_label_scores(labels=labels, y_true=y_true, y_pred=y_pred)
    f1_values = [scores[label]["f1"] for label in labels]
    return sum(f1_values) / len(f1_values)


def summary_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    label_scores = per_label_scores(labels=labels, y_true=y_true, y_pred=y_pred)
    return {
        "labels": labels,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": macro_f1_score(labels=labels, y_true=y_true, y_pred=y_pred),
        "per_label": label_scores,
        "confusion_matrix": confusion_matrix(labels=labels, y_true=y_true, y_pred=y_pred),
        "support": dict(sorted(defaultdict(int, ((label, y_true.count(label)) for label in labels)).items())),
    }
