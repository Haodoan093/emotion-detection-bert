from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
from transformers import pipeline

from app.domain.entities import EmotionPrediction, EmotionResult
from app.domain.ports import EmotionService


class HfEmotionService(EmotionService):
    def __init__(self, model_name: str, device: str = "auto") -> None:
        self._model_name = model_name
        self._device = self._resolve_device(device)
        self._classifier: Callable[..., Any] | None = None

    @staticmethod
    def _resolve_device(requested_device: str) -> int:
        device = (requested_device or "auto").strip().lower()
        if device == "auto":
            return 0 if torch.cuda.is_available() else -1
        if device == "cpu":
            return -1
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                return -1
            if ":" in device:
                try:
                    return int(device.split(":", maxsplit=1)[1])
                except ValueError:
                    return 0
            return 0
        return -1

    def load_model(self) -> None:
        if self._classifier is None:
            self._classifier = cast(
                Callable[..., Any],
                pipeline(
                    "text-classification",
                    model=self._model_name,
                    device=self._device,
                    truncation=True,
                ),
            )

    def _get_classifier(self) -> Callable[..., Any]:
        self.load_model()
        if self._classifier is None:
            raise RuntimeError("Emotion classifier is not initialized.")
        return self._classifier

    @staticmethod
    def _normalize_predictions(raw: Any) -> list[EmotionPrediction]:
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            rows = raw[0]
        elif isinstance(raw, list):
            rows = raw
        else:
            rows = [raw]

        predictions: list[EmotionPrediction] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip().lower()
            if not label:
                continue
            score = float(item.get("score", 0.0) or 0.0)
            predictions.append(EmotionPrediction(label=label, score=score))
        predictions.sort(key=lambda pred: pred.score, reverse=True)
        return predictions

    def detect(self, text: str, top_k: int = 3) -> EmotionResult:
        classifier = self._get_classifier()

        raw = classifier(
            text,
            top_k=max(1, top_k),
            truncation=True,
        )
        predictions = self._normalize_predictions(raw)
        if not predictions:
            raise RuntimeError("Emotion classifier returned no predictions.")
        best = predictions[0]
        return EmotionResult(
            text=text,
            label=best.label,
            score=best.score,
            predictions=predictions,
        )

    def detect_batch(self, texts: list[str], top_k: int = 1, batch_size: int = 32) -> list[EmotionResult]:
        if not texts:
            return []

        classifier = self._get_classifier()
        raw_outputs = classifier(
            texts,
            top_k=max(1, top_k),
            truncation=True,
            batch_size=max(1, batch_size),
        )

        if len(texts) == 1:
            normalized_outputs = [raw_outputs]
        elif isinstance(raw_outputs, list):
            normalized_outputs = raw_outputs
        else:
            normalized_outputs = [raw_outputs]

        if len(normalized_outputs) != len(texts):
            raise RuntimeError("Emotion classifier returned mismatched batch output size.")

        results: list[EmotionResult] = []
        for text, raw in zip(texts, normalized_outputs, strict=False):
            predictions = self._normalize_predictions(raw)
            if not predictions:
                raise RuntimeError("Emotion classifier returned no predictions.")
            best = predictions[0]
            results.append(
                EmotionResult(
                    text=text,
                    label=best.label,
                    score=best.score,
                    predictions=predictions,
                )
            )
        return results

