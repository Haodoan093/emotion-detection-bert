from __future__ import annotations

import argparse
from collections import Counter
import inspect
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from app.utils.emotion_dataset import label_distribution, read_jsonl
from app.utils.emotion_metrics import summary_metrics


@dataclass
class EmotionSample:
    text: str
    label: str


class EmotionTextDataset(Dataset):
    def __init__(
        self,
        samples: list[EmotionSample],
        tokenizer: Any,
        label_to_id: dict[str, int],
        max_length: int,
    ) -> None:
        self._texts = [sample.text for sample in samples]
        self._labels = [label_to_id[sample.label] for sample in samples]
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self._tokenizer(
            self._texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self._max_length,
        )
        item = {key: torch.tensor(value) for key, value in encoding.items()}
        item["labels"] = torch.tensor(self._labels[idx])
        return item


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class WeightedLossTrainer(Trainer):
    def __init__(self, *args: Any, class_weights: torch.Tensor | None = None, use_focal_loss: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights
        self._use_focal_loss = use_focal_loss

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        _ = num_items_in_batch
        labels = inputs.get("labels")
        outputs = model(**inputs)

        loss: torch.Tensor | None = None
        if labels is not None:
            logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits
            if getattr(self, "_use_focal_loss", False):
                loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            elif self._class_weights is not None:
                class_weights = self._class_weights.to(device=logits.device, dtype=logits.dtype)
                loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        if loss is None:
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs.loss
            if loss is None and labels is not None:
                logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        if loss is None:
            raise RuntimeError("Unable to compute training loss.")

        if return_outputs:
            return loss, outputs
        return loss


def _load_samples(path: Path) -> list[EmotionSample]:
    rows = read_jsonl(path)
    samples: list[EmotionSample] = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        label = str(row.get("label", "")).strip()
        if not text or not label:
            continue
        samples.append(EmotionSample(text=text, label=label))
    if not samples:
        raise RuntimeError(f"No valid samples found in {path}")
    return samples


def _compute_metrics_factory(id_to_label: dict[int, str]):
    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        y_true = [id_to_label[int(idx)] for idx in labels]
        y_pred = [id_to_label[int(idx)] for idx in predictions]
        metrics = summary_metrics(y_true=y_true, y_pred=y_pred)
        return {
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
        }

    return compute_metrics


def _build_class_weights(train_samples: list[EmotionSample], labels: list[str]) -> torch.Tensor:
    counts = Counter(sample.label for sample in train_samples)
    total = float(sum(counts.values()))
    num_labels = float(len(labels))

    weights: list[float] = []
    for label in labels:
        count = float(counts.get(label, 0))
        if count <= 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_labels * count))

    tensor = torch.tensor(weights, dtype=torch.float32)
    if torch.any(tensor > 0):
        mean_non_zero = tensor[tensor > 0].mean()
        tensor = tensor / mean_non_zero
    return tensor


def _save_train_summary(
    path: Path,
    args: argparse.Namespace,
    labels: list[str],
    train_samples: list[EmotionSample],
    val_samples: list[EmotionSample],
    class_weights: dict[str, float] | None,
    eval_result: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model,
        "output_dir": args.output_dir,
        "best_dir": args.best_dir,
        "labels": labels,
        "train_distribution": label_distribution([{"label": sample.label} for sample in train_samples]),
        "val_distribution": label_distribution([{"label": sample.label} for sample in val_samples]),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "use_class_weights": not args.disable_class_weights,
        "class_weights": class_weights,
        "eval_result": eval_result,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local emotion classification model.")
    parser.add_argument("--train-file", default="data/emotion/splits/train.jsonl")
    parser.add_argument("--val-file", default="data/emotion/splits/val.jsonl")
    parser.add_argument("--model", default="distilbert-base-multilingual-cased")
    parser.add_argument("--output-dir", default="models/emotion/distilbert-v1")
    parser.add_argument("--best-dir", default="models/emotion/distilbert-v1/best")
    parser.add_argument("--summary-file", default="reports/emotion/train_summary.json")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument("--use-focal-loss", action="store_true", help="Use Focal Loss instead of CrossEntropy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    train_samples = _load_samples(Path(args.train_file))
    val_samples = _load_samples(Path(args.val_file))

    labels = sorted({sample.label for sample in train_samples + val_samples})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        label2id=label_to_id,
        id2label=id_to_label,
    )

    train_dataset = EmotionTextDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_length=args.max_length,
    )
    val_dataset = EmotionTextDataset(
        samples=val_samples,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_length=args.max_length,
    )

    class_weight_tensor: torch.Tensor | None = None
    class_weight_map: dict[str, float] | None = None
    if not args.disable_class_weights:
        class_weight_tensor = _build_class_weights(train_samples=train_samples, labels=labels)
        class_weight_map = {
            label: float(weight)
            for label, weight in zip(labels, class_weight_tensor.tolist(), strict=False)
        }

    training_kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "logging_steps": 20,
        "save_total_limit": 2,
        "report_to": "none",
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"
    if torch.cuda.is_available():
        training_kwargs["fp16"] = True

    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "compute_metrics": _compute_metrics_factory(id_to_label=id_to_label),
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedLossTrainer(
        **trainer_kwargs, 
        class_weights=class_weight_tensor,
        use_focal_loss=args.use_focal_loss
    )

    trainer.train()
    eval_result = trainer.evaluate()

    best_dir = Path(args.best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    _save_train_summary(
        path=Path(args.summary_file),
        args=args,
        labels=labels,
        train_samples=train_samples,
        val_samples=val_samples,
        class_weights=class_weight_map,
        eval_result={k: float(v) for k, v in eval_result.items()},
    )

    print("Training completed")
    print(f"  labels={labels}")
    if class_weight_map is not None:
        print(f"  class_weights={class_weight_map}")
    print(f"  best_model={best_dir}")
    print(f"  eval_result={eval_result}")


if __name__ == "__main__":
    main()
