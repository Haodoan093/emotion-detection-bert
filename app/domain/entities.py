from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AsrResult:
    text: str


@dataclass(frozen=True)
class LlmResult:
    answer: str


@dataclass(frozen=True)
class EmotionPrediction:
    label: str
    score: float


@dataclass(frozen=True)
class EmotionResult:
    text: str
    label: str
    score: float
    predictions: list[EmotionPrediction]


@dataclass(frozen=True)
class Citation:
    source: str | None
    doc_id: str | None
    chunk_index: int | None


@dataclass(frozen=True)
class RagResult:
    answer: str
    citations: list[Citation]


@dataclass(frozen=True)
class ChatVoiceResult:
    text: str
    answer: str
    citations: list[Citation]
    emotion: EmotionResult | None = None


@dataclass(frozen=True)
class DocumentChunk:
    content: str
    embedding: list[float]
    metadata: dict[str, Any]
