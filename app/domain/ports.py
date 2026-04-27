from typing import Protocol

from app.domain.entities import DocumentChunk, EmotionResult


class ASRService(Protocol):
    def load_model(self) -> None:
        ...

    def transcribe(self, audio_path: str) -> str:
        ...


class LLMService(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class EmbeddingService(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class VectorStore(Protocol):
    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        ...

    def query_similar(self, query_embedding: list[float], top_k: int) -> list[DocumentChunk]:
        ...


class EmotionService(Protocol):
    def load_model(self) -> None:
        ...

    def detect(self, text: str, top_k: int = 3) -> EmotionResult:
        ...


