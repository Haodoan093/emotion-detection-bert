from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app.domain.ports import EmbeddingService


class SentenceTransformerEmbeddingService(EmbeddingService):
    def __init__(self, model_name: str, batch_size: int = 32, normalize: bool = True) -> None:
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._normalize = normalize

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
        )
        return [embedding.tolist() for embedding in embeddings]

