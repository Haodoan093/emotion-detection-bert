from __future__ import annotations

from typing import Any, TYPE_CHECKING
from uuid import uuid4

from langchain_core.documents import Document

if TYPE_CHECKING:
    from langchain_classic.chains import RetrievalQA

from app.domain.entities import Citation, DocumentChunk, RagResult
from app.domain.ports import EmbeddingService, VectorStore
from app.utils.chunking import chunk_text
from app.prompts.financial_prompts import build_emotion_block


class IngestDocumentsUseCase:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def execute(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        source: str | None = None,
    ) -> int:
        if metadatas is not None and len(metadatas) != len(documents):
            raise ValueError("metadatas length must match documents length.")
        chunks_text: list[str] = []
        chunks_meta: list[dict[str, Any]] = []
        for index, document in enumerate(documents):
            base_meta: dict[str, Any] = {
                "doc_index": index,
                "doc_id": str(uuid4()),
            }
            if source:
                base_meta["source"] = source
            if metadatas is not None:
                base_meta.update(metadatas[index])
            for chunk_index, chunk in enumerate(
                chunk_text(document, self._chunk_size, self._chunk_overlap)
            ):
                meta = dict(base_meta)
                meta["chunk_index"] = chunk_index
                chunks_text.append(chunk)
                chunks_meta.append(meta)
        if not chunks_text:
            return 0
        embeddings = self._embedding_service.embed_texts(chunks_text)
        chunks = [
            DocumentChunk(content=text, embedding=embedding, metadata=meta)
            for text, embedding, meta in zip(chunks_text, embeddings, chunks_meta, strict=False)
        ]
        self._vector_store.upsert_chunks(chunks)
        return len(chunks)


class QuerySimilarChunksUseCase:
    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def execute(self, query: str, top_k: int) -> list[DocumentChunk]:
        embeddings = self._embedding_service.embed_texts([query])
        if not embeddings:
            return []
        return self._vector_store.query_similar(embeddings[0], top_k)


class GenerateRagAnswerUseCase:
    def __init__(self, qa_chain: Any) -> None:
        self._qa_chain = qa_chain

    def execute(self, question: str, emotion: str | None = None, emotion_score: float = 1.0) -> RagResult:
        emotion_block = build_emotion_block(emotion or "neutral", emotion_score)
        enhanced_question = (
            f"[THONG TIN CAM XUC GOI Y]\n{emotion_block.strip()}\n\n"
            f"[CAU HOI CUA KHACH HANG]\n{question.strip()}"
        )
        inputs = {
            "query": enhanced_question,
        }
        result = self._qa_chain.invoke(inputs)
        answer = str(result.get("result", "")).strip()
        documents: list[Document] = result.get("source_documents", []) or []
        citations = [
            Citation(
                source=doc.metadata.get("source"),
                doc_id=doc.metadata.get("doc_id"),
                chunk_index=doc.metadata.get("chunk_index"),
            )
            for doc in documents
        ]
        return RagResult(answer=answer, citations=citations)


