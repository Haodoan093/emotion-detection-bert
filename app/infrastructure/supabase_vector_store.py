from __future__ import annotations

from typing import Any

from supabase import Client, create_client

from app.domain.entities import DocumentChunk
from app.domain.ports import VectorStore


class SupabaseVectorStore(VectorStore):
    def __init__(
        self,
        url: str,
        key: str,
        table: str = "document_chunks",
        match_rpc: str = "match_documents",
    ) -> None:
        self._client: Client = create_client(url, key)
        self._table = table
        self._match_rpc = match_rpc

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        rows = [
            {
                "content": chunk.content,
                "embedding": chunk.embedding,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]
        response = self._client.table(self._table).upsert(rows).execute()
        if hasattr(response, "error") and response.error:
            raise RuntimeError(f"Supabase upsert failed: {response.error}")

    def query_similar(self, query_embedding: list[float], top_k: int) -> list[DocumentChunk]:
        payload: dict[str, Any] = {
            "query_embedding": query_embedding,
            "match_count": top_k,
        }
        response = self._client.rpc(self._match_rpc, payload).execute()
        if hasattr(response, "error") and response.error:
            raise RuntimeError(f"Supabase query failed: {response.error}")
        data = getattr(response, "data", None) or []
        return [
            DocumentChunk(
                content=item.get("content", ""),
                embedding=item.get("embedding", []),
                metadata=item.get("metadata", {}),
            )
            for item in data
        ]

