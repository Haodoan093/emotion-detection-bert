from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.application.rag_use_cases import QuerySimilarChunksUseCase


class SupabaseRetriever(BaseRetriever):
    def __init__(self, query_use_case: QuerySimilarChunksUseCase, top_k: int) -> None:
        super().__init__()
        self._query_use_case = query_use_case
        self._top_k = top_k

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        chunks = self._query_use_case.execute(query=query, top_k=self._top_k)
        return [
            Document(page_content=chunk.content, metadata=chunk.metadata)
            for chunk in chunks
        ]

    async def _aget_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        return self._get_relevant_documents(query, **kwargs)

