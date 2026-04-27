from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import argparse
import csv
import json
from typing import Any

from datasets import load_dataset
import kagglehub
from kagglehub import KaggleDatasetAdapter

from app.application.rag_use_cases import IngestDocumentsUseCase
from app.core.config import settings
from app.infrastructure.sentence_transformer_embeddings import SentenceTransformerEmbeddingService
from app.infrastructure.supabase_vector_store import SupabaseVectorStore


def record_to_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in record.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key}: {value}")
        else:
            try:
                parts.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
            except TypeError:
                parts.append(f"{key}: {value}")
    return "\n".join(parts)


def load_financebench(limit: int | None) -> list[str]:
    dataset = load_dataset("PatronusAI/financebench")
    splits = []
    for split in dataset.values():
        splits.extend(split)
    documents: list[str] = []
    for record in splits:
        documents.append(record_to_text(record))
        if limit is not None and len(documents) >= limit:
            break
    return documents


def load_kaggle_stock_data(limit: int | None) -> list[str]:
    path = Path(kagglehub.dataset_download("ngwkhai/vietnamese-stock-market-data"))
    documents: list[str] = []
    for csv_path in path.rglob("*.csv"):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                documents.append(record_to_text(row))
                if limit is not None and len(documents) >= limit:
                    return documents
    return documents


def load_kaggle_vifid_segraw(limit: int | None) -> list[str]:
    dataset_name = "daddychillonkaggle/vifid-segraw-200k"
    documents: list[str] = []

    # Prefer KaggleDatasetAdapter.PANDAS when available, then fallback to file iteration.
    try:
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            dataset_name,
            "",
        )
        for row in df.to_dict(orient="records"):
            documents.append(record_to_text(row))
            if limit is not None and len(documents) >= limit:
                return documents
        if documents:
            return documents
    except Exception:
        # Some environments require an explicit file_path for the adapter.
        pass

    path = Path(kagglehub.dataset_download(dataset_name))
    for csv_path in path.rglob("*.csv"):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                documents.append(record_to_text(row))
                if limit is not None and len(documents) >= limit:
                    return documents

    for jsonl_path in path.rglob("*.jsonl"):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                documents.append(record_to_text(payload if isinstance(payload, dict) else {"value": payload}))
                if limit is not None and len(documents) >= limit:
                    return documents

    for json_path in path.rglob("*.json"):
        with json_path.open("r", encoding="utf-8") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError:
                continue

            if isinstance(payload, list):
                for item in payload:
                    record = item if isinstance(item, dict) else {"value": item}
                    documents.append(record_to_text(record))
                    if limit is not None and len(documents) >= limit:
                        return documents
            elif isinstance(payload, dict):
                documents.append(record_to_text(payload))
                if limit is not None and len(documents) >= limit:
                    return documents

    if not documents:
        raise RuntimeError(
            "No readable records found for daddychillonkaggle/vifid-segraw-200k. "
            "Try installing kagglehub[pandas-datasets] or provide supported CSV/JSON files."
        )
    return documents


def build_use_case() -> IngestDocumentsUseCase:
    if not settings.supabase_url or not settings.supabase_service_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required.")
    embedder = SentenceTransformerEmbeddingService(
        model_name=settings.embedding_model_name,
        batch_size=settings.embedding_batch_size,
    )
    vector_store = SupabaseVectorStore(
        url=settings.supabase_url,
        key=settings.supabase_service_key,
        table=settings.supabase_table,
        match_rpc=settings.supabase_match_rpc,
    )
    return IngestDocumentsUseCase(
        embedding_service=embedder,
        vector_store=vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest datasets into Supabase vector store.")
    parser.add_argument(
        "--source",
        choices=["financebench", "vietnam-stock", "vifid-segraw-200k"],
        required=True,
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    use_case = build_use_case()
    if args.source == "financebench":
        documents = load_financebench(args.limit)
        source_name = "financebench"
    elif args.source == "vifid-segraw-200k":
        documents = load_kaggle_vifid_segraw(args.limit)
        source_name = "vifid-segraw-200k"
    else:
        documents = load_kaggle_stock_data(args.limit)
        source_name = "vietnam-stock"

    count = use_case.execute(documents=documents, source=source_name)
    print(f"Ingested {count} chunks from {source_name}.")


if __name__ == "__main__":
    main()
