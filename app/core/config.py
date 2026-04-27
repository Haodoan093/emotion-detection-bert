from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
    )
    app_title: str = "Financial Advice Chatbot API"
    app_version: str = "0.1.0"
    docs_url: str = "/docs"
    openapi_url: str = "/openapi.json"
    asr_model_name: str = "turbo"
    asr_device: str = "auto"
    asr_use_fp16: bool | None = None
    asr_language: str | None = "vi"
    asr_initial_prompt: str | None = None
    llm_provider: str = "gemini"
    openai_api_key: str | None = None
    openai_model_name: str = "gpt-oss-120b"
    gemini_api_key: str | None = None
    gemini_model_name: str = "gemini-1.5-flash"
    local_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    local_max_new_tokens: int = 256
    emotion_enabled: bool = True
    emotion_model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    emotion_device: str = "auto"
    rag_enabled: bool = True
    supabase_url: str | None = None
    supabase_service_key: str | None = None
    supabase_table: str = "document_chunks"
    supabase_match_rpc: str = "match_documents"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    chunk_size: int = 300
    chunk_overlap: int = 50
    rag_top_k: int = 4


settings = Settings()
