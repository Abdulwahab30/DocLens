from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str
    redis_url: str = "redis://localhost:6379/0"

    jwt_secret: str
    jwt_lifetime_seconds: int = 60 * 60 * 24  # 1 day

    openrouter_api_key: str
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_vision_model: str = "google/gemini-2.0-flash-exp:free"

    # --- MinIO / S3-compatible object storage (raw uploaded files) ---
    minio_endpoint_url: str = "http://localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "documents"

    # --- Qdrant (vector store for chunk embeddings) ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "document_chunks"

    # --- Local embedding model (sentence-transformers) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384


@lru_cache
def get_settings() -> Settings:
    return Settings()
