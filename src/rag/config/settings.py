from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "RAG MVP"
    app_version: str = "0.1.0"
    api_prefix: str = "/api"

    data_dir: Path = Field(default=Path("data"))
    chroma_dir: Path = Field(default=Path("data/chroma"))
    chroma_collection_name: str = "documents"

    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 5
    embedding_dimension: int = 256

    allowed_extensions: tuple[str, ...] = (".md", ".txt")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return settings
