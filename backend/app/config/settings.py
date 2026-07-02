"""Centralized application settings loaded from the repository .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application configuration backed by environment variables."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Aetheris"
    api_v1_prefix: str = "/api/v1"
    qwen_api_key: str = Field(default="", alias="QWEN_API_KEY")
    database_url: str = Field(default="", alias="DATABASE_URL")
    secret_key: str = Field(default="", alias="SECRET_KEY")
    chroma_db_path: str = Field(default="./database/chroma", alias="CHROMA_DB_PATH")
    embedding_model: str = Field(default="BAAI/bge-base-en-v1.5", alias="EMBEDDING_MODEL")
    llm_provider: str = Field(default="qwen", alias="LLM_PROVIDER")
    llm_model: str = Field(default="qwen-3.7-plus", alias="LLM_MODEL")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings instance."""

    return Settings()