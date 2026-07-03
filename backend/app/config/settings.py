"""Centralized application settings loaded from the repository .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[3]
ENV_PATH = BASE_DIR / ".env"

# Absolute default paths — never depend on the CWD at runtime.
_DEFAULT_CHROMA_PATH = str(BASE_DIR / "database" / "chroma")


def _load_env_file() -> dict[str, str]:
    values: dict[str, str] = {}

    if not ENV_PATH.exists():
        return values

    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")

    # If CHROMA_DB_PATH is a relative path in the .env file, resolve it
    # against BASE_DIR so the database always lands in the same place
    # regardless of where the server process is launched from.
    if "CHROMA_DB_PATH" in values:
        p = Path(values["CHROMA_DB_PATH"])
        if not p.is_absolute():
            values["CHROMA_DB_PATH"] = str(BASE_DIR / p)

    return values


@dataclass(frozen=True)
class Settings:
    """Application configuration backed by environment variables."""

    app_name: str = "Aetheris"
    api_v1_prefix: str = "/api/v1"
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    database_url: str = ""
    secret_key: str = ""
    chroma_db_path: str = str(Path(__file__).resolve().parents[3] / "database" / "chroma")
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    llm_provider: str = "qwen"
    llm_model: str = "qwen-3.7-plus"
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings instance."""

    env_file_values = _load_env_file()

    return Settings(
        app_name=os.getenv("APP_NAME", env_file_values.get("APP_NAME", "Aetheris")),
        api_v1_prefix=os.getenv("API_V1_PREFIX", env_file_values.get("API_V1_PREFIX", "/api/v1")),
        qwen_api_key=os.getenv("QWEN_API_KEY", env_file_values.get("QWEN_API_KEY", "")),
        qwen_base_url=os.getenv(
            "QWEN_BASE_URL",
            env_file_values.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        ),
        database_url=os.getenv("DATABASE_URL", env_file_values.get("DATABASE_URL", "")),
        secret_key=os.getenv("SECRET_KEY", env_file_values.get("SECRET_KEY", "")),
        chroma_db_path=os.getenv("CHROMA_DB_PATH", env_file_values.get("CHROMA_DB_PATH", _DEFAULT_CHROMA_PATH)),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            env_file_values.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
        ),
        llm_provider=os.getenv("LLM_PROVIDER", env_file_values.get("LLM_PROVIDER", "qwen")),
        llm_model=os.getenv("LLM_MODEL", env_file_values.get("LLM_MODEL", "qwen-3.7-plus")),
        log_level=os.getenv("LOG_LEVEL", env_file_values.get("LOG_LEVEL", "INFO")),
    )