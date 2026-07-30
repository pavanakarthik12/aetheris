from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[3]
ENV_PATH = BASE_DIR / ".env"

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

    if "CHROMA_DB_PATH" in values:
        p = Path(values["CHROMA_DB_PATH"])
        if not p.is_absolute():
            values["CHROMA_DB_PATH"] = str(BASE_DIR / p)

    return values


def _resolve(values: dict[str, str], *names: str) -> str:
    for name in names:
        v = os.getenv(name, values.get(name))
        if v:
            return v
    return ""


@dataclass(frozen=True)
class Settings:
    app_name: str = "Aetheris"
    api_v1_prefix: str = "/api/v1"
    database_url: str = ""
    secret_key: str = ""
    chroma_db_path: str = str(Path(__file__).resolve().parents[3] / "database" / "chroma")
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    log_level: str = "INFO"
    reflection_max_tokens: int = 128
    reflection_temperature: float = 0.2
    reflection_enabled: bool = True

    # Provider config
    llm_provider: str = "groq"

    # Groq
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"

    # External Knowledge
    tavily_api_key: str = ""

    # Conversation Intelligence Engine
    followup_confidence_threshold: float = 0.75
    max_context_messages: int = 4
    max_reference_distance: int = 6

    # Context Relevance Engine
    memory_relevance_threshold: float = 0.55
    conversation_relevance_threshold: float = 0.40
    search_relevance_threshold: float = 0.55
    max_memory_entries: int = 5
    max_conversation_entries: int = 3
    max_search_results: int = 5
    relevance_weights: dict[str, float] = field(default_factory=lambda: {
        "chroma": 0.45,
        "keyword": 0.20,
        "topic": 0.20,
        "recency": 0.05,
        "importance": 0.10,
        "existing_score": 0.50,
    })

    # Global LLM settings
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024
    llm_timeout: float = 30.0
    enable_circuit_breaker: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    env_file_values = _load_env_file()

    legacy_api_key = _resolve(env_file_values, "QWEN_API_KEY")
    legacy_base_url = _resolve(env_file_values, "QWEN_BASE_URL") or "https://api.groq.com/openai/v1"
    legacy_model = _resolve(env_file_values, "LLM_MODEL") or "llama-3.3-70b-versatile"

    return Settings(
        app_name=os.getenv("APP_NAME", env_file_values.get("APP_NAME", "Aetheris")),
        api_v1_prefix=os.getenv("API_V1_PREFIX", env_file_values.get("API_V1_PREFIX", "/api/v1")),
        database_url=os.getenv("DATABASE_URL", env_file_values.get("DATABASE_URL", "")),
        secret_key=os.getenv("SECRET_KEY", env_file_values.get("SECRET_KEY", "")),
        chroma_db_path=os.getenv("CHROMA_DB_PATH", env_file_values.get("CHROMA_DB_PATH", _DEFAULT_CHROMA_PATH)),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            env_file_values.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
        ),
        log_level=os.getenv("LOG_LEVEL", env_file_values.get("LOG_LEVEL", "INFO")),
        reflection_max_tokens=int(
            os.getenv("REFLECTION_MAX_TOKENS", env_file_values.get("REFLECTION_MAX_TOKENS", "128"))
        ),
        reflection_temperature=float(
            os.getenv("REFLECTION_TEMPERATURE", env_file_values.get("REFLECTION_TEMPERATURE", "0.2"))
        ),
        reflection_enabled=(
            os.getenv("REFLECTION_ENABLED", env_file_values.get("REFLECTION_ENABLED", "true")).lower()
            in ("true", "1", "yes")
        ),
        # Provider
        llm_provider=os.getenv("LLM_PROVIDER", env_file_values.get("LLM_PROVIDER", "groq")),
        # Groq (falls back to legacy QWEN_* vars for backward compat)
        groq_api_key=os.getenv("GROQ_API_KEY", env_file_values.get("GROQ_API_KEY", legacy_api_key)),
        groq_base_url=os.getenv(
            "GROQ_BASE_URL",
            env_file_values.get("GROQ_BASE_URL", legacy_base_url),
        ),
        groq_model=os.getenv("GROQ_MODEL", env_file_values.get("GROQ_MODEL", legacy_model)),
        # Global LLM settings
        llm_temperature=float(
            os.getenv("LLM_TEMPERATURE", env_file_values.get("LLM_TEMPERATURE", "0.7"))
        ),
        llm_max_tokens=int(
            os.getenv("LLM_MAX_TOKENS", env_file_values.get("LLM_MAX_TOKENS", "256"))
        ),
        tavily_api_key=os.getenv("TAVILY_API_KEY", env_file_values.get("TAVILY_API_KEY", "")),
        # Conversation Intelligence Engine
        followup_confidence_threshold=float(
            os.getenv("FOLLOWUP_CONFIDENCE_THRESHOLD", env_file_values.get("FOLLOWUP_CONFIDENCE_THRESHOLD", "0.75"))
        ),
        max_context_messages=int(
            os.getenv("MAX_CONTEXT_MESSAGES", env_file_values.get("MAX_CONTEXT_MESSAGES", "4"))
        ),
        max_reference_distance=int(
            os.getenv("MAX_REFERENCE_DISTANCE", env_file_values.get("MAX_REFERENCE_DISTANCE", "6"))
        ),
        # Context Relevance Engine
        memory_relevance_threshold=float(
            os.getenv("MEMORY_RELEVANCE_THRESHOLD", env_file_values.get("MEMORY_RELEVANCE_THRESHOLD", "0.55"))
        ),
        conversation_relevance_threshold=float(
            os.getenv("CONVERSATION_RELEVANCE_THRESHOLD", env_file_values.get("CONVERSATION_RELEVANCE_THRESHOLD", "0.65"))
        ),
        search_relevance_threshold=float(
            os.getenv("SEARCH_RELEVANCE_THRESHOLD", env_file_values.get("SEARCH_RELEVANCE_THRESHOLD", "0.55"))
        ),
        max_memory_entries=int(
            os.getenv("MAX_MEMORY_ENTRIES", env_file_values.get("MAX_MEMORY_ENTRIES", "5"))
        ),
        max_conversation_entries=int(
            os.getenv("MAX_CONVERSATION_ENTRIES", env_file_values.get("MAX_CONVERSATION_ENTRIES", "3"))
        ),
        max_search_results=int(
            os.getenv("MAX_SEARCH_RESULTS", env_file_values.get("MAX_SEARCH_RESULTS", "5"))
        ),
        relevance_weights={
            "chroma": float(os.getenv("RELEVANCE_WEIGHT_CHROMA", env_file_values.get("RELEVANCE_WEIGHT_CHROMA", "0.45"))),
            "keyword": float(os.getenv("RELEVANCE_WEIGHT_KEYWORD", env_file_values.get("RELEVANCE_WEIGHT_KEYWORD", "0.20"))),
            "topic": float(os.getenv("RELEVANCE_WEIGHT_TOPIC", env_file_values.get("RELEVANCE_WEIGHT_TOPIC", "0.20"))),
            "recency": float(os.getenv("RELEVANCE_WEIGHT_RECENCY", env_file_values.get("RELEVANCE_WEIGHT_RECENCY", "0.05"))),
            "importance": float(os.getenv("RELEVANCE_WEIGHT_IMPORTANCE", env_file_values.get("RELEVANCE_WEIGHT_IMPORTANCE", "0.10"))),
            "existing_score": float(os.getenv("RELEVANCE_WEIGHT_EXISTING", env_file_values.get("RELEVANCE_WEIGHT_EXISTING", "0.50"))),
        },
        llm_timeout=float(
            os.getenv("LLM_TIMEOUT", env_file_values.get("LLM_TIMEOUT", "30"))
        ),
        enable_circuit_breaker=(
            os.getenv("ENABLE_CIRCUIT_BREAKER", env_file_values.get("ENABLE_CIRCUIT_BREAKER", "true")).lower()
            in ("true", "1", "yes")
        ),
    )
