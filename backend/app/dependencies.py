"""Dependency providers for FastAPI endpoints and application services."""

from functools import lru_cache

from app.config.settings import Settings, get_settings
from app.services.chroma_service import ChromaService
from app.services.database_service import DatabaseService
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService


def get_app_settings() -> Settings:
    """Provide the shared settings instance for dependency injection."""

    return get_settings()


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    """Provide the shared LLM service boundary."""

    return LLMService(get_settings())


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Provide the shared embedding service boundary."""

    return EmbeddingService(get_settings())


@lru_cache(maxsize=1)
def get_chroma_service() -> ChromaService:
    """Provide the shared ChromaDB service boundary."""

    return ChromaService(get_settings())


@lru_cache(maxsize=1)
def get_database_service() -> DatabaseService:
    """Provide the shared SQLAlchemy database service."""

    return DatabaseService(get_settings())