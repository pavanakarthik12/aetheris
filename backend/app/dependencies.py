"""Dependency providers for FastAPI endpoints and application services."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from .config.settings import Settings, get_settings
from .services.llm_service import LLMService


def get_app_settings() -> Settings:
    """Provide the shared settings instance for dependency injection."""

    return get_settings()


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    """Provide the shared LLM service boundary."""

    return LLMService(get_settings())


@lru_cache(maxsize=1)
def get_embedding_service() -> "EmbeddingService":
    """Provide the shared embedding service boundary."""

    from .services.embedding_service import EmbeddingService

    return EmbeddingService(get_settings())


@lru_cache(maxsize=1)
def get_chroma_service() -> "ChromaService":
    """Provide the shared ChromaDB service boundary."""

    from .services.chroma_service import ChromaService

    return ChromaService(get_settings())


@lru_cache(maxsize=1)
def get_memory_service() -> "MemoryService":
    """Provide the shared MemoryService instance for dependency injection."""

    from .services.memory_service import MemoryService

    return MemoryService(
        embedding_service=get_embedding_service(),
        chroma_service=get_chroma_service(),
    )


@lru_cache(maxsize=1)
def get_context_builder_service() -> "ContextBuilderService":
    """Provide the shared ContextBuilderService instance for dependency injection."""

    from .services.context_builder import ContextBuilderService

    return ContextBuilderService()


def get_memory_evaluator_service(
    llm_service: LLMService = Depends(get_llm_service),
) -> "MemoryEvaluatorService":
    """Provide the memory evaluator wired to the shared LLM service."""

    from .services.memory_evaluator import MemoryEvaluatorService

    return MemoryEvaluatorService(llm_service=llm_service)


@lru_cache(maxsize=1)
def get_memory_evolution_service() -> "MemoryEvolutionService":
    """Provide the shared MemoryEvolutionService instance."""

    from .services.memory_evolution_service import MemoryEvolutionService

    return MemoryEvolutionService(
        memory_service=get_memory_service(),
        chroma_service=get_chroma_service(),
        embedding_service=get_embedding_service(),
    )


@lru_cache(maxsize=1)
def get_database_service() -> "DatabaseService":
    """Provide the shared SQLAlchemy database service."""

    from .services.database_service import DatabaseService

    return DatabaseService(get_settings())