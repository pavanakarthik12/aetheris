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


def get_reflection_service(
    llm_service: LLMService = Depends(get_llm_service),
    memory_service: "MemoryService" = Depends(get_memory_service),
    chroma_service: "ChromaService" = Depends(get_chroma_service),
    embedding_service: "EmbeddingService" = Depends(get_embedding_service),
) -> "ReflectionService":
    """Provide a ReflectionService wired to the shared LLM, memory, and chroma services."""

    from .services.reflection_service import ReflectionService

    return ReflectionService(
        llm_service=llm_service,
        memory_service=memory_service,
        chroma_service=chroma_service,
        embedding_service=embedding_service,
    )


def get_immediate_memory_processor(
    memory_service: "MemoryService" = Depends(get_memory_service),
    memory_evaluator: "MemoryEvaluatorService" = Depends(get_memory_evaluator_service),
    memory_evolution: "MemoryEvolutionService" = Depends(get_memory_evolution_service),
    chroma_service: "ChromaService" = Depends(get_chroma_service),
    embedding_service: "EmbeddingService" = Depends(get_embedding_service),
) -> "ImmediateMemoryProcessor":
    """Provide an ImmediateMemoryProcessor wired to shared services."""

    from .services.immediate_memory_processor import ImmediateMemoryProcessor

    return ImmediateMemoryProcessor(
        memory_service=memory_service,
        memory_evaluator=memory_evaluator,
        memory_evolution=memory_evolution,
        chroma_service=chroma_service,
        embedding_service=embedding_service,
    )


def get_intent_classifier(
    llm_service: "LLMService" = Depends(get_llm_service),
) -> "IntentClassifier":
    """Provide the combined intent classifier with rule + LLM fallback."""

    from .services.intent_classifier import IntentClassifier

    return IntentClassifier(llm_service=llm_service)


def get_conversation_memory() -> "ConversationMemory":
    """Provide the shared in-memory conversation store."""

    from .services.conversation_memory import ConversationMemory

    return ConversationMemory()


def get_system_memory() -> "SystemMemory":
    """Provide the shared system memory store."""

    from .services.system_memory import SystemMemory

    return SystemMemory()


def get_memory_hierarchy_service(
    conversation_memory: "ConversationMemory" = Depends(get_conversation_memory),
    memory_service: "MemoryService" = Depends(get_memory_service),
    system_memory: "SystemMemory" = Depends(get_system_memory),
) -> "MemoryHierarchyService":
    """Provide the shared MemoryHierarchyService."""

    from .services.memory_hierarchy_service import MemoryHierarchyService

    return MemoryHierarchyService(
        conversation_memory=conversation_memory,
        long_term_memory=memory_service,
        system_memory=system_memory,
    )


def get_request_router(
    llm_service: "LLMService" = Depends(get_llm_service),
    memory_service: "MemoryService" = Depends(get_memory_service),
    memory_evaluator: "MemoryEvaluatorService" = Depends(get_memory_evaluator_service),
    memory_evolution: "MemoryEvolutionService" = Depends(get_memory_evolution_service),
    chroma_service: "ChromaService" = Depends(get_chroma_service),
    embedding_service: "EmbeddingService" = Depends(get_embedding_service),
    context_builder: "ContextBuilderService" = Depends(get_context_builder_service),
    reflection_service: "ReflectionService" = Depends(get_reflection_service),
    intent_classifier: "IntentClassifier" = Depends(get_intent_classifier),
    immediate_memory_processor: "ImmediateMemoryProcessor" = Depends(get_immediate_memory_processor),
    memory_hierarchy: "MemoryHierarchyService" = Depends(get_memory_hierarchy_service),
) -> "CognitiveRequestRouter":
    """Provide the CognitiveRequestRouter wired to all subsystems."""

    from .services.request_router import CognitiveRequestRouter

    return CognitiveRequestRouter(
        llm_service=llm_service,
        memory_service=memory_service,
        memory_evaluator=memory_evaluator,
        memory_evolution=memory_evolution,
        chroma_service=chroma_service,
        embedding_service=embedding_service,
        context_builder=context_builder,
        reflection_service=reflection_service,
        intent_classifier=intent_classifier,
        immediate_memory_processor=immediate_memory_processor,
        memory_hierarchy=memory_hierarchy,
    )