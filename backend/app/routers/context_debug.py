from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_conversation_memory, get_context_builder_service, get_llm_service, get_memory_service
from ..services.context_builder import ContextBuilderService
from ..services.conversation_memory import ConversationMemory
from ..services.llm_service import LLMService
from ..services.memory_service import MemoryService

router = APIRouter(prefix="/api/context", tags=["context"])
logger = logging.getLogger(__name__)


@router.get("/debug")
async def debug_filter(
    query: str = Query(..., min_length=1, max_length=500),
    limit: int = Query(default=20, ge=1, le=100),
    context_builder: ContextBuilderService = Depends(get_context_builder_service),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """Run the Memory Relevance Filter on live memories for a given query."""
    memories = await memory_service.search_memory(query=query, top_k=limit)

    filter_result = context_builder.debug_filter(query=query, memories=memories)

    return {
        "query": query,
        "query_type": filter_result.query_type.value,
        "relevant_count": len(filter_result.relevant_memories),
        "discarded_count": len(filter_result.discarded_memories),
        "execution_time_ms": filter_result.execution_time_ms,
        "relevant": [
            {
                "id": m.get("id"),
                "document": m.get("document", "")[:200],
                "score": m.get("_relevance_score", 0.0),
                "original_score": m.get("score", 0.0),
            }
            for m in filter_result.relevant_memories
        ],
        "discarded": [
            {
                "id": m.get("id"),
                "document": m.get("document", "")[:200],
                "original_score": m.get("score", 0.0),
            }
            for m in filter_result.discarded_memories
        ],
        "scores": filter_result.scores,
    }


@router.get("/conversation")
async def debug_conversation(
    query: str = Query(..., min_length=1, max_length=500),
    conversation_memory: ConversationMemory = Depends(get_conversation_memory),
):
    """Show conversation history with relevance filtering for a given query."""
    from ..services.conversation_context_filter import filter_conversation

    history = conversation_memory.history
    filter_result = filter_conversation(query, history)

    return {
        "query": query,
        "query_type": filter_result.query_type.value,
        "total_before": filter_result.total_before,
        "total_after": filter_result.total_after,
        "discarded_count": len(filter_result.discarded),
        "execution_time_ms": filter_result.execution_time_ms,
        "filtered": [
            {
                "role": m.get("role"),
                "content": m.get("content", "")[:300],
            }
            for m in filter_result.filtered_history
        ],
        "discarded": [
            {
                "role": m.get("role"),
                "content": m.get("content", "")[:300],
                "reason": "below relevance threshold",
            }
            for m in filter_result.discarded
        ],
    }
