from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_context_builder_service, get_memory_service
from ..services.context_builder import ContextBuilderService
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
    results = memory_service.search_memories(query=query, limit=limit)
    memories = results.get("memories", []) if isinstance(results, dict) else []

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
