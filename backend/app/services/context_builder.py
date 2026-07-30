from __future__ import annotations

import logging
from typing import Any

from .context_relevance_engine import ContextRelevanceEngine
from .memory_relevance_filter import FilterResult
from .prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

_MIN_SCORE: float = 0.20
_MAX_MEMORIES_IN_CONTEXT: int = 5


class ContextBuilderService:
    def __init__(self, relevance_engine: ContextRelevanceEngine | None = None) -> None:
        self._relevance = relevance_engine or ContextRelevanceEngine()

    async def build_memory_context(
        self,
        memories: list[dict[str, Any]],
        min_score: float = _MIN_SCORE,
        max_memories: int = _MAX_MEMORIES_IN_CONTEXT,
        query: str = "",
    ) -> str:
        if not memories:
            logger.debug("build_memory_context | no memories provided")
            return ""

        filter_result = await self._relevance.filter_memories(
            query=query,
            memories=memories,
        )

        filtered = self._remove_archived(filter_result.relevant)
        deduplicated = self._deduplicate(filtered)
        capped = deduplicated[:max_memories]

        if not capped:
            logger.info(
                "build_memory_context | all %d memories dropped | %d discarded by filter | %d deduplicated",
                len(memories),
                len(filter_result.discarded),
                len(filtered) - len(deduplicated),
            )
            return ""

        result = PromptBuilder.user_facts_block(capped)

        logger.info(
            "build_memory_context | total=%d | relevant=%d | discarded=%d | deduped=%d | final=%d | %.2fms",
            len(memories),
            len(filter_result.relevant),
            len(filter_result.discarded),
            len(filtered) - len(deduplicated),
            len(capped),
            filter_result.execution_time_ms,
        )
        return result

    async def debug_filter(
        self,
        query: str,
        memories: list[dict[str, Any]],
    ) -> FilterResult:
        result = await self._relevance.filter_memories(
            query=query,
            memories=memories,
        )
        return FilterResult(
            query_type="DEBUG",
            relevant_memories=result.relevant,
            discarded_memories=result.discarded,
            scores=result.scores,
            execution_time_ms=result.execution_time_ms,
        )

    @staticmethod
    def _remove_archived(
        memories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for m in memories:
            meta = m.get("metadata", {})
            if meta.get("status") == "archived":
                continue
            result.append(m)
        return result

    @staticmethod
    def _deduplicate(
        memories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for m in memories:
            key = m.get("document", "").strip().lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(m)
        return result
