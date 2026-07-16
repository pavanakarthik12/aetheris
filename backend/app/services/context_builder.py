from __future__ import annotations

import logging
from typing import Any

from .memory_relevance_filter import (
    FilterResult,
    QueryType,
    filter_memories,
)
from .prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

_MIN_SCORE: float = 0.20
_MAX_MEMORIES_IN_CONTEXT: int = 5


class ContextBuilderService:
    def build_memory_context(
        self,
        memories: list[dict[str, Any]],
        min_score: float = _MIN_SCORE,
        max_memories: int = _MAX_MEMORIES_IN_CONTEXT,
        query: str = "",
    ) -> str:
        if not memories:
            logger.debug("build_memory_context | no memories provided")
            return ""

        filter_result = filter_memories(
            query=query,
            memories=memories,
            threshold=min_score,
            max_memories=max_memories,
        )

        if filter_result.query_type in (QueryType.MATH, QueryType.GREETING):
            logger.debug(
                "build_memory_context | query_type=%s — skipping all memories",
                filter_result.query_type.value,
            )
            return ""

        filtered = self._remove_archived(filter_result.relevant_memories)
        deduplicated = self._deduplicate(filtered)
        capped = deduplicated[:max_memories]

        if not capped:
            logger.info(
                "build_memory_context | query_type=%s | all %d memories dropped | %d discarded by filter | %d deduplicated",
                filter_result.query_type.value,
                len(memories),
                len(filter_result.discarded_memories),
                len(filtered) - len(deduplicated),
            )
            return ""

        result = PromptBuilder.user_facts_block(capped)

        logger.info(
            "build_memory_context | query_type=%s | total=%d | relevant=%d | discarded=%d | deduped=%d | final=%d | %.2fms",
            filter_result.query_type.value,
            len(memories),
            len(filter_result.relevant_memories),
            len(filter_result.discarded_memories),
            len(filtered) - len(deduplicated),
            len(capped),
            filter_result.execution_time_ms,
        )
        return result

    def debug_filter(
        self,
        query: str,
        memories: list[dict[str, Any]],
    ) -> FilterResult:
        return filter_memories(
            query=query,
            memories=memories,
            threshold=_MIN_SCORE,
            max_memories=_MAX_MEMORIES_IN_CONTEXT,
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
