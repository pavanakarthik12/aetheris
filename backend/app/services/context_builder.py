"""Context Builder — formats retrieved memories into a clean prompt block."""

from __future__ import annotations

import logging
from typing import Any

from .prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

_MIN_SCORE: float = 0.20
_MAX_MEMORIES_IN_CONTEXT: int = 5


class ContextBuilderService:
    """Format retrieved memory records into a structured LLM context block."""

    def build_memory_context(
        self,
        memories: list[dict[str, Any]],
        min_score: float = _MIN_SCORE,
        max_memories: int = _MAX_MEMORIES_IN_CONTEXT,
    ) -> str:
        if not memories:
            logger.debug("build_memory_context | no memories provided")
            return ""

        filtered = self._remove_archived(memories)
        filtered = self._filter(filtered, min_score)
        deduplicated = self._deduplicate(filtered)
        capped = deduplicated[:max_memories]

        if not capped:
            logger.debug(
                "build_memory_context | all %d memories dropped by filter/dedup",
                len(memories),
            )
            return ""

        result = PromptBuilder.memory_context_block(capped)

        logger.info(
            "Context block built | total_input=%d | surviving=%d",
            len(memories),
            len(capped),
        )
        return result

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
    def _filter(
        memories: list[dict[str, Any]],
        min_score: float,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for m in memories:
            doc = m.get("document", "")
            score = m.get("score", 0.0)
            if not isinstance(doc, str) or not doc.strip():
                continue
            if score < min_score:
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
