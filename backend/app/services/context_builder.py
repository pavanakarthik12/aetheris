"""Context Builder — formats retrieved memories into a clean prompt block.

This service is the only place in Aetheris that knows how to turn a list
of raw ChromaDB results into structured text that the LLM can consume.
It is intentionally stateless: every method is pure and side-effect free.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Minimum similarity score a memory must reach to be included in context.
# Cosine similarity floor — 0.20 is permissive enough for short queries
# while still excluding clearly unrelated memories.
_MIN_SCORE: float = 0.20

# Hard cap on how many memory lines appear in a single context block.
# Keeps prompts from ballooning even if top_k is set high.
_MAX_MEMORIES_IN_CONTEXT: int = 5


class ContextBuilderService:
    """Format retrieved memory records into a structured LLM context block.

    All public methods are stateless and can be called from any async or
    sync context without locks or shared mutable state.
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_memory_context(
        self,
        memories: list[dict[str, Any]],
        min_score: float = _MIN_SCORE,
        max_memories: int = _MAX_MEMORIES_IN_CONTEXT,
    ) -> str:
        """Convert a list of memory search results into a formatted context block.

        Steps applied in order:
        1. Drop entries with empty or whitespace-only ``document`` fields.
        2. Drop entries whose ``score`` is below *min_score*.
        3. Deduplicate by normalised document text (case-insensitive, stripped).
        4. Cap the list at *max_memories*.
        5. Render the surviving entries as a bullet list under a header.

        Args:
            memories:    Raw results from ``MemoryService.search_memory()``.
                         Each dict must contain ``document`` and ``score``.
            min_score:   Cosine similarity floor (0–1).  Defaults to 0.20.
            max_memories: Maximum bullet points to include.  Defaults to 5.

        Returns:
            A formatted multi-line string ready for prompt injection, or an
            empty string ``""`` when no memories survive filtering.
        """
        if not memories:
            logger.debug("build_memory_context | no memories provided")
            return ""

        filtered = self._filter(memories, min_score)
        deduplicated = self._deduplicate(filtered)
        capped = deduplicated[:max_memories]

        if not capped:
            logger.debug(
                "build_memory_context | all %d memories dropped by filter/dedup",
                len(memories),
            )
            return ""

        lines = [f"- {entry['document'].strip()}" for entry in capped]
        context_block = "Relevant Memories:\n" + "\n".join(lines)

        logger.info(
            "Context block built | total_input=%d | surviving=%d",
            len(memories),
            len(capped),
        )
        return context_block

    def build_system_prompt(self) -> str:
        """Return the base system identity prompt for Aetheris.

        This is the fixed preamble that appears before any memories or user
        message.  It must remain minimal and phase-appropriate — no
        personality, emotion, or reflection language.

        Returns:
            A single-paragraph system prompt string.
        """
        return (
            "You are Aetheris, a cognitive AI assistant.\n"
            "You have access to relevant memories from past interactions.\n"
            "Use them to provide accurate, context-aware responses.\n"
            "If no memories are provided, answer from your general knowledge."
        )

    def assemble_prompt(
        self,
        user_message: str,
        memory_context: str,
    ) -> str:
        """Assemble the final prompt string sent to the LLM.

        Structure:
            <system prompt>

            <memory context block>       ← omitted when empty

            Current User Message:
            <user_message>

        Args:
            user_message:   The raw message from the user.
            memory_context: Output of ``build_memory_context()``.
                            Pass ``""`` when there are no memories.

        Returns:
            The complete, ready-to-send prompt string.
        """
        parts: list[str] = [self.build_system_prompt()]

        if memory_context.strip():
            parts.append(memory_context.strip())

        parts.append(f"Current User Message:\n{user_message.strip()}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter(
        memories: list[dict[str, Any]],
        min_score: float,
    ) -> list[dict[str, Any]]:
        """Drop memories with empty text or insufficient similarity score."""
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
        """Remove entries whose normalised text has already been seen.

        Comparison is case-insensitive and strips leading/trailing whitespace.
        Order is preserved — first occurrence wins.
        """
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for m in memories:
            key = m.get("document", "").strip().lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(m)
        return result
