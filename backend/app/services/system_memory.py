from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_SYSTEM_FACTS: list[dict[str, Any]] = [
    {"id": "identity_name", "attribute": "name", "value": "Aetheris"},
    {"id": "identity_creator", "attribute": "creator", "value": "Pavan"},
    {"id": "identity_purpose", "attribute": "purpose", "value": "An AI assistant with hierarchical memory, reflection, and persistent user knowledge."},
    {"id": "capability_memory", "attribute": "capability", "value": "Remembers user facts across sessions using long-term memory."},
    {"id": "capability_conversation", "attribute": "capability", "value": "Maintains short-term conversation memory for the current session."},
    {"id": "capability_reflection", "attribute": "capability", "value": "Reflects on interactions to strengthen important memories."},
    {"id": "capability_tools", "attribute": "capability", "value": "Can perform web searches when requested."},
    {"id": "version", "attribute": "version", "value": "0.1.0 (Phase 9 — Hierarchical Memory)"},
    {"id": "model", "attribute": "model", "value": "llama-3.3-70b-versatile (via Groq)"},
    {"id": "personality", "attribute": "personality", "value": "Helpful, knowledgeable, and context-aware."},
]


class SystemMemory:
    def __init__(self) -> None:
        self._facts: list[dict[str, Any]] = list(_SYSTEM_FACTS)

    def get_all(self) -> list[dict[str, Any]]:
        return list(self._facts)

    def search(self, query: str) -> list[dict[str, Any]]:
        query_lower = query.lower()
        results: list[dict[str, Any]] = []
        for fact in self._facts:
            attribute = (fact.get("attribute") or "").lower()
            value = (fact.get("value") or "").lower()
            if any(word in attribute or word in value for word in query_lower.split() if len(word) > 2):
                results.append(fact)
        if not results:
            results = list(self._facts)
        return results

    def to_context_block(self) -> str:
        lines: list[str] = []
        for fact in self._facts:
            attr = fact.get("attribute", "")
            value = fact.get("value", "")
            if attr and value:
                lines.append(f"- {attr}: {value}")
        if not lines:
            return ""
        return "System Information:\n" + "\n".join(lines)
