"""Centralized prompt template repository for all LLM calls.

Every prompt in Aetheris originates from this single service.
Templates are as short as possible while preserving behavior.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT: str = (
    "You are Aetheris. Use relevant memories for context. "
    "Answer from general knowledge otherwise."
)

_IDENTITY_BLOCK: str = (
    "You are Aetheris. You are an AI assistant. "
    "You are not the user. User memories describe the user. "
    "Never interpret user memories as your own experiences. "
    "Never claim you performed actions that belong to the user. "
    "Never claim emotions, fatigue, hunger, sleep, or physical experiences "
    "unless explicitly simulating them for a requested roleplay."
)

_INTENT_PROMPT: str = (
    'Classify intent. Return {"intent":"<TYPE>","confidence":0.0}'
    ' where TYPE is one of: NORMAL_CHAT, CREATE_MEMORY, UPDATE_MEMORY,'
    " DELETE_MEMORY, MERGE_MEMORY, SEARCH_MEMORY, CONVERSATION_QUERY,"
    " WEB_SEARCH, SYSTEM_QUERY,"
    " MULTI_ACTION, UNKNOWN.\n"
    "If unsure, prefer NORMAL_CHAT."
)

_MEMORY_EVAL_SYSTEM: str = (
    "Decide if this message should be a permanent memory."
    ' Return {"store":true/false,"importance":0.0,"category":"<cat>","reason":"<text>",'
    '"memory_fact":"<fact>","attribute":"<attr>","value":"<val>"}.'
    " Categories: Preference,Project,Goal,Skill,Relationship,Fact,"
    "Achievement,Event,Task,Other.\n"
    "memory_fact: A third-person sentence describing the user fact."
    " Never use first person (I, my, me)."
    ' Example: "The user is building a project called Aetheris."'
    "attribute: A short machine-readable key like 'current_project'"
    " or 'favorite_programming_language'.\n"
    "value: The value of the attribute, e.g., 'Aetheris' or 'Python'.\n"
    "Store: long-term goals, preferences, projects, skills, facts,"
    " relationships, achievements, events, tasks.\n"
    "Skip: greetings, small talk, one-off queries, filler."
)

_REFLECTION_PROMPT: str = (
    'Return JSON {"action":"NO_ACTION","confidence":0.9}. '
    "Actions: NO_ACTION (no change needed), STRENGTHEN_MEMORY "
    "(interaction reinforces an existing memory). "
    "Only STRENGTHEN_MEMORY if new meaningful information is shared."
)


class PromptBuilder:
    """Single source of truth for all LLM prompts."""

    @staticmethod
    def chat_system() -> str:
        """Return the complete system prompt with identity block + instructions."""
        return f"{_IDENTITY_BLOCK}\n\n{_SYSTEM_PROMPT}"

    @staticmethod
    def identity_block() -> str:
        """Return the permanent identity block for the system prompt."""
        return _IDENTITY_BLOCK

    @staticmethod
    def intent_classification(message: str) -> str:
        return f"{_INTENT_PROMPT}\nUser message:\n{message.strip()}"

    @staticmethod
    def memory_evaluation_system() -> str:
        return _MEMORY_EVAL_SYSTEM

    @staticmethod
    def memory_evaluation_user(message: str) -> str:
        return message.strip()

    @staticmethod
    def reflection(user_message: str, assistant_response: str) -> str:
        return f"{_REFLECTION_PROMPT}\nUser: {user_message.strip()}\nAssistant: {assistant_response.strip()}"

    @staticmethod
    def memory_context_block(memories: list[dict[str, Any]]) -> str:
        """Legacy: build a 'Relevant Memories' block from memory documents.
        
        Deprecated: use user_facts_block instead.
        """
        if not memories:
            return ""
        lines = [f"- {m['document'].strip()}" for m in memories]
        return "Relevant Memories:\n" + "\n".join(lines)

    @staticmethod
    def user_facts_block(memories: list[dict[str, Any]]) -> str:
        """Build a 'Relevant User Facts' block using third-person fact text.
        
        Uses metadata.memory_fact or metadata.fact.memory_fact when available,
        falling back to the document field (which should also be third-person
        for new memories).
        """
        if not memories:
            return ""
        lines: list[str] = []
        for m in memories:
            fact_text = _extract_fact_text(m)
            if fact_text:
                lines.append(f"\u2022 {fact_text}")
        if not lines:
            return ""
        return "Relevant User Facts:\n" + "\n".join(lines)

    @staticmethod
    def conversation_block(conversation: list[dict[str, Any]]) -> str:
        if not conversation:
            return ""
        lines: list[str] = []
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if content:
                label = "User" if role == "user" else "Assistant"
                lines.append(f"{label}: {content}")
        return "Previous Conversation:\n" + "\n".join(lines)

    @staticmethod
    def combine_memory_context(
        memory_block: str,
        user_message: str,
    ) -> str:
        if memory_block.strip():
            return f"{memory_block.strip()}\n\nCurrent User Message:\n{user_message.strip()}"
        return user_message.strip()


def _extract_fact_text(memory: dict[str, Any]) -> str:
    """Extract the best third-person fact text from a memory record."""
    meta = memory.get("metadata", {}) or {}

    # Priority 1: structured fact data with memory_fact
    fact_data = meta.get("fact")
    if isinstance(fact_data, dict):
        mf = fact_data.get("memory_fact", "")
        if mf and _is_third_person(mf):
            return mf.strip()

    # Priority 2: top-level memory_fact in metadata
    mf = meta.get("memory_fact", "")
    if mf and _is_third_person(mf):
        return mf.strip()

    # Priority 3: document field (should be third-person for new memories)
    doc = memory.get("document", "").strip()
    if doc and _is_third_person(doc):
        return doc

    # Priority 4: document with first-person fallback conversion
    if doc:
        from .memory_normalizer import normalize_to_third_person
        cat = meta.get("category", "")
        return normalize_to_third_person(doc, category=cat)

    return ""


def _is_third_person(text: str) -> bool:
    """Check if text is in third-person (does not use I/my/me/we/our)."""
    from .memory_normalizer import is_first_person
    return not is_first_person(text)
