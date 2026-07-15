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

_INTENT_PROMPT: str = (
    'Classify intent. Return {"intent":"<TYPE>","confidence":0.0}'
    ' where TYPE is one of: NORMAL_CHAT, CREATE_MEMORY, UPDATE_MEMORY,'
    " DELETE_MEMORY, MERGE_MEMORY, SEARCH_MEMORY, WEB_SEARCH, SYSTEM_QUERY,"
    " MULTI_ACTION, UNKNOWN.\n"
    "If unsure, prefer NORMAL_CHAT."
)

_MEMORY_EVAL_SYSTEM: str = (
    "Decide if this message should be a permanent memory."
    ' Return {"store":true/false,"importance":0.0,"category":"<cat>","reason":"<text>"}.'
    " Categories: Preference,Project,Goal,Skill,Relationship,Fact,"
    "Achievement,Event,Task,Other.\n"
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
        return _SYSTEM_PROMPT

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
        if not memories:
            return ""
        lines = [f"- {m['document'].strip()}" for m in memories]
        return "Relevant Memories:\n" + "\n".join(lines)

    @staticmethod
    def combine_memory_context(
        memory_block: str,
        user_message: str,
    ) -> str:
        if memory_block.strip():
            return f"{memory_block.strip()}\n\nCurrent User Message:\n{user_message.strip()}"
        return user_message.strip()
