"""Dynamic token budget selection based on request complexity."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..schemas.routing import IntentType


@dataclass(frozen=True)
class TokenBudget:
    max_tokens: int
    temperature: float = 0.7
    label: str = "normal"


_GREETING_PATTERN = re.compile(
    r"^(hi|hey|hello|howdy|sup|yo|thanks|bye|goodbye|ok|okay)\W*$",
    re.IGNORECASE,
)
_QUESTION_PATTERN = re.compile(r"\?$")
_CODE_PATTERN = re.compile(
    r"\b(code|function|class|def|import|implement|write a|create a|generate)\b",
    re.IGNORECASE,
)
_EXPLANATION_PATTERN = re.compile(
    r"\b(explain|describe|what is|how does|tell me about|elaborate|detail)\b",
    re.IGNORECASE,
)


def select_budget(
    message: str,
    intent: IntentType | None = None,
    memory_count: int = 0,
) -> TokenBudget:
    text = message.strip()

    if intent == IntentType.SYSTEM_QUERY:
        return TokenBudget(96, 0.7, "system_query")
    if intent == IntentType.SEARCH_MEMORY:
        return TokenBudget(128, 0.7, "memory_search")
    if intent == IntentType.CREATE_MEMORY:
        return TokenBudget(96, 0.7, "create_memory")
    if intent == IntentType.DELETE_MEMORY:
        return TokenBudget(64, 0.7, "delete_memory")

    if _CODE_PATTERN.search(text):
        if len(text) > 200:
            return TokenBudget(512, 0.3, "large_code")
        return TokenBudget(384, 0.3, "code_gen")

    if _GREETING_PATTERN.search(text) and len(text) < 30:
        return TokenBudget(32, 0.7, "greeting")

    if memory_count > 2:
        if _EXPLANATION_PATTERN.search(text) or len(text) > 200:
            return TokenBudget(384, 0.7, "long_explain")
        return TokenBudget(256, 0.7, "memory_rich")

    if _QUESTION_PATTERN.search(text):
        return TokenBudget(128, 0.7, "question")

    if len(text) < 50:
        return TokenBudget(96, 0.7, "short_chat")

    if len(text) > 300:
        return TokenBudget(384, 0.7, "long_input")

    return TokenBudget(192, 0.7, "normal")
