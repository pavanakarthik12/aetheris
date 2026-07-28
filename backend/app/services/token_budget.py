"""Dynamic token budget selection based on request complexity.

Output token limits are deliberately generous to prevent mid-response
truncation (finish_reason="length"). If you need tighter limits for
cost control, override via LLM_MAX_TOKENS in .env.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config.settings import get_settings
from ..schemas.routing import IntentType

_DEFAULT_MAX = 1024


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
    r"\b(explain|describe|what is|how does|tell me about|elaborate|detail"
    r"|who is|what was|overview|biography|summary)\b",
    re.IGNORECASE,
)


def _max_limit() -> int:
    """Respect the user-configured maximum if set, else use default."""
    try:
        return get_settings().llm_max_tokens
    except Exception:
        return _DEFAULT_MAX


def select_budget(
    message: str,
    intent: IntentType | None = None,
    memory_count: int = 0,
) -> TokenBudget:
    text = message.strip()
    max_limit = _max_limit()

    if intent == IntentType.SYSTEM_QUERY:
        return TokenBudget(96, 0.7, "system_query")
    if intent == IntentType.SEARCH_MEMORY:
        return TokenBudget(128, 0.7, "memory_search")
    if intent == IntentType.CREATE_MEMORY:
        return TokenBudget(96, 0.7, "create_memory")
    if intent == IntentType.DELETE_MEMORY:
        return TokenBudget(64, 0.7, "delete_memory")

    if _GREETING_PATTERN.search(text) and len(text) < 30:
        return TokenBudget(32, 0.7, "greeting")

    if _EXPLANATION_PATTERN.search(text):
        if len(text) > 150 or memory_count > 2:
            return TokenBudget(min(2048, max_limit), 0.7, "long_explain")
        return TokenBudget(min(1024, max_limit), 0.7, "explain")

    if _CODE_PATTERN.search(text):
        if len(text) > 200:
            return TokenBudget(min(2048, max_limit), 0.3, "large_code")
        return TokenBudget(min(1024, max_limit), 0.3, "code_gen")

    if memory_count > 2:
        return TokenBudget(min(1024, max_limit), 0.7, "memory_rich")

    if _QUESTION_PATTERN.search(text):
        return TokenBudget(min(512, max_limit), 0.7, "question")

    if len(text) < 50:
        return TokenBudget(96, 0.7, "short_chat")

    if len(text) > 300:
        return TokenBudget(min(1024, max_limit), 0.7, "long_input")

    return TokenBudget(min(384, max_limit), 0.7, "normal")
