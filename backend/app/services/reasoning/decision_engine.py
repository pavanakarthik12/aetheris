"""Module 4 — Decision Engine.

Determines which memory sources are needed and whether
clarification or planning is required. Minimizes unnecessary work.
"""

from __future__ import annotations

from ...schemas.reasoning import (
    ComplexityLevel,
    ConfidenceLevel,
    MemorySourceDecision,
    SemanticIntentType,
)

_INTENTS_REQUIRING_CONVERSATION: set[SemanticIntentType] = {
    SemanticIntentType.CONVERSATION_CONTINUATION,
    SemanticIntentType.GENERAL_CONVERSATION,
    SemanticIntentType.EXPLANATION,
    SemanticIntentType.DEBUGGING,
}

_INTENTS_REQUIRING_LONG_TERM: set[SemanticIntentType] = {
    SemanticIntentType.MEMORY_RETRIEVAL,
    SemanticIntentType.CONVERSATION_CONTINUATION,
}

_INTENTS_REQUIRING_REFLECTION: set[SemanticIntentType] = {
    SemanticIntentType.GENERAL_CONVERSATION,
    SemanticIntentType.EXPLANATION,
    SemanticIntentType.DEBUGGING,
    SemanticIntentType.REASONING,
}

_INTENTS_REQUIRING_PLANNING: set[SemanticIntentType] = {
    SemanticIntentType.PLANNING,
    SemanticIntentType.CODE_GENERATION,
    SemanticIntentType.DEBUGGING,
    SemanticIntentType.REASONING,
}

_INTENTS_REQUIRING_CLARIFICATION: set[SemanticIntentType] = {
    SemanticIntentType.UNKNOWN,
}

_CLARIFICATION_KEYWORDS: list[str] = [
    "build me a", "create a", "make me a", "write me a",
    "give me something", "do something",
    "i don't know", "what should i",
]

_SYSTEM_MEMORY_INTENTS: set[SemanticIntentType] = {
    SemanticIntentType.SIMPLE_QUESTION,
    SemanticIntentType.EXPLANATION,
}


def decide(
    semantic_intent: SemanticIntentType,
    complexity: ComplexityLevel,
    message: str,
    has_decomposed_tasks: bool,
) -> tuple[MemorySourceDecision, bool, str, bool]:
    msg_lower = message.strip().lower()

    needs_conversation = semantic_intent in _INTENTS_REQUIRING_CONVERSATION
    needs_long_term = semantic_intent in _INTENTS_REQUIRING_LONG_TERM
    needs_system = semantic_intent in _SYSTEM_MEMORY_INTENTS
    needs_reflection = semantic_intent in _INTENTS_REQUIRING_REFLECTION
    needs_planning = semantic_intent in _INTENTS_REQUIRING_PLANNING
    needs_clarification = semantic_intent in _INTENTS_REQUIRING_CLARIFICATION

    for kw in _CLARIFICATION_KEYWORDS:
        if kw in msg_lower:
            needs_clarification = True
            clarification_q = _clarification_for(msg_lower)
            break
    else:
        clarification_q = ""

    if complexity == ComplexityLevel.SIMPLE:
        needs_conversation = False
        needs_long_term = False
        needs_system = False
        needs_reflection = False
        needs_planning = False

    if complexity == ComplexityLevel.COMPLEX:
        needs_planning = True
        if not needs_long_term and not needs_conversation:
            needs_long_term = True

    if semantic_intent == SemanticIntentType.GREETING:
        needs_conversation = False
        needs_long_term = False
        needs_system = False
        needs_reflection = False
        needs_planning = False

    if has_decomposed_tasks:
        needs_planning = True

    decision = MemorySourceDecision(
        conversation_memory=needs_conversation,
        long_term_memory=needs_long_term,
        system_memory=needs_system,
        reflection=needs_reflection,
    )

    return decision, needs_clarification, clarification_q, needs_planning


def _clarification_for(msg: str) -> str:
    if "website" in msg or "web app" in msg:
        return "What kind of website? (e.g., portfolio, dashboard, e-commerce, blog)"
    if "app" in msg:
        return "What kind of application? (e.g., web, mobile, CLI, desktop)"
    if "tool" in msg:
        return "What should this tool do?"
    if "system" in msg:
        return "What kind of system? (e.g., authentication, monitoring, payment)"
    return "Could you provide more details about what you'd like me to create?"
