"""Module 8 — Confidence Estimator.

Generates an internal confidence score based on intent,
complexity, and available information.
"""

from __future__ import annotations

from ...schemas.reasoning import (
    CognitiveTrace,
    ComplexityLevel,
    ConfidenceLevel,
    SemanticIntentType,
)

_HIGH_CONFIDENCE_INTENTS: set[SemanticIntentType] = {
    SemanticIntentType.MATHEMATICS,
    SemanticIntentType.MEMORY_RETRIEVAL,
    SemanticIntentType.GREETING,
    SemanticIntentType.SIMPLE_QUESTION,
    SemanticIntentType.GENERAL_CONVERSATION,
}

_MEDIUM_CONFIDENCE_INTENTS: set[SemanticIntentType] = {
    SemanticIntentType.PROGRAMMING,
    SemanticIntentType.EXPLANATION,
    SemanticIntentType.CONVERSATION_CONTINUATION,
    SemanticIntentType.DEBUGGING,
}

_LOW_CONFIDENCE_INTENTS: set[SemanticIntentType] = {
    SemanticIntentType.UNKNOWN,
    SemanticIntentType.PLANNING,
    SemanticIntentType.CREATIVE_WRITING,
}


def estimate_confidence(
    semantic_intent: SemanticIntentType,
    complexity: ComplexityLevel,
    trace: CognitiveTrace,
) -> ConfidenceLevel:
    if semantic_intent in _HIGH_CONFIDENCE_INTENTS:
        return ConfidenceLevel.HIGH

    if semantic_intent in _MEDIUM_CONFIDENCE_INTENTS:
        base = ConfidenceLevel.MEDIUM
    elif semantic_intent in _LOW_CONFIDENCE_INTENTS:
        base = ConfidenceLevel.LOW
    else:
        base = ConfidenceLevel.MEDIUM

    if complexity == ComplexityLevel.COMPLEX:
        if base == ConfidenceLevel.LOW:
            return ConfidenceLevel.LOW
        if base == ConfidenceLevel.MEDIUM:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.MEDIUM

    return base
