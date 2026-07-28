"""Module 2 — Complexity Classifier.

Determines how difficult a request is: Simple, Medium, or Complex.
Simple requests bypass unnecessary processing stages.
"""

from __future__ import annotations

import re
from typing import Any

from ...schemas.reasoning import ComplexityLevel, SemanticIntentType

_SIMPLE_INTENTS: set[SemanticIntentType] = {
    SemanticIntentType.GREETING,
    SemanticIntentType.MATHEMATICS,
    SemanticIntentType.SIMPLE_QUESTION,
}

_MEDIUM_INTENTS: set[SemanticIntentType] = {
    SemanticIntentType.EXPLANATION,
    SemanticIntentType.MEMORY_RETRIEVAL,
    SemanticIntentType.CONVERSATION_CONTINUATION,
    SemanticIntentType.PROGRAMMING,
    SemanticIntentType.CODE_GENERATION,
}

_COMPLEX_INTENTS: set[SemanticIntentType] = {
    SemanticIntentType.PLANNING,
    SemanticIntentType.CREATIVE_WRITING,
    SemanticIntentType.CODE_GENERATION,
    SemanticIntentType.DEBUGGING,
    SemanticIntentType.REASONING,
}

_LENGTH_SIMPLE_THRESHOLD: int = 80
_LENGTH_COMPLEX_THRESHOLD: int = 200

_MULTI_SENTENCE_PATTERN = re.compile(r"[.!?]\s+[A-Z]")
_COMPLEX_KEYWORDS: list[re.Pattern[str]] = [
    re.compile(r"\b(because|however|therefore|consequently|furthermore|additionally|specifically|particularly)\b", re.IGNORECASE),
    re.compile(r"\b(design|architecture|compare|contrast|analyze|evaluate|justify|recommend|propose|outline)\b", re.IGNORECASE),
    re.compile(r"\b(if\s+.*\s+then|scenario|alternative|trade.?off|pros\s+and\s+cons)\b", re.IGNORECASE),
]


def classify_complexity(
    message: str,
    semantic_intent: SemanticIntentType,
) -> ComplexityLevel:
    if semantic_intent in _SIMPLE_INTENTS:
        return ComplexityLevel.SIMPLE

    if semantic_intent in _COMPLEX_INTENTS:
        return ComplexityLevel.COMPLEX

    if semantic_intent == SemanticIntentType.GENERAL_CONVERSATION:
        base = ComplexityLevel.MEDIUM if len(message.strip()) >= _LENGTH_SIMPLE_THRESHOLD else ComplexityLevel.SIMPLE
    elif semantic_intent in _MEDIUM_INTENTS:
        base = ComplexityLevel.MEDIUM
    else:
        base = ComplexityLevel.MEDIUM

    length = len(message.strip())

    if length > _LENGTH_COMPLEX_THRESHOLD:
        return ComplexityLevel.COMPLEX

    if length < _LENGTH_SIMPLE_THRESHOLD:
        sentences = _MULTI_SENTENCE_PATTERN.split(message.strip())
        if len(sentences) <= 1:
            if semantic_intent in _MEDIUM_INTENTS:
                return ComplexityLevel.MEDIUM
            return ComplexityLevel.SIMPLE

    complex_keyword_hits = sum(1 for p in _COMPLEX_KEYWORDS if p.search(message))

    if complex_keyword_hits >= 2:
        return ComplexityLevel.COMPLEX

    return base
