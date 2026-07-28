"""Module 1 — Intent Analyzer.

Classifies the semantic intent of a user message using
lightweight deterministic rules. Avoids LLM calls.
"""

from __future__ import annotations

import re
from typing import Any

from ...schemas.reasoning import SemanticIntentType

_GREETING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening|night)|bye|thanks?|thank you)\s*[!.]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(hi|hello|hey)(\s+there)?\s*[!.]*\s*$", re.IGNORECASE),
]

_MATH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(\d+\s*[\+\-\*/%]\s*\d+|what\s+is\s+\d+|calculate|compute|solve|math|algebra|equation|(\d+\s*\+\s*\d+))\b", re.IGNORECASE),
    re.compile(r"^\s*[\d\s+\-*/%\(\)]+\s*=\s*\?\s*$"),
    re.compile(r"\b(add|subtract|multiply|divide|sum|difference|product|quotient)\b.*\d", re.IGNORECASE),
]

_PROGRAMMING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(code|program|function|class|method|variable|import|syntax|compile|debug|refactor|api|endpoint|route|middleware|algorithm|data\s+structure)\b", re.IGNORECASE),
    re.compile(r"\bpython|javascript|typescript|rust|java|golang|c\+\+|ruby|php|swift|kotlin|sql|html|css\b", re.IGNORECASE),
]

_MEMORY_RETRIEVAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(what\s+(do|did)\s+(i|you)\s+(know|remember|store|have|recall|say|tell)|do\s+you\s+remember|recall\s+(that|what|my|the|our)|what\s+(was|were)\s+we\s+(talking|discussing))\b", re.IGNORECASE),
    re.compile(r"\b(search|find|look\s+up|retrieve|get)\s+.*\b(memory|memories|fact|information|detail)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(is|are|was|were|am)\s+(my|the|i)\s+(name|project|goal|preference|skill|language|favorite|current|building|working)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+\w+\s+(am|is|are)\s+(i|you)\b.*\b(building|working|doing|creating|making|using)\b", re.IGNORECASE),
]

_CONVERSATION_CONTINUATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(continue|previous|last\s+(message|response|answer|chat|conversation|exchange)|what\s+(did|was)\s+(i|we|you)\s+(just\s+)?(say|talk|ask|discuss))\b", re.IGNORECASE),
    re.compile(r"\btwo\s+messages\s+ago|earlier\s+in\s+the\s+conversation|as\s+i\s+(said|mentioned|told)\b", re.IGNORECASE),
]

_EXPLANATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(explain|what\s+is|define|describe|how\s+(does|do|would|can)|why\s+(does|is|do|would)|elaborate|clarify|tell\s+me\s+about)\b", re.IGNORECASE),
]

_PLANNING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(plan|planning|roadmap|strategy|timeline|milestone|step\s+by\s+step|approach|design\s+a|architecture|system\s+design)\b", re.IGNORECASE),
    re.compile(r"\b(build\s+a|create\s+a|develop\s+a|implement\s+a)\s+\w+\s+(system|app|platform|service|tool|framework)\b", re.IGNORECASE),
]

_CREATIVE_WRITING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(write|story|poem|essay|article|blog|content|creative|narrative|fiction|script|dialogue)\b", re.IGNORECASE),
]

_CODE_GENERATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(generate|write|create|implement)\s+(a|an|the\s+)?(\w+\s+)?(function|class|program|script|module|component|app|code)\b", re.IGNORECASE),
    re.compile(r"\b(show\s+me|give\s+me|need)\s+(a|an|the)?\s*(code\s+)?example\b", re.IGNORECASE),
    re.compile(r"\bgenerate\s+(a|an|the)\s+\w+\s+(script|code|program|function|class|module|app)\b", re.IGNORECASE),
]

_DEBUGGING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(debug|bug|error|issue|problem|not\s+working|broken|fix|crash|exception|traceback|stack\s+trace|unexpected\s+behavior)\b", re.IGNORECASE),
    re.compile(r"\b(why\s+(isn't|is\s+not|doesn't|does\s+not|won't|will\s+not))\b", re.IGNORECASE),
]

_REASONING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(think|reason|logical\s+|deduce|infer|conclude|analyze|if\s+.*\s+then\b|compare\s+and\s+contrast|what\s+if|scenario)\b", re.IGNORECASE),
]


def analyze_intent(message: str) -> SemanticIntentType:
    if not message or not message.strip():
        return SemanticIntentType.UNKNOWN

    msg = message.strip()

    if any(p.search(msg) for p in _GREETING_PATTERNS):
        return SemanticIntentType.GREETING

    if any(p.search(msg) for p in _MATH_PATTERNS):
        return SemanticIntentType.MATHEMATICS

    if any(p.search(msg) for p in _MEMORY_RETRIEVAL_PATTERNS):
        return SemanticIntentType.MEMORY_RETRIEVAL

    if any(p.search(msg) for p in _CONVERSATION_CONTINUATION_PATTERNS):
        return SemanticIntentType.CONVERSATION_CONTINUATION

    if any(p.search(msg) for p in _CODE_GENERATION_PATTERNS):
        return SemanticIntentType.CODE_GENERATION

    if any(p.search(msg) for p in _DEBUGGING_PATTERNS):
        return SemanticIntentType.DEBUGGING

    if any(p.search(msg) for p in _PROGRAMMING_PATTERNS):
        return SemanticIntentType.PROGRAMMING

    if any(p.search(msg) for p in _CREATIVE_WRITING_PATTERNS):
        return SemanticIntentType.CREATIVE_WRITING

    if any(p.search(msg) for p in _PLANNING_PATTERNS):
        return SemanticIntentType.PLANNING

    if any(p.search(msg) for p in _REASONING_PATTERNS):
        return SemanticIntentType.REASONING

    if any(p.search(msg) for p in _EXPLANATION_PATTERNS):
        return SemanticIntentType.EXPLANATION

    return SemanticIntentType.GENERAL_CONVERSATION
