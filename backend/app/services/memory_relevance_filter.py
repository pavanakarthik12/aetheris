from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryType(str, Enum):
    MATH = "MATH"
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"
    MEMORY_QUERY = "MEMORY_QUERY"
    PERSONAL_INFORMATION = "PERSONAL_INFORMATION"
    PROJECT_DISCUSSION = "PROJECT_DISCUSSION"
    PROGRAMMING = "PROGRAMMING"
    CASUAL_CHAT = "CASUAL_CHAT"
    GREETING = "GREETING"
    WEB_SEARCH = "WEB_SEARCH"
    UNKNOWN = "UNKNOWN"


@dataclass
class FilterResult:
    query_type: QueryType
    relevant_memories: list[dict[str, Any]] = field(default_factory=list)
    discarded_memories: list[dict[str, Any]] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    execution_time_ms: float = 0.0


_RELEVANCE_THRESHOLD: float = 0.65

_MATH_PATTERN = re.compile(
    r"^what\s+is\s+[\d\s+\-*/%().,]+$|^calculate\s|^solve\s|^\d+\s*[+\-*/]|^\d+\s*\+",
    re.IGNORECASE,
)

_GREETING_WORDS = frozenset({
    "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
    "good evening", "good night", "bye", "thanks", "thank you",
})

_PROGRAMMING_KEYWORDS = frozenset({
    "code", "programming", "function", "class", "variable", "algorithm",
    "debug", "compile", "syntax", "recursion", "loop", "array", "object",
    "python", "java", "javascript", "typescript", "rust", "go", "c++",
    "api", "framework", "library", "dependency", "npm", "pip", "import",
    "refactor", "test", "deploy", "docker", "kubernetes", "git",
})

_PROJECT_KEYWORDS = frozenset({
    "project", "aetheris", "building", "build", "feature", "roadmap",
    "milestone", "architecture", "system design", "module", "service",
})

_PERSONAL_PHRASES = frozenset({
    "my name", "my favorite", "i like", "i love",
    "i prefer", "i enjoy", "i hate", "i dislike", "i want",
    "my hobby", "my interest", "my job", "my work",
    "my family", "my pet", "my home",
})

_FOOD_KEYWORDS = frozenset({
    "food", "eat", "pizza", "pasta", "cook", "recipe", "hungry",
    "meal", "breakfast", "lunch", "dinner", "snack", "fruit",
    "vegetable", "cuisine", "dish", "tasty",
})


def _words(text: str) -> set[str]:
    return set(text.lower().split())


def classify_query(message: str) -> QueryType:
    lowered = message.strip().lower()
    words = _words(lowered)

    if not lowered:
        return QueryType.UNKNOWN

    if _MATH_PATTERN.match(lowered):
        return QueryType.MATH

    if words & _GREETING_WORDS and len(words) <= 5:
        return QueryType.GREETING

    if any(word in lowered for word in ("remember", "recall", "what did i", "what was", "do you know", "search memory", "find memory")):
        return QueryType.MEMORY_QUERY

    if any(word in lowered for word in ("search", "find online", "look up", "google", "browse")):
        return QueryType.WEB_SEARCH

    if words & _PROGRAMMING_KEYWORDS:
        return QueryType.PROGRAMMING

    if words & _PROJECT_KEYWORDS:
        return QueryType.PROJECT_DISCUSSION

    personal_match_count = sum(1 for phrase in _PERSONAL_PHRASES if phrase in lowered)
    if personal_match_count >= 1:
        return QueryType.PERSONAL_INFORMATION

    if words & _FOOD_KEYWORDS:
        return QueryType.CASUAL_CHAT

    if any(word in lowered for word in ("what is", "who is", "how does", "why does", "define", "meaning of", "explain")):
        return QueryType.GENERAL_KNOWLEDGE

    return QueryType.UNKNOWN


def _category_boost(memory: dict[str, Any], query_type: QueryType) -> float:
    meta = memory.get("metadata", {}) or {}
    category = (meta.get("category") or "").lower()
    tags = (meta.get("tags") or "").lower()
    combined = f"{category} {tags}"

    if query_type == QueryType.PROGRAMMING:
        if any(kw in combined for kw in ("programming", "code", "language", "tech")):
            return 0.15
        if any(kw in combined for kw in ("food", "hobby", "swimming")):
            return -0.20

    elif query_type == QueryType.PROJECT_DISCUSSION:
        if any(kw in combined for kw in ("project", "aetheris", "feature", "build")):
            return 0.15
        if any(kw in combined for kw in ("food", "hobby", "language")):
            return -0.15

    elif query_type == QueryType.PERSONAL_INFORMATION:
        if any(kw in combined for kw in ("preference", "like", "dislike", "hobby", "interest")):
            return 0.10

    elif query_type == QueryType.CASUAL_CHAT:
        if any(kw in combined for kw in ("food", "hobby", "sport", "swimming")):
            return 0.10

    return 0.0


def _recency_boost(memory: dict[str, Any]) -> float:
    import datetime
    meta = memory.get("metadata", {}) or {}
    created_str = meta.get("created_at") or meta.get("updated_at") or ""
    if not created_str:
        return 0.0
    try:
        created = datetime.datetime.fromisoformat(created_str)
        days_old = (datetime.datetime.now(datetime.timezone.utc) - created).days
        if days_old <= 1:
            return 0.10
        if days_old <= 7:
            return 0.05
        if days_old <= 30:
            return 0.02
    except (ValueError, TypeError):
        pass
    return 0.0


def score_memory(memory: dict[str, Any], query_type: QueryType, query: str) -> float:
    base_score = memory.get("score", 0.0)

    if base_score < 0.1:
        return 0.0

    category_boost = _category_boost(memory, query_type)
    recency_boost = _recency_boost(memory)

    meta = memory.get("metadata", {}) or {}
    strength = meta.get("memory_strength", 0.5) if isinstance(meta, dict) else 0.5
    strength_factor = 1.0 + (strength - 0.5) * 0.1

    final = (base_score + category_boost + recency_boost) * strength_factor
    return max(0.0, min(1.0, final))


def filter_memories(
    query: str,
    memories: list[dict[str, Any]],
    threshold: float = _RELEVANCE_THRESHOLD,
    max_memories: int = 5,
) -> FilterResult:
    started = time.perf_counter()

    query_type = classify_query(query)

    if query_type in (QueryType.MATH, QueryType.GREETING):
        elapsed = (time.perf_counter() - started) * 1000
        return FilterResult(
            query_type=query_type,
            execution_time_ms=round(elapsed, 2),
        )

    relevant: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    scores: dict[str, float] = {}

    for memory in memories:
        mem_id = memory.get("id") or str(id(memory))
        doc = memory.get("document", "")
        if not isinstance(doc, str) or not doc.strip():
            discarded.append(memory)
            continue

        score = score_memory(memory, query_type, query)
        scores[mem_id] = round(score, 4)

        if score >= threshold:
            memory["_relevance_score"] = round(score, 4)
            relevant.append(memory)
        else:
            discarded.append(memory)

    relevant.sort(key=lambda m: m.get("_relevance_score", 0.0), reverse=True)
    relevant = relevant[:max_memories]

    elapsed = (time.perf_counter() - started) * 1000

    return FilterResult(
        query_type=query_type,
        relevant_memories=relevant,
        discarded_memories=discarded,
        scores=scores,
        execution_time_ms=round(elapsed, 2),
    )
