from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    GREETING = "GREETING"
    SMALL_TALK = "SMALL_TALK"
    MATH = "MATH"
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"
    PROGRAMMING = "PROGRAMMING"
    MEMORY_QUERY = "MEMORY_QUERY"
    PROJECT_DISCUSSION = "PROJECT_DISCUSSION"
    PERSONAL_INFORMATION = "PERSONAL_INFORMATION"
    WEB_SEARCH = "WEB_SEARCH"
    UNKNOWN = "UNKNOWN"


@dataclass
class FilterResult:
    query_type: QueryType
    filtered_history: list[dict[str, Any]] = field(default_factory=list)
    discarded: list[dict[str, Any]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    total_before: int = 0
    total_after: int = 0
    execution_time_ms: float = 0.0


_RELEVANCE_THRESHOLD: float = 0.70

_GREETING_WORDS = frozenset({
    "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
    "good evening", "good night", "bye", "thanks", "thank you",
})

_SMALL_TALK_WORDS = frozenset({
    "how are you", "how's it going", "what's up", "how do you do",
    "nice to meet", "pleasure", "howdy",
})

_MATH_PATTERN = re.compile(
    r"^what\s+is\s+[\d\s+\-*/%().,]+$|^calculate\s|^solve\s|^\d+\s*[+\-*/]|^\d+\s*\+|"
    r"square\s+root|derivative|integral|equation|factor|"
    r"\b[\d]+\s*[+\-*/]\s*[\d]+\b",
    re.IGNORECASE,
)

_PROGRAMMING_KEYWORDS = frozenset({
    "code", "programming", "function", "class", "variable", "algorithm",
    "debug", "compile", "syntax", "recursion", "loop", "array", "object",
    "python", "java", "javascript", "typescript", "rust", "go", "c++",
    "api", "framework", "library", "dependency", "npm", "pip", "import",
    "refactor", "test", "deploy", "docker", "kubernetes", "git",
    "binary", "search", "sort", "queue", "stack", "tree", "graph",
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

_MOVIE_KEYWORDS = frozenset({
    "movie", "film", "cinema", "watch", "show", "series", "episode",
    "netflix", "hulu", "disney", "streaming",
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

    if any(phrase in lowered for phrase in _SMALL_TALK_WORDS):
        return QueryType.SMALL_TALK

    if any(word in lowered for word in (
        "remember", "recall", "what did i", "what was", "do you know",
        "search memory", "find memory",
    )):
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

    if any(word in lowered for word in ("what is", "who is", "how does", "why does", "define", "meaning of", "explain")):
        return QueryType.GENERAL_KNOWLEDGE

    return QueryType.UNKNOWN


def _message_type_label(msg: dict[str, Any]) -> str:
    content = (msg.get("content") or "").lower()
    words = _words(content)
    if not content:
        return "empty"
    if words & _GREETING_WORDS and len(words) <= 5:
        return "greeting"
    if words & _FOOD_KEYWORDS:
        return "food"
    if words & _MOVIE_KEYWORDS:
        return "entertainment"
    if words & _PROGRAMMING_KEYWORDS:
        return "programming"
    if words & _PROJECT_KEYWORDS:
        return "project"
    if _MATH_PATTERN.match(content):
        return "math"
    return "general"


def score_message(msg: dict[str, Any], query_type: QueryType, query: str) -> float:
    content = (msg.get("content") or "").lower().strip()
    if not content:
        return 0.0

    msg_type = _message_type_label(msg)
    base = 0.5

    if query_type == QueryType.GREETING:
        return 0.0

    if query_type == QueryType.MATH:
        if msg_type == "math":
            base = 0.75
        elif msg_type == "greeting":
            base = 0.0
        else:
            base = 0.0

    elif query_type == QueryType.GENERAL_KNOWLEDGE:
        if msg_type in ("general", "programming", "project"):
            base = 0.60
        elif msg_type in ("food", "entertainment"):
            base = 0.0

    elif query_type == QueryType.PROGRAMMING:
        if msg_type == "programming":
            base = 0.85
        elif msg_type == "project":
            base = 0.65
        elif msg_type in ("food", "entertainment", "greeting"):
            base = 0.0

    elif query_type == QueryType.PROJECT_DISCUSSION:
        if msg_type == "project":
            base = 0.85
        elif msg_type == "programming":
            base = 0.60
        elif msg_type in ("food", "entertainment", "greeting"):
            base = 0.0

    elif query_type == QueryType.PERSONAL_INFORMATION:
        if msg_type == "general":
            base = 0.60
        elif msg_type in ("food", "programming"):
            base = 0.50
        elif msg_type in ("entertainment", "greeting"):
            base = 0.0

    elif query_type == QueryType.MEMORY_QUERY:
        base = 0.65

    elif query_type == QueryType.WEB_SEARCH:
        base = 0.0

    elif query_type == QueryType.SMALL_TALK:
        if msg_type == "greeting":
            base = 0.40
        else:
            base = 0.0

    return max(0.0, min(1.0, base))


def filter_conversation(
    query: str,
    history: list[dict[str, Any]],
    threshold: float = _RELEVANCE_THRESHOLD,
    max_turns: int = 10,
) -> FilterResult:
    started = time.perf_counter()
    total_before = len(history)

    query_type = classify_query(query)

    if query_type in (QueryType.GREETING, QueryType.MATH):
        elapsed = (time.perf_counter() - started) * 1000
        return FilterResult(
            query_type=query_type,
            total_before=total_before,
            total_after=0,
            discarded=list(history),
            execution_time_ms=round(elapsed, 2),
        )

    if query_type == QueryType.SMALL_TALK:
        elapsed = (time.perf_counter() - started) * 1000
        recent = history[-2:] if len(history) >= 2 else list(history)
        return FilterResult(
            query_type=query_type,
            filtered_history=recent,
            total_before=total_before,
            total_after=len(recent),
            discarded=history[:-len(recent)] if len(history) > len(recent) else [],
            execution_time_ms=round(elapsed, 2),
        )

    scored: list[tuple[float, dict[str, Any]]] = []
    discarded: list[dict[str, Any]] = []

    for msg in history:
        s = score_message(msg, query_type, query)
        if s >= threshold:
            scored.append((s, msg))
        else:
            discarded.append(msg)

    scored.sort(key=lambda x: x[0], reverse=True)
    filtered = [msg for _, msg in scored[:max_turns]]

    elapsed = (time.perf_counter() - started) * 1000

    return FilterResult(
        query_type=query_type,
        filtered_history=filtered,
        discarded=discarded,
        scores=[s for s, _ in scored],
        total_before=total_before,
        total_after=len(filtered),
        execution_time_ms=round(elapsed, 2),
    )
