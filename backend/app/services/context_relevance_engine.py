from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..config.settings import get_settings

logger = logging.getLogger(__name__)

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "about",
    "up", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "it", "its", "my", "your", "his", "her",
    "our", "their", "me", "him", "us", "them", "i",
})


@dataclass
class RelevanceResult:
    relevant: list[dict[str, Any]] = field(default_factory=list)
    discarded: list[dict[str, Any]] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    total_before: int = 0
    total_after: int = 0
    execution_time_ms: float = 0.0


@dataclass
class RelevanceLog:
    query: str
    category: str
    total_retrieved: int
    total_injected: int
    total_rejected: int
    scores: dict[str, float]
    execution_time_ms: float


class ContextRelevanceEngine:

    def __init__(self, embedding_service=None):
        self._embedding = embedding_service
        self._settings = get_settings()

    async def filter_memories(
        self,
        query: str,
        memories: list[dict[str, Any]],
    ) -> RelevanceResult:
        started = time.perf_counter()
        total_before = len(memories)

        if not memories:
            elapsed = (time.perf_counter() - started) * 1000
            return RelevanceResult(
                total_before=0,
                total_after=0,
                execution_time_ms=round(elapsed, 2),
            )

        threshold = self._settings.memory_relevance_threshold
        max_results = self._settings.max_memory_entries

        query_embedding = None
        if self._embedding is not None and len(memories) > 0:
            try:
                query_embedding = await self._embedding.embed_text(query)
            except Exception:
                logger.debug("Embedding failed for query, falling back to keyword signals")

        relevant: list[dict[str, Any]] = []
        discarded: list[dict[str, Any]] = []
        scores: dict[str, float] = {}

        for memory in memories:
            mem_id = memory.get("id") or str(id(memory))
            score = await self._score_memory(memory, query, query_embedding)
            scores[mem_id] = round(score, 4)

            if score >= threshold:
                memory["_relevance_score"] = round(score, 4)
                relevant.append(memory)
            else:
                discarded.append(memory)

        relevant.sort(key=lambda m: m.get("_relevance_score", 0.0), reverse=True)
        relevant = relevant[:max_results]

        elapsed = (time.perf_counter() - started) * 1000

        logger.info(
            "ContextRelevanceEngine | memories | query=%.50s | total=%d | injected=%d | rejected=%d | threshold=%.2f | %.2fms",
            query,
            total_before,
            len(relevant),
            len(discarded),
            threshold,
            elapsed,
        )

        for mem_id, score in list(scores.items())[:10]:
            doc = ""
            for m in memories:
                if (m.get("id") or str(id(m))) == mem_id:
                    doc = (m.get("document") or "")[:80]
                    break
            logger.debug(
                "  memory score | id=%.8s | score=%.4f | doc=%.60s",
                mem_id, score, doc,
            )

        return RelevanceResult(
            relevant=relevant,
            discarded=discarded,
            scores=scores,
            total_before=total_before,
            total_after=len(relevant),
            execution_time_ms=round(elapsed, 2),
        )

    async def filter_conversation(
        self,
        query: str,
        messages: list[dict[str, Any]],
    ) -> RelevanceResult:
        started = time.perf_counter()
        total_before = len(messages)

        if not messages:
            elapsed = (time.perf_counter() - started) * 1000
            return RelevanceResult(
                total_before=0,
                total_after=0,
                execution_time_ms=round(elapsed, 2),
            )

        threshold = self._settings.conversation_relevance_threshold
        max_results = self._settings.max_conversation_entries

        relevant: list[dict[str, Any]] = []
        discarded: list[dict[str, Any]] = []
        scores: dict[str, float] = {}

        for msg in messages:
            msg_id = str(id(msg))
            score = self._score_conversation_message(msg, query)
            scores[msg_id] = round(score, 4)

            if score >= threshold:
                msg["_relevance_score"] = round(score, 4)
                relevant.append(msg)
            else:
                discarded.append(msg)

        relevant.sort(key=lambda m: m.get("_relevance_score", 0.0), reverse=True)
        relevant = relevant[:max_results]

        elapsed = (time.perf_counter() - started) * 1000

        logger.info(
            "ContextRelevanceEngine | conversation | query=%.50s | total=%d | injected=%d | rejected=%d | threshold=%.2f | %.2fms",
            query,
            total_before,
            len(relevant),
            len(discarded),
            threshold,
            elapsed,
        )

        for msg_id, score in list(scores.items())[:6]:
            content = ""
            for m in messages:
                if id(m) == int(msg_id) if msg_id.isdigit() else False:
                    content = (m.get("content") or "")[:60]
                    break
            logger.debug(
                "  conv score | score=%.4f | content=%.60s",
                score, content,
            )

        return RelevanceResult(
            relevant=relevant,
            discarded=discarded,
            scores=scores,
            total_before=total_before,
            total_after=len(relevant),
            execution_time_ms=round(elapsed, 2),
        )

    async def filter_search_results(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> RelevanceResult:
        started = time.perf_counter()
        total_before = len(results)

        if not results:
            elapsed = (time.perf_counter() - started) * 1000
            return RelevanceResult(
                total_before=0,
                total_after=0,
                execution_time_ms=round(elapsed, 2),
            )

        threshold = self._settings.search_relevance_threshold
        max_results = self._settings.max_search_results

        relevant: list[dict[str, Any]] = []
        discarded: list[dict[str, Any]] = []
        scores: dict[str, float] = {}

        for result in results:
            result_id = result.get("url") or result.get("id") or str(id(result))
            score = self._score_search_result(result, query)
            scores[result_id] = round(score, 4)

            if score >= threshold:
                result["_relevance_score"] = round(score, 4)
                relevant.append(result)
            else:
                discarded.append(result)

        relevant.sort(key=lambda r: r.get("_relevance_score", 0.0), reverse=True)
        relevant = relevant[:max_results]

        elapsed = (time.perf_counter() - started) * 1000

        logger.info(
            "ContextRelevanceEngine | search | query=%.50s | total=%d | injected=%d | rejected=%d | threshold=%.2f | %.2fms",
            query,
            total_before,
            len(relevant),
            len(discarded),
            threshold,
            elapsed,
        )

        return RelevanceResult(
            relevant=relevant,
            discarded=discarded,
            scores=scores,
            total_before=total_before,
            total_after=len(relevant),
            execution_time_ms=round(elapsed, 2),
        )

    async def _score_memory(
        self,
        memory: dict[str, Any],
        query: str,
        query_embedding: list[float] | None = None,
    ) -> float:
        document = memory.get("document", "") or ""
        meta = memory.get("metadata", {}) or {}

        if not document:
            return 0.0

        chroma_score = memory.get("score", 0.0)
        keyword_sim = _keyword_similarity(query, document)
        topic_align = _topic_alignment(query, document, meta)
        recency = _recency_boost(meta)
        importance = _importance_boost(meta)

        weights = self._settings.relevance_weights

        final = (
            chroma_score * weights.get("chroma", 0.30) +
            keyword_sim * weights.get("keyword", 0.25) +
            topic_align * weights.get("topic", 0.20) +
            recency * weights.get("recency", 0.10) +
            importance * weights.get("importance", 0.15)
        )

        return max(0.0, min(1.0, final))

    def _score_conversation_message(
        self,
        msg: dict[str, Any],
        query: str,
    ) -> float:
        content = (msg.get("content") or "").strip()
        if not content:
            return 0.0

        keyword_sim = _keyword_similarity(query, content)
        topic_align = _conversation_topic_alignment(query, content, msg.get("role", "user"))

        keyword_score = keyword_sim if keyword_sim > 0 else 0.0
        topic_score = topic_align if topic_align > 0 else 0.0

        any_match = (
            keyword_score >= 0.30 or
            topic_score >= 0.40 or
            (keyword_score + topic_score) >= 0.50
        )

        if not any_match:
            return 0.0

        final = max(keyword_score, topic_score) * 0.60 + min(keyword_score, topic_score) * 0.40

        return max(0.0, min(1.0, final))

    def _score_search_result(
        self,
        result: dict[str, Any],
        query: str,
    ) -> float:
        title = (result.get("title") or result.get("name") or "")
        snippet = (result.get("snippet") or result.get("content") or "")
        text = f"{title} {snippet}"

        if not text.strip():
            return 0.0

        keyword_sim = _keyword_similarity(query, text)

        existing_score = result.get("score") or result.get("relevance_score") or result.get("confidence") or 0.0
        if isinstance(existing_score, (int, float)):
            existing_score = float(existing_score)
        else:
            existing_score = 0.0

        weights = self._settings.relevance_weights

        final = (
            keyword_sim * weights.get("keyword", 0.50) +
            existing_score * weights.get("existing_score", 0.50)
        )

        return max(0.0, min(1.0, final))


def _tokenize(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-zA-Z0-9_\-]+", text.lower()))
    return tokens - _STOP_WORDS


def _keyword_similarity(a: str, b: str) -> float:
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    jaccard = len(intersection) / len(union) if union else 0.0

    overlap_ratio = len(intersection) / len(tokens_a) if tokens_a else 0.0

    return (jaccard * 0.5 + overlap_ratio * 0.5)


_MATH_KEYWORDS = frozenset({
    "math", "calculate", "equation", "solve", "number", "algebra",
    "geometry", "calculus", "derivative", "integral", "sum",
})
_GENERAL_KNOWLEDGE_KEYWORDS = frozenset({
    "what is", "who is", "how does", "why does", "define", "meaning of",
    "explain", "describe", "tell me about", "what was", "history",
    "overview", "summary", "biography",
})
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
    "milestone", "architecture", "module", "service",
})
_PERSONAL_PHRASES = frozenset({
    "my name", "my favorite", "i like", "i love",
    "i prefer", "i enjoy", "i hate", "i dislike", "i want",
    "my hobby", "my interest", "my job", "my work",
    "my family", "my pet", "my home", "who am i",
})


def _classify_query_topic(query: str) -> str:
    lowered = query.strip().lower()

    if any(phrase in lowered for phrase in _PERSONAL_PHRASES):
        return "personal"

    if lowered.startswith("what is") or lowered.startswith("who is") or lowered.startswith("how does") or lowered.startswith("why does"):
        query_words = _tokenize(query)
        if query_words & _PROGRAMMING_KEYWORDS:
            return "programming"
        if query_words & _MATH_KEYWORDS:
            return "math"
        return "general_knowledge"

    query_words = _tokenize(query)
    if query_words & _PROGRAMMING_KEYWORDS:
        return "programming"
    if query_words & _PROJECT_KEYWORDS:
        return "project"
    if query_words & _MATH_KEYWORDS:
        return "math"

    return "general"


_CATEGORY_TOPIC_MAP: dict[str, set[str]] = {
    "programming": {"programming", "code", "language", "tech", "software", "development"},
    "project": {"project", "aetheris", "feature", "build", "architecture", "milestone"},
    "preference": {"preference", "like", "dislike", "hobby", "interest", "favorite"},
    "personal": {"preference", "like", "dislike", "hobby", "interest", "favorite", "name", "family"},
    "general_knowledge": {"general", "knowledge", "history", "science", "education"},
    "math": {"math", "number", "calculation", "equation"},
    "food": {"food", "cuisine", "dish", "cook", "recipe", "meal"},
    "entertainment": {"movie", "music", "show", "game", "entertainment"},
}


def _topic_alignment(query: str, document: str, metadata: dict[str, Any]) -> float:
    query_topic = _classify_query_topic(query)

    category = (metadata.get("category") or "").lower()
    tags = (metadata.get("tags") or "").lower()
    doc_lower = document.lower()

    query_topics = _CATEGORY_TOPIC_MAP.get(query_topic, set())
    doc_topics = _CATEGORY_TOPIC_MAP.get(category, set())

    if query_topic == "general_knowledge":
        if category in ("fact", "general", ""):
            return 0.10
        return 0.0

    if query_topic == "personal":
        if category in ("preference", "personal", "fact"):
            return 0.80
        query_words = _tokenize(query)
        if query_words & _PROGRAMMING_KEYWORDS and category == "programming":
            return 0.70
        if query_words & _PROJECT_KEYWORDS and category == "project":
            return 0.70
        return 0.20

    if query_topic == "programming":
        if category == "programming":
            return 0.90
        if category == "project":
            return 0.40
        if any(kw in doc_lower for kw in ("programming", "code", "python", "language")):
            return 0.50
        return 0.0

    if query_topic == "project":
        if category == "project":
            return 0.90
        if category == "programming":
            return 0.40
        if any(kw in doc_lower for kw in ("project", "aetheris", "build")):
            return 0.50
        return 0.0

    if query_topic == "math":
        if category == "math":
            return 0.80
        return 0.0

    shared = query_topics & doc_topics
    if shared:
        return 0.30 + (len(shared) * 0.10)

    return 0.0


def _conversation_topic_alignment(query: str, content: str, role: str) -> float:
    query_topic = _classify_query_topic(query)
    content_lower = content.lower()

    if query_topic == "general_knowledge":
        if any(phrase in content_lower for phrase in _GENERAL_KNOWLEDGE_KEYWORDS):
            return 0.60
        if any(kw in content_lower for kw in ("what", "who", "how", "why", "explain")):
            return 0.40
        content_words = _tokenize(content)
        query_words = _tokenize(query)
        shared = content_words & query_words if query_words else set()
        if len(shared) >= 2:
            return 0.50
        if len(shared) >= 1:
            return 0.30
        return 0.10

    if query_topic == "personal":
        if any(phrase in content_lower for phrase in _PERSONAL_PHRASES):
            return 0.80
        return 0.10

    content_words = _tokenize(content)
    content_set = set(content_words)

    if query_topic == "programming":
        shared = content_set & _PROGRAMMING_KEYWORDS
        if shared:
            return 0.60 + (len(shared) * 0.05)
        if any(kw in content_lower for kw in ("function", "class", "code")):
            return 0.50
        return 0.0

    if query_topic == "project":
        shared = content_set & _PROJECT_KEYWORDS
        if shared:
            return 0.60 + (len(shared) * 0.05)
        return 0.0

    if query_topic == "math":
        if any(kw in content_lower for kw in _MATH_KEYWORDS):
            return 0.60
        return 0.0

    return 0.20


def _recency_boost(metadata: dict[str, Any]) -> float:
    import datetime
    created_str = metadata.get("created_at") or metadata.get("updated_at") or ""
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


def _importance_boost(metadata: dict[str, Any]) -> float:
    strength = metadata.get("memory_strength", 0.5) if isinstance(metadata, dict) else 0.5
    if isinstance(strength, (int, float)):
        return max(0.0, min(0.15, (strength - 0.5) * 0.3))
    return 0.0
