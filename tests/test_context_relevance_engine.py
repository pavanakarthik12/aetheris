from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.context_relevance_engine import (
    ContextRelevanceEngine,
    _classify_query_topic,
    _keyword_similarity,
    _topic_alignment,
)


class FakeEmbeddingService:
    async def embed_text(self, text: str) -> list[float]:
        normalized = text.lower()
        if "kingdom" in normalized:
            return [0.1, 0.1, 0.9]
        if "project" in normalized or "aetheris" in normalized:
            return [0.9, 0.1, 0.1]
        if "python" in normalized or "programming" in normalized:
            return [0.1, 0.9, 0.1]
        return [0.33, 0.33, 0.34]


def _make_memory(
    document: str,
    score: float = 0.5,
    category: str = "general",
    memory_strength: float = 0.5,
    memory_id: str | None = None,
) -> dict[str, Any]:
    return {
        "id": memory_id or document[:16],
        "document": document,
        "score": score,
        "metadata": {
            "category": category,
            "memory_strength": memory_strength,
        },
    }


def _make_message(content: str, role: str = "user") -> dict[str, Any]:
    return {"role": role, "content": content}


def _make_search_result(
    title: str,
    snippet: str,
    url: str = "https://example.com",
    score: float = 0.5,
) -> dict[str, Any]:
    return {
        "title": title,
        "snippet": snippet,
        "url": url,
        "score": score,
    }


class TestContextRelevanceEngine:
    pytestmark = pytest.mark.asyncio

    async def test_filter_memories_empty(self):
        engine = ContextRelevanceEngine()
        result = await engine.filter_memories("hello", [])
        assert result.total_before == 0
        assert result.total_after == 0
        assert len(result.relevant) == 0
        assert len(result.discarded) == 0

    async def test_filter_memories_all_relevant(self):
        engine = ContextRelevanceEngine()
        memories = [
            _make_memory("The user is building a project called Aetheris", score=0.85, category="project"),
            _make_memory("The user prefers Python for programming", score=0.80, category="programming"),
        ]
        result = await engine.filter_memories("What project am I building?", memories)
        assert result.total_after > 0

    async def test_filter_memories_all_irrelevant(self):
        engine = ContextRelevanceEngine()
        memories = [
            _make_memory("The user likes pizza", score=0.12, category="food"),
            _make_memory("The user enjoys swimming", score=0.08, category="hobby"),
        ]
        result = await engine.filter_memories("What is a kingdom?", memories)
        assert result.total_after == 0

    async def test_filter_memories_mixed(self):
        engine = ContextRelevanceEngine()
        memories = [
            _make_memory("The user is building a project called Aetheris", score=0.85, category="project"),
            _make_memory("The user likes pizza", score=0.12, category="food"),
            _make_memory("The user prefers Python for programming", score=0.80, category="programming"),
        ]
        result = await engine.filter_memories("What project am I building?", memories)
        assert len(result.relevant) <= 5
        assert result.total_before == 3

    async def test_filter_memories_with_embedding(self):
        engine = ContextRelevanceEngine(embedding_service=FakeEmbeddingService())
        memories = [
            _make_memory("The user is building a project called Aetheris", score=0.85, category="project"),
            _make_memory("The user likes pizza", score=0.12, category="food"),
        ]
        result = await engine.filter_memories("What project am I building?", memories)
        relevant_ids = {m["id"] for m in result.relevant}
        assert memories[0]["id"] in relevant_ids

    async def test_filter_memories_respects_max(self):
        engine = ContextRelevanceEngine()
        memories = [_make_memory(f"Memory {i}", score=0.9, category="general") for i in range(20)]
        result = await engine.filter_memories("test query", memories)
        assert result.total_after <= 5

    async def test_filter_conversation_empty(self):
        engine = ContextRelevanceEngine()
        result = await engine.filter_conversation("hello", [])
        assert result.total_before == 0
        assert result.total_after == 0

    async def test_filter_conversation_relevant(self):
        engine = ContextRelevanceEngine()
        messages = [
            _make_message("What is the capital of France?"),
            _make_message("The capital of France is Paris."),
            _make_message("I like pizza."),
            _make_message("Pizza is great!"),
        ]
        result = await engine.filter_conversation("What is the capital of France?", messages)
        assert result.total_after > 0

    async def test_filter_conversation_irrelevant(self):
        engine = ContextRelevanceEngine()
        messages = [
            _make_message("I like swimming."),
            _make_message("Swimming is great exercise."),
        ]
        result = await engine.filter_conversation("What is a kingdom?", messages)
        assert result.total_after == 0

    async def test_filter_conversation_respects_max(self):
        engine = ContextRelevanceEngine()
        messages = [_make_message(f"Message {i}") for i in range(20)]
        result = await engine.filter_conversation("test query", messages)
        assert result.total_after <= 3

    async def test_filter_search_results_empty(self):
        engine = ContextRelevanceEngine()
        result = await engine.filter_search_results("hello", [])
        assert result.total_before == 0

    async def test_filter_search_results(self):
        engine = ContextRelevanceEngine()
        results = [
            _make_search_result("Python Programming", "Learn Python programming language", score=0.9),
            _make_search_result("Pizza Recipes", "How to make pizza at home", score=0.1),
        ]
        result = await engine.filter_search_results("Learn Python", results)
        assert result.total_after > 0

    async def test_filter_search_results_all_low(self):
        engine = ContextRelevanceEngine()
        results = [
            _make_search_result("Pizza Recipes", "How to make pizza at home", score=0.1),
            _make_search_result("Swimming Tips", "Best swimming techniques", score=0.08),
        ]
        result = await engine.filter_search_results("Learn Python", results)
        assert result.total_after == 0


class TestKeywordSimilarity:
    def test_same_text(self):
        assert _keyword_similarity("hello world", "hello world") > 0.9

    def test_different_text(self):
        sim = _keyword_similarity("hello world", "goodbye python")
        assert sim < 0.1

    def test_partial_overlap(self):
        sim = _keyword_similarity("python programming", "python is fun")
        assert 0.1 < sim < 0.9

    def test_empty_a(self):
        assert _keyword_similarity("", "hello world") == 0.0

    def test_empty_both(self):
        assert _keyword_similarity("", "") == 0.0


class TestClassifyQueryTopic:
    def test_general_knowledge(self):
        assert _classify_query_topic("What is a kingdom?") == "general_knowledge"

    def test_programming(self):
        assert _classify_query_topic("How to write a Python function") == "programming"

    def test_personal(self):
        assert _classify_query_topic("What is my name?") == "personal"

    def test_project(self):
        assert _classify_query_topic("What project am I building?") == "project"

    def test_math(self):
        assert _classify_query_topic("Solve 2+2") == "math"
        assert _classify_query_topic("Calculate 5*10") == "math"

    def test_fallback(self):
        assert _classify_query_topic("Hello, how are you?") == "general"


class TestTopicAlignment:
    def test_programming_query_programming_memory(self):
        score = _topic_alignment("How to code in Python", "", {"category": "programming"})
        assert score >= 0.80

    def test_general_knowledge_query_personal_memory(self):
        score = _topic_alignment("What is a kingdom?", "", {"category": "preference"})
        assert score < 0.50

    def test_project_query_project_memory(self):
        score = _topic_alignment("What project am I building?", "", {"category": "project"})
        assert score >= 0.80

    def test_math_query_food_memory(self):
        score = _topic_alignment("Calculate 5+3", "", {"category": "food"})
        assert score == 0.0

    def test_personal_query_preference_memory(self):
        score = _topic_alignment("What is my favorite color?", "", {"category": "preference"})
        assert score >= 0.70


class TestEndToEndScenarios:
    pytestmark = pytest.mark.asyncio

    async def test_what_is_a_kingdom(self):
        engine = ContextRelevanceEngine()
        memories = [
            _make_memory("The user prefers Python for programming", score=0.40, category="programming"),
            _make_memory("The user is building a project called Aetheris", score=0.35, category="project"),
            _make_memory("The user's name is John", score=0.80, category="preference"),
        ]
        result = await engine.filter_memories("What is a kingdom?", memories)
        assert result.total_after == 0

    async def test_what_project_am_i_building(self):
        engine = ContextRelevanceEngine()
        memories = [
            _make_memory("The user prefers Python for programming", score=0.40, category="programming"),
            _make_memory("The user is building a project called Aetheris", score=0.85, category="project"),
            _make_memory("The user's name is John", score=0.20, category="preference"),
        ]
        result = await engine.filter_memories("What project am I building?", memories)
        assert result.total_after > 0
        relevant_docs = [m["document"] for m in result.relevant]
        assert any("Aetheris" in d for d in relevant_docs)

    async def test_what_programming_language(self):
        engine = ContextRelevanceEngine()
        memories = [
            _make_memory("The user prefers Python for programming", score=0.85, category="programming"),
            _make_memory("The user is building a project called Aetheris", score=0.40, category="project"),
            _make_memory("The user likes pizza", score=0.10, category="food"),
        ]
        result = await engine.filter_memories("What programming language do I prefer?", memories)
        assert result.total_after > 0
        relevant_docs = [m["document"] for m in result.relevant]
        assert any("Python" in d for d in relevant_docs)

    async def test_what_is_machine_learning(self):
        engine = ContextRelevanceEngine()
        memories = [
            _make_memory("The user prefers Python for programming", score=0.40, category="programming"),
            _make_memory("The user's name is John", score=0.80, category="preference"),
            _make_memory("The user likes pizza", score=0.30, category="food"),
        ]
        result = await engine.filter_memories("What is machine learning?", memories)
        assert result.total_after == 0

    async def test_conversation_topic_switching(self):
        engine = ContextRelevanceEngine()
        messages = [
            _make_message("Tell me about Gandhi"),
            _make_message("Gandhi was a leader of India's independence movement"),
            _make_message("What is the capital of France?"),
            _make_message("The capital of France is Paris"),
            _make_message("Can you explain quantum physics?"),
            _make_message("Quantum physics deals with atomic particles"),
        ]
        result_current = await engine.filter_conversation("What is a kingdom?", messages)
        assert result_current.total_after == 0

        result_france = await engine.filter_conversation("What is the capital of France?", messages)
        assert result_france.total_after > 0


def run_tests():
    test_classes = [
        TestContextRelevanceEngine,
        TestKeywordSimilarity,
        TestClassifyQueryTopic,
        TestTopicAlignment,
        TestEndToEndScenarios,
    ]

    passed = 0
    failed = 0

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(cls) if m.startswith("test_")]
        for method_name in methods:
            method = getattr(instance, method_name)
            is_coro = asyncio.iscoroutinefunction(getattr(cls, method_name))
            try:
                if is_coro:
                    asyncio.run(method())
                else:
                    method()
                print(f"  PASS  {cls.__name__}.{method_name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {cls.__name__}.{method_name}: {e}")
                failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
