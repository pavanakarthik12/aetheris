"""Tests for the External Knowledge Search Pipeline (Phase 11.2)."""

from __future__ import annotations

import sys
from pathlib import Path

import unittest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.schemas.external_knowledge import (
    NormalizedSearchResultItem,
    SearchDebugResponse,
    SearchResult,
    SearchResponse,
)
from backend.app.services.external_knowledge.provider_manager import ExternalKnowledgeManager
from backend.app.services.external_knowledge.search_pipeline import SearchPipeline


class NormalizedResponseSchemaTests(unittest.TestCase):
    def test_normalized_item_defaults(self) -> None:
        item = NormalizedSearchResultItem()
        self.assertEqual(item.title, "")
        self.assertEqual(item.url, "")
        self.assertEqual(item.score, 0.0)

    def test_normalized_item_with_values(self) -> None:
        item = NormalizedSearchResultItem(
            title="Test",
            url="https://example.com",
            snippet="snippet",
            content="content",
            source="tavily",
            score=0.95,
        )
        self.assertEqual(item.title, "Test")
        self.assertEqual(item.score, 0.95)

    def test_debug_response_defaults(self) -> None:
        resp = SearchDebugResponse()
        self.assertFalse(resp.success)
        self.assertEqual(resp.result_count, 0)
        self.assertEqual(len(resp.results), 0)
        self.assertIsNone(resp.error)

    def test_debug_response_with_results(self) -> None:
        results = [NormalizedSearchResultItem(title="A"), NormalizedSearchResultItem(title="B")]
        resp = SearchDebugResponse(
            provider="tavily",
            query="test",
            success=True,
            execution_time_ms=100.0,
            result_count=2,
            results=results,
        )
        self.assertTrue(resp.success)
        self.assertEqual(len(resp.results), 2)
        self.assertEqual(resp.provider, "tavily")

    def test_debug_response_error(self) -> None:
        resp = SearchDebugResponse(
            success=False,
            error="Missing API Key",
        )
        self.assertFalse(resp.success)
        self.assertEqual(resp.error, "Missing API Key")


class FakeExternalKnowledgeManager(ExternalKnowledgeManager):
    def __init__(self, available: bool = True, fail_search: bool = False) -> None:
        self._fake_available = available
        self._fail_search = fail_search
        self._initialized = True
        self._init_error = None if available else "TAVILY_API_KEY is not configured"
        self._active_provider = None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def init_error(self) -> str | None:
        return self._init_error

    @property
    def active_provider(self) -> None:
        return None

    @property
    def active_provider_name(self) -> str:
        return "tavily" if self._fake_available else "none"

    @property
    def available_providers(self) -> list[str]:
        return ["tavily"] if self._fake_available else []

    def is_available(self) -> bool:
        return self._fake_available

    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        if self._fail_search:
            raise RuntimeError("Provider connection failed")
        return SearchResponse(
            results=[
                SearchResult(
                    title="Python 3.13 Released",
                    url="https://python.org",
                    snippet="Python 3.13 is the latest version.",
                    content="Full content here.",
                    source="tavily",
                    score=0.95,
                ),
                SearchResult(
                    title="FastAPI Documentation",
                    url="https://fastapi.tiangolo.com",
                    snippet="FastAPI is a modern web framework.",
                    content="Full content.",
                    source="tavily",
                    score=0.89,
                ),
            ],
            answer="Python 3.13 and FastAPI are popular.",
            total_results=2,
            duration_ms=500.0,
            provider="tavily",
        )


class SearchPipelineTests(unittest.TestCase):
    def test_empty_query_returns_validation_error(self) -> None:
        import asyncio
        manager = FakeExternalKnowledgeManager(available=True)
        pipeline = SearchPipeline(manager)
        result = asyncio.run(pipeline.execute(""))
        self.assertFalse(result.success)
        self.assertIn("empty", result.error.lower())

    def test_whitespace_query_returns_validation_error(self) -> None:
        import asyncio
        manager = FakeExternalKnowledgeManager(available=True)
        pipeline = SearchPipeline(manager)
        result = asyncio.run(pipeline.execute("   "))
        self.assertFalse(result.success)
        self.assertIn("empty", result.error.lower())

    def test_long_query_returns_validation_error(self) -> None:
        import asyncio
        manager = FakeExternalKnowledgeManager(available=True)
        pipeline = SearchPipeline(manager)
        result = asyncio.run(pipeline.execute("x" * 501))
        self.assertFalse(result.success)
        self.assertIn("exceeds", result.error.lower())

    def test_missing_api_key_returns_error(self) -> None:
        import asyncio
        manager = FakeExternalKnowledgeManager(available=False)
        pipeline = SearchPipeline(manager)
        result = asyncio.run(pipeline.execute("test query"))
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Missing API Key")

    def test_successful_search_returns_normalized_results(self) -> None:
        import asyncio
        manager = FakeExternalKnowledgeManager(available=True)
        pipeline = SearchPipeline(manager)
        result = asyncio.run(pipeline.execute("Python version"))
        self.assertTrue(result.success)
        self.assertEqual(result.result_count, 2)
        self.assertEqual(result.provider, "tavily")
        self.assertEqual(len(result.results), 2)
        self.assertEqual(result.results[0].title, "Python 3.13 Released")
        self.assertEqual(result.results[1].url, "https://fastapi.tiangolo.com")
        self.assertGreater(result.execution_time_ms, 0)

    def test_search_failure_returns_error(self) -> None:
        import asyncio
        manager = FakeExternalKnowledgeManager(available=True, fail_search=True)
        pipeline = SearchPipeline(manager)
        result = asyncio.run(pipeline.execute("test query"))
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_result_items_have_all_fields(self) -> None:
        import asyncio
        manager = FakeExternalKnowledgeManager(available=True)
        pipeline = SearchPipeline(manager)
        result = asyncio.run(pipeline.execute("test"))
        item = result.results[0]
        self.assertTrue(all([item.title, item.url, item.snippet, item.source]))
        self.assertGreater(item.score, 0.0)


if __name__ == "__main__":
    unittest.main()
