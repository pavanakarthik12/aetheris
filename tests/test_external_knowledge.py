"""Tests for the External Knowledge Provider layer (Phase 11.1)."""

from __future__ import annotations

import sys
from pathlib import Path

import unittest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.schemas.external_knowledge import SearchResult, SearchResponse
from backend.app.services.external_knowledge.base_provider import ExternalKnowledgeProvider
from backend.app.services.external_knowledge.provider_manager import ExternalKnowledgeManager
from backend.app.services.external_knowledge.tavily_provider import TavilyProvider


class SearchResultSchemaTests(unittest.TestCase):
    def test_search_result_defaults(self) -> None:
        r = SearchResult()
        self.assertEqual(r.title, "")
        self.assertEqual(r.url, "")
        self.assertEqual(r.snippet, "")
        self.assertEqual(r.content, "")
        self.assertEqual(r.source, "")
        self.assertEqual(r.score, 0.0)

    def test_search_result_with_values(self) -> None:
        r = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="A snippet",
            content="Full content",
            source="tavily",
            score=0.95,
        )
        self.assertEqual(r.title, "Test Title")
        self.assertEqual(r.url, "https://example.com")
        self.assertEqual(r.score, 0.95)


class SearchResponseSchemaTests(unittest.TestCase):
    def test_search_response_defaults(self) -> None:
        resp = SearchResponse()
        self.assertEqual(len(resp.results), 0)
        self.assertEqual(resp.answer, "")
        self.assertEqual(resp.total_results, 0)
        self.assertEqual(resp.provider, "")

    def test_search_response_with_results(self) -> None:
        results = [SearchResult(title="A"), SearchResult(title="B")]
        resp = SearchResponse(
            results=results,
            answer="Test answer",
            total_results=2,
            duration_ms=100.0,
            provider="tavily",
        )
        self.assertEqual(len(resp.results), 2)
        self.assertEqual(resp.answer, "Test answer")
        self.assertEqual(resp.provider, "tavily")


class ExternalKnowledgeProviderInterfaceTests(unittest.TestCase):
    def test_interface_cannot_be_instantiated(self) -> None:
        with self.assertRaises(TypeError):
            ExternalKnowledgeProvider()  # type: ignore


class TavilyProviderTests(unittest.TestCase):
    def test_provider_name(self) -> None:
        provider = TavilyProvider(api_key="test-key")
        self.assertEqual(provider.provider_name, "tavily")

    def test_is_available_with_key(self) -> None:
        provider = TavilyProvider(api_key="test-key")
        self.assertTrue(provider.is_available())

    def test_is_available_without_key(self) -> None:
        provider = TavilyProvider(api_key="")
        self.assertFalse(provider.is_available())

    def test_health_check_without_key(self) -> None:
        import asyncio
        provider = TavilyProvider(api_key="")
        result = asyncio.run(provider.health_check())
        self.assertFalse(result)

    def test_search_without_key_returns_empty(self) -> None:
        import asyncio
        provider = TavilyProvider(api_key="")
        result = asyncio.run(provider.search("test query"))
        self.assertEqual(len(result.results), 0)
        self.assertEqual(result.provider, "tavily")


class ExternalKnowledgeManagerTests(unittest.TestCase):
    def test_manager_no_key_disabled(self) -> None:
        manager = ExternalKnowledgeManager(tavily_api_key="")
        self.assertIsNone(manager.active_provider)
        self.assertEqual(manager.active_provider_name, "none")
        self.assertFalse(manager.is_available())
        self.assertIsNotNone(manager.init_error)
        self.assertIn("TAVILY_API_KEY", manager.init_error)

    def test_manager_with_key_initializes(self) -> None:
        manager = ExternalKnowledgeManager(tavily_api_key="test-key")
        self.assertIsNotNone(manager.active_provider)
        self.assertEqual(manager.active_provider_name, "tavily")
        self.assertTrue(manager.is_available())
        self.assertTrue(manager.is_initialized)
        self.assertIn("tavily", manager.available_providers)

    def test_health_check_no_provider(self) -> None:
        import asyncio
        manager = ExternalKnowledgeManager(tavily_api_key="")
        result = asyncio.run(manager.health_check())
        self.assertFalse(result)

    def test_search_without_provider_returns_empty(self) -> None:
        import asyncio
        manager = ExternalKnowledgeManager(tavily_api_key="")
        result = asyncio.run(manager.search("test"))
        self.assertEqual(len(result.results), 0)
        self.assertEqual(result.provider, "none")

    def test_snapshot_no_provider(self) -> None:
        manager = ExternalKnowledgeManager(tavily_api_key="")
        snap = manager.snapshot()
        self.assertFalse(snap["is_available"])
        self.assertEqual(snap["active_provider"], "none")
        self.assertEqual(snap["available_providers"], [])
        self.assertIsNotNone(snap["init_error"])

    def test_snapshot_with_provider(self) -> None:
        manager = ExternalKnowledgeManager(tavily_api_key="test-key")
        snap = manager.snapshot()
        self.assertTrue(snap["is_available"])
        self.assertEqual(snap["active_provider"], "tavily")
        self.assertIn("tavily", snap["available_providers"])
        self.assertIsNone(snap["init_error"])


if __name__ == "__main__":
    unittest.main()
