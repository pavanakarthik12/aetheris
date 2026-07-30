"""Tests for the Context Intelligence Engine (Phase 12)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.schemas.context_intelligence import ScoredResultItem, StructuredContext, TrustConfig
from backend.app.schemas.external_knowledge import NormalizedSearchResultItem
from backend.app.services.external_knowledge.context_intelligence import ContextIntelligenceEngine
from backend.app.services.external_knowledge.context_formatter import ExternalKnowledgeContextFormatter


def _make_item(
    title: str = "Test Result",
    snippet: str = "This is a test snippet with useful content for the query.",
    url: str = "https://example.com/page",
    score: float = 0.7,
    source: str = "tavily",
) -> NormalizedSearchResultItem:
    return NormalizedSearchResultItem(
        title=title,
        snippet=snippet,
        url=url,
        content="",
        source=source,
        score=score,
    )


class RemoveInvalidTests(TestCase):

    def setUp(self) -> None:
        self.engine = ContextIntelligenceEngine()

    def test_keeps_valid_results(self) -> None:
        items = [_make_item()]
        valid, filtered = self.engine._remove_invalid(items)
        self.assertEqual(len(valid), 1)
        self.assertEqual(filtered, 0)

    def test_removes_empty_title_and_snippet(self) -> None:
        items = [_make_item(title="", snippet="")]
        valid, filtered = self.engine._remove_invalid(items)
        self.assertEqual(len(valid), 0)
        self.assertEqual(filtered, 1)

    def test_removes_missing_url(self) -> None:
        items = [_make_item(url="")]
        valid, filtered = self.engine._remove_invalid(items)
        self.assertEqual(len(valid), 0)
        self.assertEqual(filtered, 1)

    def test_removes_low_confidence(self) -> None:
        items = [_make_item(score=0.01)]
        valid, filtered = self.engine._remove_invalid(items)
        self.assertEqual(len(valid), 0)
        self.assertEqual(filtered, 1)

    def test_keeps_item_with_only_snippet(self) -> None:
        items = [_make_item(title="", snippet="A valid snippet here")]
        valid, filtered = self.engine._remove_invalid(items)
        self.assertEqual(len(valid), 1)

    def test_keeps_item_with_only_title(self) -> None:
        items = [_make_item(title="Valid Title", snippet="")]
        valid, filtered = self.engine._remove_invalid(items)
        self.assertEqual(len(valid), 1)

    def test_removes_too_short_content(self) -> None:
        items = [_make_item(title="A", snippet="short")]
        valid, filtered = self.engine._remove_invalid(items)
        self.assertEqual(len(valid), 0)

    def test_extracts_domain(self) -> None:
        items = [_make_item(url="https://docs.python.org/3/tutorial/")]
        valid, filtered = self.engine._remove_invalid(items)
        self.assertEqual(valid[0].domain, "docs.python.org")


class DuplicateRemovalTests(TestCase):

    def setUp(self) -> None:
        self.engine = ContextIntelligenceEngine()

    def _to_scored(self, items: list[NormalizedSearchResultItem]) -> list[ScoredResultItem]:
        valid, _ = self.engine._remove_invalid(items)
        return valid

    def test_removes_duplicate_urls(self) -> None:
        items = [
            _make_item(url="https://example.com/page"),
            _make_item(title="Second", url="https://example.com/page"),
        ]
        scored = self._to_scored(items)
        deduped, removed = self.engine._remove_duplicates(scored)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(removed, 1)

    def test_removes_duplicate_titles(self) -> None:
        items = [
            _make_item(title="Same Title", url="https://a.com"),
            _make_item(title="Same Title", url="https://b.com"),
        ]
        scored = self._to_scored(items)
        deduped, removed = self.engine._remove_duplicates(scored)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(removed, 1)

    def test_keeps_different_results(self) -> None:
        items = [
            _make_item(title="Result A", url="https://a.com", snippet="Snippet A"),
            _make_item(title="Result B", url="https://b.com", snippet="Snippet B"),
        ]
        scored = self._to_scored(items)
        deduped, removed = self.engine._remove_duplicates(scored)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(removed, 0)

    def test_marks_duplicate_flag(self) -> None:
        items = [
            _make_item(title="Same", url="https://a.com"),
            _make_item(title="Same", url="https://b.com"),
        ]
        scored = self._to_scored(items)
        deduped, _ = self.engine._remove_duplicates(scored)
        self.assertFalse(deduped[0].is_duplicate)


class TrustScoringTests(TestCase):

    def setUp(self) -> None:
        self.engine = ContextIntelligenceEngine()

    def test_official_docs_high_trust(self) -> None:
        item = ScoredResultItem(
            title="Python Docs",
            snippet="Official documentation",
            url="https://docs.python.org/3/",
            domain="docs.python.org",
            confidence=0.9,
        )
        score = self.engine._score_trust(item)
        self.assertGreaterEqual(score, 0.9)

    def test_github_high_trust(self) -> None:
        item = ScoredResultItem(
            title="Repo",
            snippet="GitHub repository",
            url="https://github.com/fastapi/fastapi",
            domain="github.com",
            confidence=0.8,
        )
        score = self.engine._score_trust(item)
        self.assertGreaterEqual(score, 0.8)

    def test_wikipedia_high_trust(self) -> None:
        item = ScoredResultItem(
            title="Wikipedia Article",
            snippet="Encyclopedia entry",
            url="https://en.wikipedia.org/wiki/Python",
            domain="en.wikipedia.org",
            confidence=0.7,
        )
        score = self.engine._score_trust(item)
        self.assertGreaterEqual(score, 0.85)

    def test_medium_lower_trust(self) -> None:
        item = ScoredResultItem(
            title="Blog Post",
            snippet="A blog post about Python",
            url="https://medium.com/some-post",
            domain="medium.com",
            confidence=0.6,
        )
        score = self.engine._score_trust(item)
        self.assertLessEqual(score, 0.5)

    def test_unknown_domain_default_trust(self) -> None:
        item = ScoredResultItem(
            title="Unknown",
            snippet="Some content",
            url="https://random-blog.example.com/post",
            domain="random-blog.example.com",
            confidence=0.5,
        )
        score = self.engine._score_trust(item)
        self.assertAlmostEqual(score, 0.5, delta=0.2)

    def test_clickbait_penalty(self) -> None:
        item = ScoredResultItem(
            title="You won't believe this shocking discovery!",
            snippet="This will blow your mind",
            url="https://clickbait.example.com",
            domain="clickbait.example.com",
            confidence=0.7,
        )
        score = self.engine._score_trust(item)
        score = self.engine._score_trust(item)
        self.assertLess(score, 0.4)

    def test_gov_domain_high_trust(self) -> None:
        item = ScoredResultItem(
            title="Government Site",
            snippet="Official government information",
            url="https://www.nasa.gov/mission",
            domain="www.nasa.gov",
            confidence=0.8,
        )
        score = self.engine._score_trust(item)
        self.assertGreaterEqual(score, 0.9)

    def test_mdn_high_trust(self) -> None:
        item = ScoredResultItem(
            title="MDN Web Docs",
            snippet="Web technology documentation",
            url="https://developer.mozilla.org/en-US/",
            domain="developer.mozilla.org",
            confidence=0.85,
        )
        score = self.engine._score_trust(item)
        self.assertGreaterEqual(score, 0.9)


class RelevanceRankingTests(TestCase):

    def setUp(self) -> None:
        self.engine = ContextIntelligenceEngine()

    def test_ranks_by_relevance_score(self) -> None:
        items = [
            ScoredResultItem(title="Low", snippet="unrelated content", url="https://a.com", domain="a.com", confidence=0.3, trust_score=0.4),
            ScoredResultItem(title="High", snippet="fastapi framework documentation", url="https://b.com", domain="b.com", confidence=0.9, trust_score=0.9),
        ]
        ranked = self.engine._rank_by_relevance(items, "FastAPI")
        self.assertGreaterEqual(ranked[0].relevance_score, ranked[1].relevance_score)

    def test_official_docs_ranked_first_for_fastapi(self) -> None:
        items = [
            ScoredResultItem(title="Blog about FastAPI", snippet="A blog post", url="https://medium.com/fastapi", domain="medium.com", confidence=0.8, trust_score=0.4),
            ScoredResultItem(title="FastAPI Documentation", snippet="Official docs", url="https://fastapi.tiangolo.com", domain="fastapi.tiangolo.com", confidence=0.9, trust_score=0.99),
        ]
        ranked = self.engine._rank_by_relevance(items, "FastAPI")
        self.assertIn("fastapi.tiangolo.com", ranked[0].domain)

    def test_query_term_match_boosts_score(self) -> None:
        items = [
            ScoredResultItem(title="Unrelated", snippet="nothing here", url="https://a.com", domain="a.com", confidence=0.5, trust_score=0.5),
            ScoredResultItem(title="React Guide", snippet="Learn React JavaScript framework", url="https://b.com", domain="b.com", confidence=0.5, trust_score=0.5),
        ]
        ranked = self.engine._rank_by_relevance(items, "React")
        self.assertGreater(ranked[0].relevance_score, ranked[1].relevance_score)


class ContentCompressionTests(TestCase):

    def setUp(self) -> None:
        self.engine = ContextIntelligenceEngine()

    def test_removes_cookie_notice(self) -> None:
        compressed = self.engine._compress_text(
            "Real content about FastAPI.\n"
            "Accept all cookies.\n"
            "More useful technical details."
        )
        self.assertNotIn("Accept all cookies", compressed)
        self.assertIn("Real content about FastAPI", compressed)
        self.assertIn("More useful technical details", compressed)

    def test_removes_privacy_policy(self) -> None:
        compressed = self.engine._compress_text(
            "Useful technical information.\n"
            "Privacy policy applies.\n"
            "More helpful content."
        )
        self.assertNotIn("Privacy policy", compressed)
        self.assertIn("Useful technical information", compressed)
        self.assertIn("More helpful content", compressed)

    def test_removes_copyright(self) -> None:
        compressed = self.engine._compress_text("Tutorial content. \u00a9 2024 All rights reserved. More.")
        self.assertNotIn("All rights reserved", compressed)

    def test_preserves_clean_text(self) -> None:
        compressed = self.engine._compress_text("FastAPI is a modern web framework for building APIs with Python.")
        self.assertIn("FastAPI", compressed)
        self.assertIn("modern web framework", compressed)

    def test_normalizes_whitespace(self) -> None:
        compressed = self.engine._compress_text("Line1.\n\n\nLine2.   \n  Line3.")
        self.assertNotIn("\n\n\n", compressed)

    def test_no_text_returns_empty(self) -> None:
        self.assertEqual(self.engine._compress_text(""), "")
        self.assertEqual(self.engine._compress_text(None), "")


class TokenBudgetTests(TestCase):

    def setUp(self) -> None:
        config = TrustConfig(max_tokens=100)
        self.engine = ContextIntelligenceEngine(config=config)

    def test_stops_when_budget_exceeded(self) -> None:
        items = [
            ScoredResultItem(title="A" * 200, snippet="B" * 200, url="https://a.com", domain="a.com"),
            ScoredResultItem(title="C" * 200, snippet="D" * 200, url="https://b.com", domain="b.com"),
        ]
        capped, total = self.engine._enforce_token_budget(items)
        self.assertLessEqual(len(capped), 1)
        self.assertLessEqual(total, 100)

    def test_all_items_fit_in_budget(self) -> None:
        config = TrustConfig(max_tokens=5000)
        engine = ContextIntelligenceEngine(config=config)
        items = [
            ScoredResultItem(title="Short", snippet="Short", url="https://a.com", domain="a.com"),
            ScoredResultItem(title="Short", snippet="Short", url="https://b.com", domain="b.com"),
        ]
        capped, total = engine._enforce_token_budget(items)
        self.assertEqual(len(capped), 2)


class FullPipelineTests(TestCase):

    def test_latest_fastapi_ranks_official_first(self) -> None:
        engine = ContextIntelligenceEngine()
        results = [
            _make_item(
                title="Getting Started with FastAPI",
                snippet="A beginner tutorial on FastAPI framework",
                url="https://medium.com/fastapi-tutorial",
                score=0.7,
            ),
            _make_item(
                title="FastAPI Documentation",
                snippet="Official FastAPI framework documentation, high performance",
                url="https://fastapi.tiangolo.com",
                score=0.95,
            ),
            _make_item(
                title="FastAPI GitHub",
                snippet="Source code repository",
                url="https://github.com/fastapi/fastapi",
                score=0.85,
            ),
        ]
        ctx = engine.process("latest FastAPI version", results)
        self.assertGreater(len(ctx.results), 0)
        self.assertEqual(ctx.original_count, 3)
        self.assertIn("fastapi.tiangolo.com", ctx.results[0].domain)

    def test_duplicates_are_removed(self) -> None:
        engine = ContextIntelligenceEngine()
        results = [
            _make_item(title="Same Result", url="https://example.com/page1", snippet="Same content here"),
            _make_item(title="Same Result", url="https://example.com/page2", snippet="Same content here"),
            _make_item(title="Different", url="https://other.com", snippet="Unique content"),
        ]
        ctx = engine.process("test", results)
        self.assertEqual(ctx.sources_used, 2)
        self.assertEqual(ctx.duplicates_removed, 1)

    def test_invalid_results_are_filtered(self) -> None:
        engine = ContextIntelligenceEngine()
        results = [
            _make_item(title="", snippet="", url="", score=0.0),
            _make_item(title="Valid", snippet="Real content here", url="https://valid.com", score=0.8),
        ]
        ctx = engine.process("test", results)
        self.assertEqual(ctx.sources_used, 1)
        self.assertEqual(ctx.filtered_count, 1)
        self.assertEqual(ctx.original_count, 2)

    def test_empty_results_returns_empty_context(self) -> None:
        engine = ContextIntelligenceEngine()
        ctx = engine.process("test", [])
        self.assertEqual(ctx.sources_used, 0)
        self.assertEqual(ctx.original_count, 0)

    def test_context_has_metadata(self) -> None:
        engine = ContextIntelligenceEngine()
        results = [
            _make_item(title="Result A", url="https://a.com"),
            _make_item(title="Result B", url="https://b.com"),
        ]
        ctx = engine.process("test query", results)
        self.assertEqual(ctx.query, "test query")
        self.assertGreater(ctx.processing_time_ms, 0)
        self.assertGreater(ctx.estimated_tokens, 0)

    def test_python_docs_ranked_above_blogs(self) -> None:
        engine = ContextIntelligenceEngine()
        results = [
            _make_item(
                title="Python Tutorial Blog",
                snippet="A blog post about Python programming",
                url="https://dev.to/python-tutorial",
                score=0.8,
            ),
            _make_item(
                title="Python 3 Documentation",
                snippet="Official Python documentation and tutorial",
                url="https://docs.python.org/3/",
                score=0.9,
            ),
            _make_item(
                title="Learn Python",
                snippet="Python programming language tutorial for beginners",
                url="https://w3schools.com/python",
                score=0.7,
            ),
        ]
        ctx = engine.process("Python documentation", results)
        self.assertIn("docs.python.org", ctx.results[0].domain)

    def test_react_latest_ranks_official_first(self) -> None:
        engine = ContextIntelligenceEngine()
        results = [
            _make_item(
                title="React Blog Post",
                snippet="Some blog about React features",
                url="https://medium.com/react-post",
                score=0.7,
            ),
            _make_item(
                title="React Documentation",
                snippet="Official React documentation and reference",
                url="https://react.dev",
                score=0.95,
            ),
            _make_item(
                title="React GitHub Releases",
                snippet="React release notes on GitHub",
                url="https://github.com/facebook/react/releases",
                score=0.85,
            ),
        ]
        ctx = engine.process("React latest release", results)
        self.assertIn("react.dev", ctx.results[0].domain)


class ContextFormatterStructuredTests(TestCase):

    def setUp(self) -> None:
        self.formatter = ExternalKnowledgeContextFormatter()

    def test_formats_structured_context(self) -> None:
        ctx = StructuredContext(
            query="test",
            results=[
                ScoredResultItem(
                    title="Result A",
                    snippet="Snippet A",
                    url="https://a.com",
                    domain="a.com",
                ),
            ],
            original_count=5,
            duplicates_removed=2,
            filtered_count=2,
            sources_used=1,
            estimated_tokens=50,
        )
        text = self.formatter.format_results(structured_context=ctx)
        self.assertIn("External Knowledge:", text)
        self.assertIn("Result A", text)
        self.assertIn("Snippet A", text)
        self.assertIn("https://a.com", text)
        self.assertIn("Domain: a.com", text)
        self.assertIn("Filtered from 5 sources", text)
        self.assertIn("2 duplicates removed", text)

    def test_empty_structured_returns_empty(self) -> None:
        ctx = StructuredContext(query="test")
        text = self.formatter.format_results(structured_context=ctx)
        self.assertEqual(text, "")

    def test_backward_compatible_with_raw(self) -> None:
        items = [
            _make_item(title="Raw", snippet="Raw snippet", url="https://raw.com", score=0.9),
        ]
        text = self.formatter.format_results(results=items)
        self.assertIn("Raw", text)
        self.assertIn("Raw snippet", text)
