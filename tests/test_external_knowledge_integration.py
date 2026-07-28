"""Tests for External Knowledge Integration (Phase 11.3).

Decision layer, context formatter, integration service,
and reasoning pipeline integration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.schemas.external_knowledge import (
    NormalizedSearchResultItem,
    SearchDebugResponse,
)
from backend.app.schemas.reasoning import (
    CognitiveTrace,
    ConfidenceLevel,
    ReasoningPlan,
    SemanticIntentType,
)
from backend.app.services.external_knowledge.context_formatter import (
    ExternalKnowledgeContextFormatter,
)
from backend.app.services.external_knowledge.decision_layer import (
    ExternalKnowledgeDecisionLayer,
)
from backend.app.services.external_knowledge.integration_service import (
    ExternalKnowledgeIntegrationService,
    ExternalKnowledgeResult,
)
from backend.app.services.reasoning.pipeline import ReasoningPipeline


class DecisionLayerTests(TestCase):

    def setUp(self) -> None:
        self.layer = ExternalKnowledgeDecisionLayer()

    def test_skip_greeting(self) -> None:
        self.assertFalse(
            self.layer.needs_external_knowledge("hello", SemanticIntentType.GREETING),
        )

    def test_skip_simple_question(self) -> None:
        self.assertFalse(
            self.layer.needs_external_knowledge("what is 2+2", SemanticIntentType.SIMPLE_QUESTION),
        )

    def test_skip_memory_retrieval(self) -> None:
        self.assertFalse(
            self.layer.needs_external_knowledge("what do you know about me", SemanticIntentType.MEMORY_RETRIEVAL),
        )

    def test_skip_planning(self) -> None:
        self.assertFalse(
            self.layer.needs_external_knowledge("plan a trip", SemanticIntentType.PLANNING),
        )

    def test_skip_code_generation(self) -> None:
        self.assertFalse(
            self.layer.needs_external_knowledge("write a python script", SemanticIntentType.CODE_GENERATION),
        )

    def test_skip_unknown(self) -> None:
        self.assertFalse(
            self.layer.needs_external_knowledge("xyz", SemanticIntentType.UNKNOWN),
        )

    def test_trigger_latest_keyword(self) -> None:
        self.assertTrue(
            self.layer.needs_external_knowledge(
                "latest FastAPI version",
                SemanticIntentType.GENERAL_CONVERSATION,
            ),
        )

    def test_trigger_current_keyword(self) -> None:
        self.assertTrue(
            self.layer.needs_external_knowledge(
                "current Java LTS version",
                SemanticIntentType.PROGRAMMING,
            ),
        )

    def test_trigger_weather(self) -> None:
        self.assertTrue(
            self.layer.needs_external_knowledge(
                "today's weather in Hyderabad",
                SemanticIntentType.GENERAL_CONVERSATION,
            ),
        )

    def test_trigger_news(self) -> None:
        self.assertTrue(
            self.layer.needs_external_knowledge(
                "OpenAI latest news",
                SemanticIntentType.GENERAL_CONVERSATION,
            ),
        )

    def test_no_keyword_no_trigger(self) -> None:
        self.assertFalse(
            self.layer.needs_external_knowledge(
                "explain binary search",
                SemanticIntentType.EXPLANATION,
            ),
        )

    def test_no_keyword_programming_explanation(self) -> None:
        self.assertFalse(
            self.layer.needs_external_knowledge(
                "what is polymorphism",
                SemanticIntentType.PROGRAMMING,
            ),
        )

    def test_trigger_release_keyword(self) -> None:
        self.assertTrue(
            self.layer.needs_external_knowledge(
                "React latest release",
                SemanticIntentType.GENERAL_CONVERSATION,
            ),
        )

    def test_trigger_stock_keyword(self) -> None:
        self.assertTrue(
            self.layer.needs_external_knowledge(
                "NVIDIA stock price",
                SemanticIntentType.GENERAL_CONVERSATION,
            ),
        )

    def test_not_triggered_by_memory_intents(self) -> None:
        self.assertFalse(
            self.layer.needs_external_knowledge(
                "my latest memory", SemanticIntentType.MEMORY_RETRIEVAL,
            ),
        )


class ContextFormatterTests(TestCase):

    def setUp(self) -> None:
        self.formatter = ExternalKnowledgeContextFormatter()

    def test_empty_results(self) -> None:
        result = self.formatter.format_results([])
        self.assertEqual(result, "")

    def test_single_result(self) -> None:
        items = [
            NormalizedSearchResultItem(
                title="FastAPI Docs",
                url="https://fastapi.tiangolo.com",
                snippet="FastAPI framework, high performance",
                content="Full content here",
                source="tavily",
                score=0.95,
            ),
        ]
        result = self.formatter.format_results(items)
        self.assertIn("Source 1", result)
        self.assertIn("Title: FastAPI Docs", result)
        self.assertIn("Snippet: FastAPI framework, high performance", result)
        self.assertIn("URL: https://fastapi.tiangolo.com", result)
        self.assertIn("External Knowledge:", result)

    def test_multiple_results(self) -> None:
        items = [
            NormalizedSearchResultItem(title="Result A", url="http://a.com", snippet="Snippet A"),
            NormalizedSearchResultItem(title="Result B", url="http://b.com", snippet="Snippet B"),
        ]
        result = self.formatter.format_results(items)
        self.assertIn("Source 1", result)
        self.assertIn("Source 2", result)
        self.assertIn("---", result)

    def test_result_without_url(self) -> None:
        items = [
            NormalizedSearchResultItem(title="No URL", snippet="Just text"),
        ]
        result = self.formatter.format_results(items)
        self.assertIn("Title: No URL", result)
        self.assertIn("Snippet: Just text", result)
        self.assertNotIn("URL:", result)

    def test_instruction_contains_priority(self) -> None:
        instruction = self.formatter.format_instruction()
        self.assertIn("prioritize the retrieved information", instruction)
        self.assertIn("admit uncertainty", instruction)
        self.assertIn("Never fabricate facts", instruction)

    def test_instruction_contains_combine(self) -> None:
        instruction = self.formatter.format_instruction()
        self.assertIn("Combine it with your own reasoning", instruction)


class IntegrationServiceTests(IsolatedAsyncioTestCase):

    def setUp(self) -> None:
        self.decision = MagicMock()
        self.pipeline = AsyncMock()
        self.formatter = MagicMock()
        self.service = ExternalKnowledgeIntegrationService(
            decision_layer=self.decision,
            search_pipeline=self.pipeline,
            formatter=self.formatter,
        )

    async def test_not_triggered_when_decision_says_false(self) -> None:
        self.decision.needs_external_knowledge.return_value = False
        result = await self.service.execute("hello", SemanticIntentType.GREETING)
        self.assertFalse(result.triggered)
        self.assertFalse(result.success)
        self.pipeline.execute.assert_not_called()

    async def test_triggered_search_success(self) -> None:
        self.decision.needs_external_knowledge.return_value = True
        self.pipeline.execute.return_value = SearchDebugResponse(
            provider="tavily",
            query="latest FastAPI version",
            success=True,
            result_count=3,
            results=[
                NormalizedSearchResultItem(title="R1", url="http://a.com", snippet="S1"),
                NormalizedSearchResultItem(title="R2", url="http://b.com", snippet="S2"),
                NormalizedSearchResultItem(title="R3", url="http://c.com", snippet="S3"),
            ],
        )
        self.formatter.format_results.return_value = "External Knowledge:\n\nSource 1..."
        self.formatter.format_instruction.return_value = "Prioritize retrieved info."

        result = await self.service.execute("latest FastAPI version", SemanticIntentType.GENERAL_CONVERSATION)

        self.assertTrue(result.triggered)
        self.assertTrue(result.success)
        self.assertEqual(result.result_count, 3)
        self.assertEqual(result.context_block, "External Knowledge:\n\nSource 1...")
        self.assertEqual(result.instruction, "Prioritize retrieved info.")
        self.assertIsNone(result.error)

    async def test_triggered_search_returns_no_results(self) -> None:
        self.decision.needs_external_knowledge.return_value = True
        self.pipeline.execute.return_value = SearchDebugResponse(
            provider="tavily",
            query="test",
            success=True,
            result_count=0,
            results=[],
        )

        result = await self.service.execute("test", SemanticIntentType.GENERAL_CONVERSATION)

        self.assertTrue(result.triggered)
        self.assertFalse(result.success)
        self.assertEqual(result.result_count, 0)
        self.assertEqual(result.context_block, "")

    async def test_triggered_search_failure(self) -> None:
        self.decision.needs_external_knowledge.return_value = True
        self.pipeline.execute.return_value = SearchDebugResponse(
            provider="tavily",
            query="test",
            success=False,
            result_count=0,
            results=[],
            error="API error",
        )

        result = await self.service.execute("test", SemanticIntentType.GENERAL_CONVERSATION)

        self.assertTrue(result.triggered)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    async def test_triggered_search_raises_exception(self) -> None:
        self.decision.needs_external_knowledge.return_value = True
        self.pipeline.execute.side_effect = RuntimeError("Unexpected error")

        result = await self.service.execute("test", SemanticIntentType.GENERAL_CONVERSATION)

        self.assertTrue(result.triggered)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)


class ReasoningPipelineExternalKnowledgeTests(IsolatedAsyncioTestCase):

    def setUp(self) -> None:
        self.pipeline = ReasoningPipeline()

    async def test_latest_version_triggers_external_knowledge(self) -> None:
        plan, trace = await self.pipeline.run("latest FastAPI version")
        self.assertTrue(plan.needs_external_knowledge)
        self.assertTrue(trace.needs_external_knowledge)

    async def test_explain_binary_search_no_external_knowledge(self) -> None:
        plan, trace = await self.pipeline.run("explain binary search")
        self.assertFalse(plan.needs_external_knowledge)
        self.assertFalse(trace.needs_external_knowledge)

    async def test_weather_triggers_external_knowledge(self) -> None:
        plan, trace = await self.pipeline.run("today's weather")
        self.assertTrue(plan.needs_external_knowledge)
        self.assertTrue(trace.needs_external_knowledge)

    async def test_polymorphism_no_external_knowledge(self) -> None:
        plan, trace = await self.pipeline.run("what is polymorphism")
        self.assertFalse(plan.needs_external_knowledge)
        self.assertFalse(trace.needs_external_knowledge)

    async def test_openai_news_triggers_external_knowledge(self) -> None:
        plan, trace = await self.pipeline.run("OpenAI latest news")
        self.assertTrue(plan.needs_external_knowledge)
        self.assertTrue(trace.needs_external_knowledge)

    async def test_programming_explanation_no_external_knowledge(self) -> None:
        plan, trace = await self.pipeline.run("reverse an array")
        self.assertFalse(plan.needs_external_knowledge)
        self.assertFalse(trace.needs_external_knowledge)

    async def test_release_keyword_triggers_external_knowledge(self) -> None:
        plan, trace = await self.pipeline.run("React latest release")
        self.assertTrue(plan.needs_external_knowledge)
        self.assertTrue(trace.needs_external_knowledge)

    async def test_greeting_skips_external_knowledge(self) -> None:
        plan, trace = await self.pipeline.run("hello")
        self.assertFalse(plan.needs_external_knowledge)
        self.assertFalse(trace.needs_external_knowledge)

    async def test_plan_triggers_maintains_all_fields(self) -> None:
        plan, trace = await self.pipeline.run("latest FastAPI version")
        self.assertIsNotNone(plan.confidence)
        self.assertTrue(plan.needs_external_knowledge)


class ExternalKnowledgeResultTests(TestCase):

    def test_default_values(self) -> None:
        r = ExternalKnowledgeResult()
        self.assertFalse(r.triggered)
        self.assertFalse(r.success)
        self.assertEqual(r.query, "")
        self.assertEqual(r.result_count, 0)
        self.assertEqual(r.context_block, "")
        self.assertEqual(r.instruction, "")
        self.assertEqual(r.execution_time_ms, 0.0)
        self.assertIsNone(r.error)

    def test_with_values(self) -> None:
        r = ExternalKnowledgeResult(
            triggered=True,
            success=True,
            query="test query",
            result_count=5,
            context_block="ext knowledge",
            instruction="prioritize",
            execution_time_ms=123.45,
            error=None,
        )
        self.assertTrue(r.triggered)
        self.assertTrue(r.success)
        self.assertEqual(r.query, "test query")
        self.assertEqual(r.result_count, 5)
        self.assertEqual(r.context_block, "ext knowledge")
        self.assertEqual(r.instruction, "prioritize")
        self.assertEqual(r.execution_time_ms, 123.45)
        self.assertIsNone(r.error)
