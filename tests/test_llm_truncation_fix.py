"""Tests for LLM truncation fix: token budgets, response length, finish_reason logging."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.token_budget import TokenBudget, select_budget
from backend.app.schemas.routing import IntentType
from backend.app.services.llm_service import LLMService


class TokenBudgetTruncationFixTests(TestCase):
    """Token budget must never truncate long responses."""

    def setUp(self) -> None:
        self.settings_patch = patch("backend.app.services.token_budget.get_settings")
        mock_settings = self.settings_patch.start()
        mock_settings.return_value.llm_max_tokens = 2048

    def tearDown(self) -> None:
        self.settings_patch.stop()

    def test_explanation_min_1024(self) -> None:
        budget = select_budget("explain quantum computing", IntentType.NORMAL_CHAT)
        self.assertGreaterEqual(budget.max_tokens, 1024,
                                f"Explanation needs ≥1024 tokens, got {budget.max_tokens}")

    def test_long_explanation_with_memory_min_2048(self) -> None:
        budget = select_budget(
            "explain quantum computing in detail step by step",
            IntentType.NORMAL_CHAT,
            memory_count=5,
        )
        self.assertGreaterEqual(budget.max_tokens, 1024,
                                f"Long explanation with memory needs ≥1024 tokens, got {budget.max_tokens}")

    def test_biography_min_1024(self) -> None:
        budget = select_budget("who is Mahatma Gandhi", IntentType.NORMAL_CHAT)
        self.assertGreaterEqual(budget.max_tokens, 1024,
                                f"Biography query needs ≥1024 tokens, got {budget.max_tokens}")

    def test_code_gen_min_1024(self) -> None:
        budget = select_budget("write a python program", IntentType.NORMAL_CHAT)
        self.assertGreaterEqual(budget.max_tokens, 1024,
                                f"Code generation needs ≥1024 tokens, got {budget.max_tokens}")

    def test_large_code_min_2048(self) -> None:
        budget = select_budget(
            "write a complex python web application with authentication and database",
        )
        self.assertGreaterEqual(budget.max_tokens, 1024,
                                f"Large code needs ≥1024 tokens, got {budget.max_tokens}")

    def test_long_input_min_1024(self) -> None:
        budget = select_budget("x" * 301)
        self.assertGreaterEqual(budget.max_tokens, 1024,
                                f"Long input needs ≥1024 tokens, got {budget.max_tokens}")

    def test_question_gets_512(self) -> None:
        budget = select_budget("what is the capital of France?")
        self.assertGreaterEqual(budget.max_tokens, 512)

    def test_greeting_stays_small(self) -> None:
        budget = select_budget("hello")
        self.assertLessEqual(budget.max_tokens, 96,
                             f"Greeting should stay small, got {budget.max_tokens}")

    def test_system_query_stays_small(self) -> None:
        budget = select_budget("who are you", IntentType.SYSTEM_QUERY)
        self.assertLessEqual(budget.max_tokens, 128)

    def test_delete_memory_stays_small(self) -> None:
        budget = select_budget("delete that", IntentType.DELETE_MEMORY)
        self.assertLessEqual(budget.max_tokens, 128)

    def test_label_reflects_purpose(self) -> None:
        budget = select_budget("explain RAG", IntentType.NORMAL_CHAT)
        self.assertIn(budget.label, ("long_explain", "explain"))

    def test_explanation_about_framework(self) -> None:
        budget = select_budget("describe Retrieval-Augmented Generation", IntentType.NORMAL_CHAT)
        self.assertGreaterEqual(budget.max_tokens, 1024)

    def test_what_is_query_about_framework(self) -> None:
        budget = select_budget("what is RAG", IntentType.NORMAL_CHAT)
        self.assertGreaterEqual(budget.max_tokens, 1024)

    def test_elaborate_query(self) -> None:
        budget = select_budget("elaborate on transformer architecture", IntentType.NORMAL_CHAT)
        self.assertGreaterEqual(budget.max_tokens, 1024)

    def test_respects_llm_max_tokens_setting(self) -> None:
        with patch("backend.app.services.token_budget.get_settings") as mock_settings:
            mock_settings.return_value.llm_max_tokens = 512
            budget = select_budget("explain quantum computing", IntentType.NORMAL_CHAT)
            self.assertLessEqual(budget.max_tokens, 512)

    def test_respects_low_llm_max_tokens(self) -> None:
        with patch("backend.app.services.token_budget.get_settings") as mock_settings:
            mock_settings.return_value.llm_max_tokens = 128
            budget = select_budget("explain quantum computing", IntentType.NORMAL_CHAT)
            self.assertLessEqual(budget.max_tokens, 128)


class GroqProviderFinishReasonLoggingTests(TestCase):
    """Groq provider must log finish_reason and token usage."""

    @patch("backend.app.services.providers.groq_provider.GroqProvider._get_client")
    def test_logs_finish_reason(self, mock_get_client) -> None:
        from backend.app.services.providers.groq_provider import GroqProvider

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "test response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        }
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        provider = GroqProvider(api_key="test-key", base_url="https://api.groq.com", model="test-model")

        result = self._run_async(provider.generate(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=100,
        ))
        self.assertEqual(result, "test response")

    @patch("backend.app.services.providers.groq_provider.GroqProvider._get_client")
    def test_logs_length_finish_reason(self, mock_get_client) -> None:
        from backend.app.services.providers.groq_provider import GroqProvider

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "incomplete response"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200},
        }
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        provider = GroqProvider(api_key="test-key", base_url="https://api.groq.com", model="test-model")

        result = self._run_async(provider.generate(
            messages=[{"role": "user", "content": "tell me a long story"}],
            max_tokens=50,
        ))
        self.assertEqual(result, "incomplete response")

    @patch("backend.app.services.providers.groq_provider.GroqProvider._get_client")
    def test_handles_missing_usage(self, mock_get_client) -> None:
        from backend.app.services.providers.groq_provider import GroqProvider

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "test"}, "finish_reason": "stop"}],
        }
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        provider = GroqProvider(api_key="test-key", base_url="https://api.groq.com", model="test-model")

        result = self._run_async(provider.generate(
            messages=[{"role": "user", "content": "hello"}],
        ))
        self.assertEqual(result, "test")

    @staticmethod
    def _run_async(coro) -> str:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)


class LLMServiceDefaultTokenTests(TestCase):
    """LLMService must have sufficiently large default max_tokens."""

    def test_default_max_tokens_is_1024(self) -> None:
        self.assertEqual(LLMService._DEFAULT_MAX_TOKENS, 1024,
                         "Default max_tokens must be 1024 to prevent truncation")

    @patch("backend.app.services.llm_service.LLMService._build_manager")
    def test_chat_completion_uses_effective_max_tokens(self, mock_build) -> None:
        mock_manager = MagicMock()
        mock_manager.generate = AsyncMock(return_value="response")
        mock_build.return_value = mock_manager

        service = LLMService.__new__(LLMService)
        service._settings = MagicMock()
        service._logger = MagicMock()
        service._manager = mock_manager

        result = self._run_async(service._chat_completion(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=None,
        ))
        self.assertEqual(result, "response")
        args, kwargs = mock_manager.generate.call_args
        self.assertEqual(kwargs["max_tokens"], 1024)

    @staticmethod
    def _run_async(coro) -> str:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)


class SettingsDefaultTokenTests(TestCase):
    """Settings must have a sufficiently large default max_tokens."""

    def test_llm_max_tokens_default_is_1024(self) -> None:
        from backend.app.config.settings import Settings
        s = Settings()
        self.assertGreaterEqual(s.llm_max_tokens, 1024,
                                f"Default llm_max_tokens must be ≥1024, got {s.llm_max_tokens}")
