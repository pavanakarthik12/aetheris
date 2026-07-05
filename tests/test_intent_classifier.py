"""Tests for the IntentClassifier service."""

from __future__ import annotations

import sys
from pathlib import Path

import unittest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.schemas.routing import IntentType
from backend.app.services.intent_classifier import (
    IntentClassifier,
    LLMIntentClassifier,
    RuleIntentClassifier,
)


class FakeLLMService:
    async def generate_text(self, prompt: str) -> str:
        return '{"intent": "NORMAL_CHAT", "confidence": 0.9, "reason": "test"}'

    async def aclose(self) -> None:
        return None


class RuleIntentClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = RuleIntentClassifier()

    async def _classify(self, message: str) -> IntentType:
        result = await self.classifier.classify(message)
        return result.primary_intent

    # --------------------------------------------------------------
    # NORMAL_CHAT
    # --------------------------------------------------------------
    def test_normal_chat_greeting(self) -> None:
        intent = self._run("Hello, how are you?")
        self.assertEqual(intent, IntentType.NORMAL_CHAT)

    def test_normal_chat_opinion(self) -> None:
        intent = self._run("I think the weather is nice today.")
        self.assertEqual(intent, IntentType.NORMAL_CHAT)

    def test_normal_chat_general_question(self) -> None:
        intent = self._run("What is the capital of France?")
        self.assertEqual(intent, IntentType.NORMAL_CHAT)

    # --------------------------------------------------------------
    # CREATE_MEMORY
    # --------------------------------------------------------------
    def test_create_memory_remember(self) -> None:
        intent = self._run("Remember that my favorite color is blue.")
        self.assertEqual(intent, IntentType.CREATE_MEMORY)

    def test_create_memory_preference(self) -> None:
        intent = self._run("I like programming in Python.")
        self.assertEqual(intent, IntentType.CREATE_MEMORY)

    def test_create_memory_save_this(self) -> None:
        intent = self._run("Save this: my email is user@example.com")
        self.assertEqual(intent, IntentType.CREATE_MEMORY)

    # --------------------------------------------------------------
    # DELETE_MEMORY
    # --------------------------------------------------------------
    def test_delete_memory_forget(self) -> None:
        intent = self._run("Forget my Java preference.")
        self.assertEqual(intent, IntentType.DELETE_MEMORY)

    def test_delete_memory_remove(self) -> None:
        intent = self._run("Remove that memory about cats.")
        self.assertEqual(intent, IntentType.DELETE_MEMORY)

    def test_delete_memory_erase(self) -> None:
        intent = self._run("Erase all my project memories.")
        self.assertEqual(intent, IntentType.DELETE_MEMORY)

    # --------------------------------------------------------------
    # UPDATE_MEMORY
    # --------------------------------------------------------------
    def test_update_memory_actually(self) -> None:
        intent = self._run("Actually, I changed my mind about Java.")
        self.assertEqual(intent, IntentType.UPDATE_MEMORY)

    def test_update_memory_switched(self) -> None:
        intent = self._run("I switched to using TypeScript.")
        self.assertEqual(intent, IntentType.UPDATE_MEMORY)

    def test_update_memory_no_longer(self) -> None:
        intent = self._run("I no longer like coffee.")
        self.assertEqual(intent, IntentType.UPDATE_MEMORY)

    # --------------------------------------------------------------
    # SEARCH_MEMORY
    # --------------------------------------------------------------
    def test_search_memory_what_is(self) -> None:
        intent = self._run("What do you know about my projects?")
        self.assertEqual(intent, IntentType.SEARCH_MEMORY)

    def test_search_memory_find(self) -> None:
        intent = self._run("Search my memories for Java preferences.")
        self.assertEqual(intent, IntentType.SEARCH_MEMORY)

    def test_search_memory_recall(self) -> None:
        intent = self._run("What do I remember about Kubernetes?")
        self.assertEqual(intent, IntentType.SEARCH_MEMORY)

    # --------------------------------------------------------------
    # WEB_SEARCH
    # --------------------------------------------------------------
    def test_web_search_explicit(self) -> None:
        intent = self._run("Search the web for AI news.")
        self.assertEqual(intent, IntentType.WEB_SEARCH)

    def test_web_search_latest(self) -> None:
        intent = self._run("What is the latest in cognitive architecture?")
        self.assertEqual(intent, IntentType.WEB_SEARCH)

    # --------------------------------------------------------------
    # SYSTEM_QUERY
    # --------------------------------------------------------------
    def test_system_query_who_are_you(self) -> None:
        intent = self._run("Who are you?")
        self.assertEqual(intent, IntentType.SYSTEM_QUERY)

    def test_system_query_capabilities(self) -> None:
        intent = self._run("What can you do?")
        self.assertEqual(intent, IntentType.SYSTEM_QUERY)

    # --------------------------------------------------------------
    # MULTI_ACTION
    # --------------------------------------------------------------
    def test_multi_action_create_and_search(self) -> None:
        result = self._run_classify("Remember I like Rust and search the web for Rust news.")
        self.assertEqual(result.primary_intent, IntentType.MULTI_ACTION)

    def test_multi_action_sub_intents(self) -> None:
        result = self._run_classify("Save this and look up the latest news.")
        self.assertIn(IntentType.CREATE_MEMORY, result.sub_intents)

    # --------------------------------------------------------------
    # Empty / edge cases
    # --------------------------------------------------------------
    def test_empty_message(self) -> None:
        intent = self._run("")
        self.assertEqual(intent, IntentType.UNKNOWN)

    def test_whitespace_message(self) -> None:
        intent = self._run("   ")
        self.assertEqual(intent, IntentType.UNKNOWN)

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------
    def _run(self, message: str) -> IntentType:
        import asyncio
        return asyncio.run(self._classify(message))

    def _run_classify(self, message: str):
        import asyncio
        return asyncio.run(self.classifier.classify(message))


class LLMIntentClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = FakeLLMService()
        self.classifier = LLMIntentClassifier(self.llm)

    def test_llm_classify_returns_intent(self) -> None:
        import asyncio
        result = asyncio.run(self.classifier.classify("Hello"))
        self.assertEqual(result.primary_intent, IntentType.NORMAL_CHAT)
        self.assertGreater(result.confidence, 0.5)


class CombinedIntentClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = FakeLLMService()
        self.classifier = IntentClassifier(self.llm)

    def test_combined_uses_rule_first(self) -> None:
        import asyncio
        result = asyncio.run(self.classifier.classify("Forget my Java memory."))
        self.assertEqual(result.primary_intent, IntentType.DELETE_MEMORY)
        self.assertEqual(result.classifier_source, "rule")

    def test_combined_returns_rule_for_high_confidence(self) -> None:
        import asyncio
        result = asyncio.run(self.classifier.classify("Hello"))
        self.assertEqual(result.primary_intent, IntentType.NORMAL_CHAT)
        self.assertEqual(result.classifier_source, "rule")


if __name__ == "__main__":
    unittest.main()
