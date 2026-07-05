"""Tests for the Cognitive Request Router."""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import unittest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.schemas.chat import MemoryActionType
from backend.app.schemas.routing import IntentClassification, IntentType
from backend.app.services.intent_classifier import IntentClassifier, RuleIntentClassifier
from backend.app.services.request_router import CognitiveRequestRouter


class FakeEmbeddingService:
    async def embed_text(self, text: str) -> list[float]:
        normalized = text.lower()
        if any(token in normalized for token in ("cat", "feline", "java")):
            return [1.0, 0.0, 0.0]
        if any(token in normalized for token in ("dog", "python", "rust")):
            return [0.0, 1.0, 0.0]
        return [0.33, 0.33, 0.34]


class FakeChromaService:
    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def add_memory(self, memory_id: str, embedding: list[float], document: str, metadata: dict[str, Any]) -> None:
        self._records[memory_id] = {"id": memory_id, "embedding": embedding, "document": document, "metadata": metadata}

    def search_memory(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        scored = []
        for rec in self._records.values():
            score = 1.0 if rec["embedding"] == query_embedding else 0.3
            scored.append({"id": rec["id"], "document": rec["document"], "score": score, "metadata": rec["metadata"]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_memory_by_id(self, memory_id: str) -> dict[str, Any] | None:
        rec = self._records.get(memory_id)
        if not rec:
            return None
        return {"id": rec["id"], "document": rec["document"], "metadata": rec["metadata"]}

    def delete_memory(self, memory_id: str) -> None:
        if memory_id not in self._records:
            raise RuntimeError("Not found")
        del self._records[memory_id]

    def update_memory(self, memory_id: str, embedding: list[float], document: str, metadata: dict[str, Any]) -> None:
        if memory_id not in self._records:
            raise RuntimeError("Not found")
        self._records[memory_id] = {"id": memory_id, "embedding": embedding, "document": document, "metadata": metadata}

    def list_all_memories(self) -> list[dict[str, Any]]:
        return [{"id": r["id"], "document": r["document"], "metadata": r["metadata"]} for r in self._records.values()]

    def get_memories_by_metadata(
        self,
        where: dict[str, Any],
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        results = []
        for mem_id, rec in self._records.items():
            if rec is None:
                continue
            meta = rec.get("metadata")
            if meta is None:
                continue
            matches = all(meta.get(k) == v for k, v in where.items())
            if matches:
                results.append({"id": rec["id"], "document": rec["document"], "metadata": dict(meta)})
        if limit is not None:
            results = results[:limit]
        return results


class FakeMemoryService:
    def __init__(self, chroma: FakeChromaService, embeddings: FakeEmbeddingService):
        self._chroma = chroma
        self._embeddings = embeddings

    async def save_memory(self, memory_text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        mem_id = str(uuid.uuid4())
        emb = await self._embeddings.embed_text(memory_text)
        self._chroma.add_memory(mem_id, emb, memory_text, metadata or {})
        return {"memory_id": mem_id, "status": "saved"}

    async def search_memory(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        emb = await self._embeddings.embed_text(query)
        return self._chroma.search_memory(emb, top_k=top_k)

    def delete_memory(self, memory_id: str) -> dict[str, str]:
        self._chroma.delete_memory(memory_id)
        return {"memory_id": memory_id, "status": "deleted"}

    def list_memories(self) -> list[dict[str, Any]]:
        return self._chroma.list_all_memories()


class FakeMemoryEvaluator:
    def __init__(self, store: bool = True, category: str = "Other", importance: float = 0.5) -> None:
        self._store = store
        self._category = category
        self._importance = importance

    async def evaluate_memory(self, message_text: str) -> Any:
        from backend.app.services.memory_evaluator import MemoryEvaluation
        return MemoryEvaluation(
            store=self._store,
            category=self._category,
            importance=self._importance,
            reason="test evaluation",
        )


class FakeMemoryEvolution:
    def __init__(self) -> None:
        self._target_id: str | None = None

    def set_target(self, target_id: str) -> None:
        self._target_id = target_id

    async def decide_evolution(self, memory_text: str, existing_evaluation: dict[str, Any] | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {"action": "CREATE", "target_id": None, "version": 1, "explanation": "test"}
        if self._target_id:
            result["action"] = "UPDATE"
            result["target_id"] = self._target_id
        return result

    async def create_memory(self, memory_text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"memory_id": str(uuid.uuid4()), "status": "created", "version": 1}

    async def update_memory(self, memory_id: str, new_text: str, new_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"memory_id": memory_id, "version": 2, "status": "updated"}

    async def get_memory_document(self, memory_id: str) -> str | None:
        return "Existing memory content."


class FakeLLMService:
    async def generate_with_context(self, user_message: str, system_prompt: str, memory_context: str = "") -> str:
        return f"Response to: {user_message}"

    async def generate_text(self, prompt: str) -> str:
        return "ok"

    async def aclose(self) -> None:
        return None


class FakeContextBuilder:
    def build_memory_context(self, memories: list[dict[str, Any]]) -> str:
        if not memories:
            return ""
        lines = [f"- {m['document']}" for m in memories]
        return "Relevant Memories:\n" + "\n".join(lines)

    def build_system_prompt(self) -> str:
        return "You are Aetheris, a cognitive AI assistant."


class FakeReflectionService:
    async def reflect_with_context(self, user_message: str, assistant_response: str) -> None:
        return None


class FakeImmediateMemoryProcessor:
    async def process_message(self, message_text: str) -> Any:
        return type("Result", (), {
            "action": MemoryActionType.SKIP,
            "success": True,
            "memory_id": None,
            "error": None,
        })()


class FakeIntentClassifier(IntentClassifier):
    def __init__(self, intent: IntentType = IntentType.NORMAL_CHAT) -> None:
        super().__init__(llm_service=FakeLLMService())  # type: ignore[arg-type]
        self._intent = intent

    async def classify(self, message: str) -> IntentClassification:
        return IntentClassification(
            primary_intent=self._intent,
            confidence=0.95,
            classifier_source="test",
        )


class CognitiveRequestRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embeddings = FakeEmbeddingService()
        self.chroma = FakeChromaService()
        self.memory_service = FakeMemoryService(self.chroma, self.embeddings)
        self.evaluator = FakeMemoryEvaluator()
        self.evolution = FakeMemoryEvolution()
        self.llm = FakeLLMService()
        self.context_builder = FakeContextBuilder()
        self.reflection = FakeReflectionService()
        self.imm = FakeImmediateMemoryProcessor()
        self.intent_classifier = FakeIntentClassifier()

        self.router = CognitiveRequestRouter(
            llm_service=self.llm,
            memory_service=self.memory_service,
            memory_evaluator=self.evaluator,
            memory_evolution=self.evolution,
            chroma_service=self.chroma,
            embedding_service=self.embeddings,
            context_builder=self.context_builder,
            reflection_service=self.reflection,
            intent_classifier=self.intent_classifier,
            immediate_memory_processor=self.imm,
        )

    def _seed_memory(self, text: str) -> str:
        mid = str(uuid.uuid4())
        self.chroma.add_memory(mid, [1.0, 0.0, 0.0], text, {"source": "user"})
        return mid

    # --------------------------------------------------------------
    # NORMAL_CHAT
    # --------------------------------------------------------------
    def test_normal_chat_returns_response(self) -> None:
        self.intent_classifier._intent = IntentType.NORMAL_CHAT
        result = self._route("Hello")
        self.assertEqual(result.debug.detected_intent, IntentType.NORMAL_CHAT)
        self.assertIn("Hello", result.response)
        self.assertTrue(result.memory_success)

    def test_normal_chat_includes_memory_count(self) -> None:
        self._seed_memory("User likes cats.")
        self.intent_classifier._intent = IntentType.NORMAL_CHAT
        result = self._route("Tell me about cats")
        self.assertGreaterEqual(result.memory_count, 0)
        self.assertEqual(result.memory_action, MemoryActionType.SKIP)

    # --------------------------------------------------------------
    # CREATE_MEMORY
    # --------------------------------------------------------------
    def test_create_memory_creates_and_responds(self) -> None:
        self.intent_classifier._intent = IntentType.CREATE_MEMORY
        result = self._route("Remember that I like Rust.")
        self.assertIn("Rust", result.response)
        self.assertTrue(result.memory_success)

    # --------------------------------------------------------------
    # DELETE_MEMORY
    # --------------------------------------------------------------
    def test_delete_memory_found(self) -> None:
        self._seed_memory("I like Java.")
        self.intent_classifier._intent = IntentType.DELETE_MEMORY
        result = self._route("Forget my Java memory.")
        self.assertEqual(result.memory_action, MemoryActionType.DELETE)
        self.assertTrue(result.memory_success)
        self.assertIn("deleted", result.response.lower())

    def test_delete_memory_not_found(self) -> None:
        self.intent_classifier._intent = IntentType.DELETE_MEMORY
        result = self._route("Forget something that doesn't exist.")
        self.assertEqual(result.memory_action, MemoryActionType.SKIP)
        self.assertIn("couldn't delete", result.response.lower())

    # --------------------------------------------------------------
    # SEARCH_MEMORY
    # --------------------------------------------------------------
    def test_search_memory_returns_found(self) -> None:
        self._seed_memory("User likes Java.")
        self.intent_classifier._intent = IntentType.SEARCH_MEMORY
        result = self._route("What do you know about my language preferences?")
        self.assertTrue(result.memory_success)
        self.assertIn("Response", result.response)

    def test_search_memory_no_results(self) -> None:
        self.intent_classifier._intent = IntentType.SEARCH_MEMORY
        result = self._route("What do you know?")
        self.assertTrue(result.memory_success)
        self.assertEqual(result.memory_count, 0)

    # --------------------------------------------------------------
    # UPDATE_MEMORY
    # --------------------------------------------------------------
    def test_update_memory_updates_and_responds(self) -> None:
        mid = self._seed_memory("I like Java.")
        self.evolution.set_target(mid)
        self.intent_classifier._intent = IntentType.UPDATE_MEMORY
        result = self._route("Actually I prefer Python.")
        self.assertTrue(result.memory_success)
        self.assertIn("Response", result.response)

    # --------------------------------------------------------------
    # WEB_SEARCH
    # --------------------------------------------------------------
    def test_web_search_returns_response(self) -> None:
        self.intent_classifier._intent = IntentType.WEB_SEARCH
        result = self._route("Search the web for AI news.")
        self.assertTrue(result.memory_success)
        self.assertTrue(result.debug.internet_used)
        self.assertIn("Response", result.response)

    # --------------------------------------------------------------
    # SYSTEM_QUERY
    # --------------------------------------------------------------
    def test_system_query_returns_response(self) -> None:
        self.intent_classifier._intent = IntentType.SYSTEM_QUERY
        result = self._route("Who are you?")
        self.assertTrue(result.memory_success)
        self.assertIn("Response", result.response)

    # --------------------------------------------------------------
    # MULTI_ACTION
    # --------------------------------------------------------------
    def test_multi_action_with_sub_intents(self) -> None:
        self.intent_classifier._intent = IntentType.MULTI_ACTION
        result = self._route_classify(
            "Remember I like Rust and search the web for Rust news.",
            sub_intents=[IntentType.CREATE_MEMORY, IntentType.WEB_SEARCH],
        )
        self.assertTrue(result.memory_success)
        self.assertIn("Response", result.response)

    # --------------------------------------------------------------
    # Debug info
    # --------------------------------------------------------------
    def test_debug_info_includes_all_fields(self) -> None:
        self.intent_classifier._intent = IntentType.NORMAL_CHAT
        result = self._route("Hello")
        debug = result.debug
        self.assertEqual(debug.detected_intent, IntentType.NORMAL_CHAT)
        self.assertGreater(debug.confidence, 0.0)
        self.assertGreater(debug.total_duration_ms, 0.0)
        self.assertGreaterEqual(len(debug.steps), 1)
        self.assertGreaterEqual(len(debug.subsystems_used), 1)

    def test_debug_tracks_steps(self) -> None:
        self.intent_classifier._intent = IntentType.NORMAL_CHAT
        result = self._route("Hello")
        step_names = [s.subsystem for s in result.debug.steps]
        self.assertIn("IntentClassifier", step_names)
        self.assertIn("LLM", step_names)

    # --------------------------------------------------------------
    # Error handling
    # --------------------------------------------------------------
    def test_router_handles_exception_gracefully(self) -> None:
        broken = BrokenRouter(
            llm_service=self.llm,
            memory_service=self.memory_service,
            memory_evaluator=self.evaluator,
            memory_evolution=self.evolution,
            chroma_service=self.chroma,
            embedding_service=self.embeddings,
            context_builder=self.context_builder,
            reflection_service=self.reflection,
            intent_classifier=self.intent_classifier,
            immediate_memory_processor=self.imm,
        )
        self.intent_classifier._intent = IntentType.NORMAL_CHAT

        import asyncio
        result = asyncio.run(broken.route("Hello"))
        self.assertTrue(result.memory_success)

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------
    def _route(self, message: str):
        import asyncio
        return asyncio.run(self.router.route(message))

    def _route_classify(self, message: str, sub_intents: list[IntentType] | None = None):
        self.intent_classifier._intent = IntentType.MULTI_ACTION
        original = self.intent_classifier.classify

        async def patched(msg: str) -> IntentClassification:
            return IntentClassification(
                primary_intent=IntentType.MULTI_ACTION,
                confidence=0.95,
                sub_intents=sub_intents or [],
                classifier_source="test",
            )

        self.intent_classifier.classify = patched
        import asyncio
        result = asyncio.run(self.router.route(message))
        self.intent_classifier.classify = original
        return result


class BrokenRouter(CognitiveRequestRouter):
    async def _handle_normal_chat(self, message: str, steps: list, debug: Any) -> Any:
        raise RuntimeError("Simulated handler failure")


if __name__ == "__main__":
    unittest.main()
