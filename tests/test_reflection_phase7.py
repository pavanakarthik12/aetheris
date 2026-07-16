"""Phase 7 Reflection Engine tests — analysis only (no memory mutations)."""

from __future__ import annotations

import json
import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import unittest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.schemas.reflection import (
    AssistantQuality,
    ReflectionAction,
    ReflectionOutput,
    ReflectionStatistics,
)
from backend.app.services.chroma_service import ChromaServiceError
from backend.app.services.reflection_service import (
    INITIAL_STRENGTH,
    MIN_CONFIDENCE_FOR_ACTION,
    STRENGTH_INCREMENT,
    ReflectionService,
    ReflectionServiceError,
)


class FakeEmbeddingService:
    async def embed_text(self, text: str) -> list[float]:
        normalized = text.lower()
        if any(token in normalized for token in ("cat", "feline", "kitten")):
            return [1.0, 0.0, 0.0]
        if any(token in normalized for token in ("dog", "canine", "puppy")):
            return [0.0, 1.0, 0.0]
        if any(token in normalized for token in ("java", "python", "kubernetes", "aetheris", "project")):
            return [0.8, 0.1, 0.1]
        return [0.33, 0.33, 0.34]

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [await self.embed_text(d) for d in documents]


class FakeChromaService:
    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def add_memory(
        self,
        memory_id: str,
        embedding: list[float],
        document: str,
        metadata: dict[str, Any],
    ) -> None:
        self._records[memory_id] = {
            "id": memory_id,
            "embedding": embedding,
            "document": document,
            "metadata": metadata,
        }

    def search_memory(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        scored_records: list[dict[str, Any]] = []
        for record in self._records.values():
            score = _cosine_similarity(query_embedding, record["embedding"])
            scored_records.append({
                "id": record["id"],
                "document": record["document"],
                "score": score,
                "metadata": record["metadata"],
            })
        scored_records.sort(key=lambda item: item["score"], reverse=True)
        return scored_records[:top_k]

    def delete_memory(self, memory_id: str) -> None:
        del self._records[memory_id]

    def get_memory_by_id(self, memory_id: str) -> dict[str, Any] | None:
        record = self._records.get(memory_id)
        if record is None:
            return None
        return {
            "id": record["id"],
            "document": record["document"],
            "metadata": record["metadata"],
        }

    def update_memory(
        self,
        memory_id: str,
        embedding: list[float],
        document: str,
        metadata: dict[str, Any],
    ) -> None:
        if memory_id not in self._records:
            raise ChromaServiceError(f"Memory '{memory_id}' not found.", status_code=404)
        self._records[memory_id] = {
            "id": memory_id,
            "embedding": embedding,
            "document": document,
            "metadata": metadata,
        }

    def list_all_memories(self) -> list[dict[str, Any]]:
        return [
            {
                "id": record["id"],
                "document": record["document"],
                "metadata": record["metadata"],
            }
            for record in self._records.values()
        ]

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
                results.append({
                    "id": rec["id"],
                    "document": rec["document"],
                    "metadata": dict(meta),
                })
        if limit is not None:
            results = results[:limit]
        return results


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(v * v for v in left))
    right_norm = math.sqrt(sum(v * v for v in right))
    if not left_norm or not right_norm:
        return 0.0
    return round(numerator / (left_norm * right_norm), 6)


class FakeLLMService:
    async def generate_text(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> str:
        is_reflection = '"action":"NO_ACTION"' in prompt
        if not is_reflection:
            return "ok"

        user_line = self._extract_user_message(prompt)

        # Scenario 1: Repeated preference confirmation -> STRENGTHEN_MEMORY
        if "still prefer Java" in user_line or "still enjoy Java" in user_line:
            return json.dumps({
                "new_memory": False,
                "update_memory": False,
                "memory_strengthened": True,
                "assistant_mistake": False,
                "user_corrected_ai": False,
                "confidence": 0.92,
                "future_behavior_change": False,
                "reflection_summary": "User confirmed their Java language preference.",
                "actions": ["STRENGTHEN_MEMORY"],
            })

        # Scenario 2: User corrects the assistant — analysis only, no mutation
        if "No, I said" in user_line or "I said Java" in user_line:
            return json.dumps({
                "new_memory": False,
                "update_memory": False,
                "memory_strengthened": False,
                "assistant_mistake": True,
                "user_corrected_ai": True,
                "confidence": 0.88,
                "future_behavior_change": True,
                "reflection_summary": "Assistant misremembered user's language preference.",
                "actions": ["NO_ACTION"],
            })

        # Scenario 3: New achievement/progress — analysis only, no mutation
        if "finished my Kubernetes" in user_line or "completed Kubernetes" in user_line:
            return json.dumps({
                "new_memory": False,
                "update_memory": False,
                "memory_strengthened": False,
                "assistant_mistake": False,
                "user_corrected_ai": False,
                "confidence": 0.95,
                "future_behavior_change": False,
                "reflection_summary": "User completed Kubernetes certification.",
                "actions": ["NO_ACTION"],
            })

        # Scenario 4: Simple greeting -> NO_ACTION
        greeting_tokens = {"hello", "hi", "hey", "greetings"}
        user_words = set(user_line.lower().split())
        if user_words & greeting_tokens:
            return json.dumps({
                "new_memory": False,
                "update_memory": False,
                "memory_strengthened": False,
                "assistant_mistake": False,
                "user_corrected_ai": False,
                "confidence": 0.99,
                "future_behavior_change": False,
                "reflection_summary": "Simple greeting, no action needed.",
                "actions": ["NO_ACTION"],
            })

        # Scenario 5: Contradiction detection (analysis only, no strength change)
        if "actually i hate" in user_line.lower():
            return self._build_response({
                "new_memory": True,
                "update_memory": False,
                "memory_strengthened": False,
                "assistant_mistake": False,
                "user_corrected_ai": False,
                "confidence": 0.75,
                "future_behavior_change": True,
                "reflection_summary": "User expressed a contradictory preference.",
                "actions": ["NO_ACTION"],
                "action": "NO_ACTION",
                "consistency": False,
                "reasoning": "User contradicted previous preference.",
            })

        # Scenario 6: Low confidence / ambiguous
        if "i'm not sure" in user_line.lower() or "maybe" in user_line.lower():
            return self._build_response({
                "new_memory": False,
                "update_memory": False,
                "memory_strengthened": False,
                "assistant_mistake": False,
                "user_corrected_ai": False,
                "confidence": 0.25,
                "future_behavior_change": False,
                "reflection_summary": "Ambiguous input, confidence too low to act.",
                "actions": ["NO_ACTION"],
                "action": "NO_ACTION",
            })

        # Default: NO_ACTION with enhanced fields
        return self._build_response({
            "new_memory": False,
            "update_memory": False,
            "memory_strengthened": False,
            "assistant_mistake": False,
            "user_corrected_ai": False,
            "confidence": 0.95,
            "future_behavior_change": False,
            "reflection_summary": "No significant changes detected.",
            "actions": ["NO_ACTION"],
            "action": "NO_ACTION",
        })

    async def generate_with_context(
        self,
        user_message: str,
        system_prompt: str,
        memory_context: str = "",
    ) -> str:
        return "ok"

    @staticmethod
    def _extract_user_message(prompt: str) -> str:
        marker = "User Message:\n"
        if marker in prompt:
            tail = prompt.split(marker, 1)[1]
            if "\n\nAssistant Response:" in tail:
                return tail.split("\n\nAssistant Response:", 1)[0].strip()
            return tail.strip()
        return prompt.strip()

    @staticmethod
    def _build_response(base: dict) -> str:
        enhanced = {
            "assistant_quality": {
                "correctness": 0.95,
                "completeness": 0.90,
                "relevance": 0.95,
                "clarity": 0.95,
                "confidence": 0.90,
                "hallucination_risk": 0.05,
            },
            "user_preferences_detected": [],
            "reasoning": "",
            "requires_manual_review": False,
            "memory_strength_delta": 0.0,
            "consistency": True,
        }
        enhanced.update(base)
        return json.dumps(enhanced)

    async def aclose(self) -> None:
        return None


class FakeMemoryService:
    def __init__(
        self,
        embedding_service: FakeEmbeddingService,
        chroma_service: FakeChromaService,
    ) -> None:
        self._embeddings = embedding_service
        self._chroma = chroma_service

    async def save_memory(
        self,
        memory_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        memory_id = str(uuid.uuid4())
        created_at = datetime.now(tz=timezone.utc).isoformat()
        resolved = dict(metadata or {})
        resolved.setdefault("created_at", created_at)
        resolved.setdefault("source", "user")
        resolved.setdefault("tags", "")
        resolved.setdefault("importance", 0.5)
        resolved.setdefault("memory_strength", 0.60)

        embedding = await self._embeddings.embed_text(memory_text)
        self._chroma.add_memory(
            memory_id=memory_id,
            embedding=embedding,
            document=memory_text,
            metadata=resolved,
        )
        return {"memory_id": memory_id, "status": "saved", "created_at": created_at}

    async def search_memory(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        query_embedding = await self._embeddings.embed_text(query)
        return self._chroma.search_memory(query_embedding=query_embedding, top_k=top_k)

    def delete_memory(self, memory_id: str) -> dict[str, str]:
        self._chroma.delete_memory(memory_id)
        return {"memory_id": memory_id, "status": "deleted"}

    def list_memories(self) -> list[dict[str, Any]]:
        return self._chroma.list_all_memories()


class ReflectionPhase7Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding_service = FakeEmbeddingService()
        self.chroma_service = FakeChromaService()
        self.llm_service = FakeLLMService()
        self.memory_service = FakeMemoryService(
            embedding_service=self.embedding_service,
            chroma_service=self.chroma_service,
        )
        self.reflection_service = ReflectionService(
            llm_service=self.llm_service,
            memory_service=self.memory_service,
            chroma_service=self.chroma_service,
            embedding_service=self.embedding_service,
        )

    def _seed_memory(self, text: str, strength: float = 0.60) -> str:
        mem_id = str(uuid.uuid4())
        embedding = [0.8, 0.1, 0.1]
        self.chroma_service.add_memory(
            memory_id=mem_id,
            embedding=embedding,
            document=text,
            metadata={
                "memory_strength": strength,
                "source": "user",
                "category": "Preference",
                "importance": 0.9,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
            },
        )
        return mem_id

    # ------------------------------------------------------------------
    # Scenario 1: Repeated preference confirmation -> STRENGTHEN_MEMORY
    # ------------------------------------------------------------------
    def test_scenario1_strengthen_memory(self) -> None:
        mem_id = self._seed_memory("My favorite language is Java.")

        record = self._run_reflection(
            user_message="I still prefer Java.",
            assistant_response="That's great! Java is a solid choice.",
        )

        actions = [a.value for a in record.reflection.actions]
        self.assertIn("STRENGTHEN_MEMORY", actions)
        self.assertTrue(record.reflection.memory_strengthened)

        updated = self.chroma_service.get_memory_by_id(mem_id)
        assert updated is not None
        expected_strength = INITIAL_STRENGTH + STRENGTH_INCREMENT
        self.assertAlmostEqual(
            updated["metadata"]["memory_strength"],
            expected_strength,
            places=2,
        )

    # ------------------------------------------------------------------
    # Scenario 2: User corrects the assistant (analysis only)
    # ------------------------------------------------------------------
    def test_scenario2_user_corrects_assistant(self) -> None:
        record = self._run_reflection(
            user_message="No, I said Java.",
            assistant_response="Oh, you said Python? Let me correct that.",
        )

        self.assertTrue(record.reflection.assistant_mistake)
        self.assertTrue(record.reflection.user_corrected_ai)
        self.assertTrue(record.reflection.future_behavior_change)
        actions = [a.value for a in record.reflection.actions]
        self.assertIn("NO_ACTION", actions)

    # ------------------------------------------------------------------
    # Scenario 3: New achievement (analysis only, no mutation)
    # ------------------------------------------------------------------
    def test_scenario3_achievement_analysis(self) -> None:
        record = self._run_reflection(
            user_message="I finished my Kubernetes certification.",
            assistant_response="Congratulations! That's a huge achievement.",
        )

        self.assertFalse(record.reflection.assistant_mistake)
        actions = [a.value for a in record.reflection.actions]
        self.assertIn("NO_ACTION", actions)
        self.assertIn("Kubernetes", record.reflection.reflection_summary)

    # ------------------------------------------------------------------
    # Scenario 4: Simple greeting -> NO_ACTION
    # ------------------------------------------------------------------
    def test_scenario4_greeting_no_action(self) -> None:
        record = self._run_reflection(
            user_message="Hello",
            assistant_response="Hi there! How can I help you today?",
        )

        self.assertFalse(record.reflection.new_memory)
        self.assertFalse(record.reflection.update_memory)
        self.assertFalse(record.reflection.memory_strengthened)
        self.assertFalse(record.reflection.assistant_mistake)
        self.assertFalse(record.reflection.future_behavior_change)
        actions = [a.value for a in record.reflection.actions]
        self.assertEqual(actions, ["NO_ACTION"])

    # ------------------------------------------------------------------
    # Scenario 5: Contradiction detection (analysis only)
    # ------------------------------------------------------------------
    def test_scenario5_contradiction_analysis(self) -> None:
        mem_id = self._seed_memory("I love Java programming.", strength=0.80)

        record = self._run_reflection(
            user_message="Actually I hate Java, I prefer Python.",
            assistant_response="I understand, let me update that.",
        )

        self.assertFalse(record.reflection.consistency)
        # Verify reflection did NOT weaken the memory (removed from engine)
        updated = self.chroma_service.get_memory_by_id(mem_id)
        assert updated is not None
        self.assertAlmostEqual(
            updated["metadata"]["memory_strength"],
            0.80,
            places=2,
        )

    # ------------------------------------------------------------------
    # Scenario 6: Low confidence threshold
    # ------------------------------------------------------------------
    def test_scenario6_low_confidence_defers_action(self) -> None:
        record = self._run_reflection(
            user_message="I'm not sure about this.",
            assistant_response="Let me help clarify.",
        )

        actions = [a.value for a in record.reflection.actions]
        self.assertEqual(actions, ["NO_ACTION"])
        self.assertLess(record.reflection.confidence, MIN_CONFIDENCE_FOR_ACTION)

    # ------------------------------------------------------------------
    # Reflection record is persisted as a file
    # ------------------------------------------------------------------
    def test_reflection_persisted_to_disk(self) -> None:
        record = self._run_reflection(
            user_message="Hello",
            assistant_response="Hi!",
        )

        fpath = self.reflection_service._reflections_dir / f"{record.id}.json"
        self.assertTrue(fpath.exists())
        saved = json.loads(fpath.read_text(encoding="utf-8"))
        self.assertEqual(saved["id"], record.id)
        self.assertEqual(saved["user_message"], "Hello")

    # ------------------------------------------------------------------
    # List and get reflection records
    # ------------------------------------------------------------------
    def test_list_and_get_reflections(self) -> None:
        self._run_reflection(
            user_message="Hello",
            assistant_response="Hi!",
        )
        self._run_reflection(
            user_message="I still prefer Java.",
            assistant_response="Great!",
        )

        reflections = self.reflection_service.list_reflections()
        self.assertGreaterEqual(len(reflections), 2)

        first_id = reflections[0]["id"]
        record = self.reflection_service.get_reflection(first_id)
        assert record is not None
        self.assertEqual(record["id"], first_id)

    # ------------------------------------------------------------------
    # Reflection failure is non-fatal
    # ------------------------------------------------------------------
    def test_reflection_error_handling(self) -> None:
        broken_llm = BrokenLLMService()
        service = ReflectionService(
            llm_service=broken_llm,
            memory_service=self.memory_service,
            chroma_service=self.chroma_service,
            embedding_service=self.embedding_service,
        )

        record = self._run_reflection_with_service(
            service,
            user_message="Hello",
            assistant_response="Hi!",
        )
        actions = [a.value for a in record.reflection.actions]
        self.assertEqual(actions, ["NO_ACTION"])

    # ------------------------------------------------------------------
    # JSON parsing handles extra text around the JSON output
    # ------------------------------------------------------------------
    def test_parse_reflection_handles_extra_text(self) -> None:
        raw = 'Here is my analysis:\n```json\n{"new_memory": true, "update_memory": false, "memory_strengthened": false, "assistant_mistake": false, "user_corrected_ai": false, "confidence": 0.9, "future_behavior_change": false, "reflection_summary": "test", "actions": ["CREATE_MEMORY"]}\n```\n---'
        result = self.reflection_service._parse_reflection(raw)
        self.assertTrue(result.new_memory)
        self.assertEqual(result.actions, [ReflectionAction.CREATE_MEMORY])

    def test_parse_reflection_handles_plain_json(self) -> None:
        raw = '{"new_memory": false, "update_memory": false, "memory_strengthened": false, "assistant_mistake": false, "user_corrected_ai": false, "confidence": 0.95, "future_behavior_change": false, "reflection_summary": "ok", "actions": ["NO_ACTION"]}'
        result = self.reflection_service._parse_reflection(raw)
        self.assertFalse(result.new_memory)
        self.assertEqual(result.actions, [ReflectionAction.NO_ACTION])

    # ------------------------------------------------------------------
    # Multiple strength increments cap at 1.0
    # ------------------------------------------------------------------
    def test_memory_strength_caps_at_max(self) -> None:
        mem_id = self._seed_memory("My favorite language is Java.", strength=0.95)

        self._run_reflection(
            user_message="I still prefer Java.",
            assistant_response="Great!",
        )

        updated = self.chroma_service.get_memory_by_id(mem_id)
        assert updated is not None
        self.assertAlmostEqual(updated["metadata"]["memory_strength"], 1.0, places=2)

    # ------------------------------------------------------------------
    # Assistant quality metrics stored correctly
    # ------------------------------------------------------------------
    def test_assistant_quality_metrics(self) -> None:
        record = self._run_reflection(
            user_message="Hello",
            assistant_response="Hi there!",
        )

        aq = record.reflection.assistant_quality
        self.assertIsInstance(aq, AssistantQuality)
        self.assertGreaterEqual(aq.correctness, 0.0)
        self.assertGreaterEqual(aq.clarity, 0.0)
        self.assertGreaterEqual(aq.confidence, 0.0)
        self.assertGreaterEqual(aq.hallucination_risk, 0.0)
        self.assertLessEqual(aq.hallucination_risk, 1.0)

    # ------------------------------------------------------------------
    # Processing time is recorded
    # ------------------------------------------------------------------
    def test_processing_time_recorded(self) -> None:
        record = self._run_reflection(
            user_message="Hello",
            assistant_response="Hi!",
        )
        self.assertGreaterEqual(record.processing_time_ms, 0.0)

    # ------------------------------------------------------------------
    # Statistics aggregation
    # ------------------------------------------------------------------
    def test_reflection_statistics(self) -> None:
        self._run_reflection(
            user_message="Hello",
            assistant_response="Hi!",
        )
        self._run_reflection(
            user_message="I still prefer Java.",
            assistant_response="Great!",
        )

        stats = self.reflection_service.statistics()
        self.assertIsInstance(stats, ReflectionStatistics)
        self.assertGreaterEqual(stats.total_reflections, 2)
        self.assertIn("NO_ACTION", stats.action_counts)

    # ------------------------------------------------------------------
    # Recent reflections
    # ------------------------------------------------------------------
    def test_recent_reflections(self) -> None:
        self._run_reflection(
            user_message="Hello",
            assistant_response="Hi!",
        )
        self._run_reflection(
            user_message="I still prefer Java.",
            assistant_response="Great!",
        )

        recent = self.reflection_service.recent(limit=5)
        self.assertGreaterEqual(len(recent), 2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _run_reflection(
        self,
        user_message: str,
        assistant_response: str,
    ):
        return self._run_reflection_with_service(
            self.reflection_service,
            user_message,
            assistant_response,
        )

    @staticmethod
    def _run_reflection_with_service(
        service: ReflectionService,
        user_message: str,
        assistant_response: str,
    ):
        import asyncio
        return asyncio.run(
            service.reflect(
                user_message=user_message,
                assistant_response=assistant_response,
            )
        )


class BrokenLLMService:
    async def generate_text(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> str:
        raise RuntimeError("LLM unavailable")

    async def aclose(self) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
