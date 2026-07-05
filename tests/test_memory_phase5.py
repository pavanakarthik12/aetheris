"""Phase 5 memory evaluator tests."""

from __future__ import annotations

import asyncio
import json
import math
import sys
import unittest
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.dependencies import get_llm_service, get_memory_evolution_service, get_memory_service
from backend.app.main import app
from backend.app.services.chroma_service import ChromaServiceError
from backend.app.services.memory_evolution_service import MemoryEvolutionService
from backend.app.services.memory_service import MemoryService


class FakeEmbeddingService:
    """Deterministic embedding stub for semantic-memory tests."""

    async def embed_text(self, text: str) -> list[float]:
        normalized = text.lower()

        if any(token in normalized for token in ("cat", "feline", "kitten")):
            return [1.0, 0.0, 0.0]
        if any(token in normalized for token in ("dog", "canine", "puppy")):
            return [0.0, 1.0, 0.0]
        if any(token in normalized for token in ("coffee", "espresso", "latte")):
            return [0.0, 0.0, 1.0]
        if any(token in normalized for token in ("aetheris", "project", "kubernetes", "interview", "java")):
            return [0.8, 0.1, 0.1]

        return [0.33, 0.33, 0.34]


class FakeChromaService:
    """In-memory stand-in for the ChromaDB service boundary."""

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

    def search_memory(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        scored_records: list[dict[str, Any]] = []

        for record in self._records.values():
            score = _cosine_similarity(query_embedding, record["embedding"])
            scored_records.append(
                {
                    "id": record["id"],
                    "document": record["document"],
                    "score": score,
                    "metadata": record["metadata"],
                }
            )

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
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return round(numerator / (left_norm * right_norm), 6)


class FakeLLMService:
    """Return deterministic JSON decisions for the evaluator and dummy chat output."""

    async def generate_with_context(
        self,
        user_message: str,
        system_prompt: str,
        memory_context: str = "",
    ) -> str:
        if "Memory Evaluator" in system_prompt:
            latest_message = self._extract_latest_message(user_message)
            lowered = latest_message.lower()

            if any(token in lowered for token in ("hello", "thank you")):
                return json.dumps(
                    {
                        "store": False,
                        "importance": 0.05,
                        "category": "Other",
                        "reason": "Greeting or acknowledgement with no long-term value.",
                    }
                )

            if "5 + 5" in lowered:
                return json.dumps(
                    {
                        "store": False,
                        "importance": 0.02,
                        "category": "Other",
                        "reason": "One-off calculation with no durable memory value.",
                    }
                )

            if "recursion" in lowered:
                return json.dumps(
                    {
                        "store": False,
                        "importance": 0.10,
                        "category": "Other",
                        "reason": "Temporary explanation request, not a lasting memory.",
                    }
                )

            if "building aetheris" in lowered or "project" in lowered:
                return json.dumps(
                    {
                        "store": True,
                        "importance": 0.95,
                        "category": "Project",
                        "reason": "The user shared an ongoing project.",
                    }
                )

            if "favorite language" in lowered or "language is java" in lowered:
                return json.dumps(
                    {
                        "store": True,
                        "importance": 0.90,
                        "category": "Preference",
                        "reason": "The user revealed a stable programming preference.",
                    }
                )

            if "goal" in lowered or "become an ai engineer" in lowered:
                return json.dumps(
                    {
                        "store": True,
                        "importance": 0.94,
                        "category": "Goal",
                        "reason": "The user described a long-term career goal.",
                    }
                )

            if "learning kubernetes" in lowered:
                return json.dumps(
                    {
                        "store": True,
                        "importance": 0.88,
                        "category": "Skill",
                        "reason": "The user is actively learning a durable technical skill.",
                    }
                )

            return json.dumps(
                {
                    "store": False,
                    "importance": 0.10,
                    "category": "Other",
                    "reason": "No durable memory value detected.",
                }
            )

        return "ok"

    @staticmethod
    def _extract_latest_message(prompt: str) -> str:
        marker = "Latest user message:\n"
        if marker not in prompt:
            return prompt.strip()

        tail = prompt.split(marker, 1)[1]
        return tail.split("\n\nRespond with only the JSON object.", 1)[0].strip()

    async def aclose(self) -> None:
        return None


class MemoryPhase5Tests(unittest.TestCase):
    """Validate the memory-evaluator gate on top of the existing chat flow."""

    def setUp(self) -> None:
        self.embedding_service = FakeEmbeddingService()
        self.chroma_service = FakeChromaService()
        self.memory_service = MemoryService(
            embedding_service=self.embedding_service,
            chroma_service=self.chroma_service,
        )
        self.llm_service = FakeLLMService()
        self.evolution_service = MemoryEvolutionService(
            memory_service=self.memory_service,
            chroma_service=self.chroma_service,
            embedding_service=self.embedding_service,
        )

        app.dependency_overrides[get_llm_service] = lambda: self.llm_service
        app.dependency_overrides[get_memory_service] = lambda: self.memory_service
        app.dependency_overrides[get_memory_evolution_service] = lambda: self.evolution_service
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_chat_stores_worthy_memories(self) -> None:
        messages = [
            "I'm building Aetheris.",
            "My favorite language is Java.",
            "My goal is to become an AI Engineer.",
            "I started learning Kubernetes.",
        ]

        for message in messages:
            response = self.client.post("/api/chat", json={"message": message})
            self.assertEqual(response.status_code, 200)

        memories = self.chroma_service.list_all_memories()
        stored_texts = {item["document"] for item in memories}

        # Phase 6 evolution may merge related memories; at minimum we
        # should have stored Aetheris (1) + Java/Goal (merged or separate)
        # + Kubernetes (1) = 3 or 4.
        self.assertGreaterEqual(len(memories), 3)
        self.assertTrue(
            any("building Aetheris" in t for t in stored_texts),
            "Expected a memory containing 'building Aetheris'",
        )
        self.assertTrue(
            any("Java" in t for t in stored_texts),
            "Expected a memory mentioning Java",
        )
        self.assertTrue(
            any("Kubernetes" in t for t in stored_texts),
            "Expected a memory mentioning Kubernetes",
        )

        retrieval = asyncio.run(self.memory_service.search_memory("What project am I building?", top_k=3))
        self.assertGreaterEqual(len(retrieval), 1)
        self.assertTrue(
            "building Aetheris" in retrieval[0]["document"],
            f"Expected search result to mention 'building Aetheris', got: {retrieval[0]['document']}",
        )

    def test_chat_skips_trivial_messages(self) -> None:
        messages = [
            "Hello",
            "Thank you",
            "What's 5 + 5?",
            "Can you explain recursion?",
        ]

        for message in messages:
            response = self.client.post("/api/chat", json={"message": message})
            self.assertEqual(response.status_code, 200)

        self.assertEqual(len(self.chroma_service.list_all_memories()), 0)


if __name__ == "__main__":
    unittest.main()