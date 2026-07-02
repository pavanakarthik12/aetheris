"""Phase 3 memory foundation tests."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.dependencies import get_memory_service
from backend.app.main import app
from backend.app.services.chroma_service import ChromaServiceError
from backend.app.services.memory_service import MemoryService


class FakeEmbeddingService:
    """Deterministic embedding stub for semantic-memory tests."""

    def embed_text(self, text: str) -> list[float]:
        normalized = text.lower()

        if any(token in normalized for token in ("cat", "feline", "kitten")):
            return [1.0, 0.0, 0.0]
        if any(token in normalized for token in ("dog", "canine", "puppy")):
            return [0.0, 1.0, 0.0]
        if any(token in normalized for token in ("coffee", "espresso", "latte")):
            return [0.0, 0.0, 1.0]

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
        if memory_id in self._records:
            raise ChromaServiceError(f"Memory with id '{memory_id}' already exists.", status_code=409)

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
        if memory_id not in self._records:
            raise ChromaServiceError(f"Memory '{memory_id}' not found.", status_code=404)

        del self._records[memory_id]

    def list_all_memories(self) -> list[dict[str, Any]]:
        return [
            {
                "id": record["id"],
                "document": record["document"],
                "metadata": record["metadata"],
            }
            for record in self._records.values()
        ]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return round(numerator / (left_norm * right_norm), 6)


class MemoryPhase3Tests(unittest.TestCase):
    """Validate the save/search/list/delete memory flow."""

    def setUp(self) -> None:
        self.embedding_service = FakeEmbeddingService()
        self.chroma_service = FakeChromaService()
        self.memory_service = MemoryService(
            embedding_service=self.embedding_service,
            chroma_service=self.chroma_service,
        )

        app.dependency_overrides[get_memory_service] = lambda: self.memory_service
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_save_memory_endpoint_persists_memory(self) -> None:
        response = self.client.post(
            "/api/memory/save",
            json={
                "memory_text": "The cat likes warm sunlight.",
                "metadata": {
                    "source": "user",
                    "tags": "animals,cat",
                    "importance": 0.8,
                },
            },
        )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["status"], "saved")
        self.assertIn("memory_id", payload)
        self.assertIn("created_at", payload)
        self.assertEqual(len(self.chroma_service.list_all_memories()), 1)

    def test_search_memory_endpoint_returns_semantic_match(self) -> None:
        save_response = self.client.post(
            "/api/memory/save",
            json={"memory_text": "A kitten is resting on the sofa."},
        )
        memory_id = save_response.json()["memory_id"]

        response = self.client.post(
            "/api/memory/search",
            json={"query": "The feline is sleeping on the couch.", "top_k": 3},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["results"][0]["id"], memory_id)
        self.assertEqual(payload["results"][0]["document"], "A kitten is resting on the sofa.")
        self.assertGreaterEqual(payload["results"][0]["score"], 0.99)

    def test_list_memories_endpoint_returns_all_memories(self) -> None:
        self.client.post("/api/memory/save", json={"memory_text": "Coffee helps me focus."})
        self.client.post("/api/memory/save", json={"memory_text": "The dog ran in the yard."})

        response = self.client.get("/api/memory/list")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total"], 2)
        self.assertEqual(len(payload["memories"]), 2)

    def test_delete_memory_endpoint_removes_memory(self) -> None:
        save_response = self.client.post(
            "/api/memory/save",
            json={"memory_text": "Delete this memory after verification."},
        )
        memory_id = save_response.json()["memory_id"]

        delete_response = self.client.delete(f"/api/memory/{memory_id}")
        self.assertEqual(delete_response.status_code, 200)
        self.assertEqual(delete_response.json()["status"], "deleted")

        list_response = self.client.get("/api/memory/list")
        self.assertEqual(list_response.json()["total"], 0)

    def test_save_memory_endpoint_rejects_empty_text(self) -> None:
        response = self.client.post(
            "/api/memory/save",
            json={"memory_text": ""},
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()