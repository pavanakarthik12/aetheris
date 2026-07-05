"""Tests for ImmediateMemoryProcessor — memory mutations before LLM response."""

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

from backend.app.schemas.chat import ImmediateMemoryResult, MemoryActionType
from backend.app.services.memory_evaluator import MemoryEvaluation
from backend.app.services.chroma_service import ChromaServiceError
from backend.app.services.immediate_memory_processor import (
    ImmediateMemoryProcessor,
    ImmediateMemoryProcessorError,
)
from backend.app.services.memory_evaluator import MemoryEvaluatorService
from backend.app.services.memory_evolution_service import MemoryEvolutionService


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeEmbeddingService:
    async def embed_text(self, text: str) -> list[float]:
        if "cat" in text.lower() or "django" in text.lower() or "java" in text.lower():
            return [1.0, 0.0, 0.0]
        return [0.5, 0.3, 0.2]


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
            raise ChromaServiceError("Not found", 404)
        del self._records[memory_id]

    def update_memory(self, memory_id: str, embedding: list[float], document: str, metadata: dict[str, Any]) -> None:
        if memory_id not in self._records:
            raise ChromaServiceError("Not found", 404)
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
                results.append({
                    "id": rec["id"],
                    "document": rec["document"],
                    "metadata": dict(meta),
                })
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


class FakeMemoryEvaluator(MemoryEvaluatorService):
    def __init__(self, store: bool = True, category: str = "Other", importance: float = 0.5) -> None:
        super().__init__(llm_service=None)  # type: ignore[arg-type]
        self._store = store
        self._category = category
        self._importance = importance

    async def evaluate_memory(self, message_text: str) -> MemoryEvaluation:
        return MemoryEvaluation(store=self._store, category=self._category, importance=self._importance, reason="test")


class FakeMemoryEvolution(MemoryEvolutionService):
    def __init__(self, action: str = "CREATE") -> None:
        super().__init__(memory_service=None, chroma_service=None, embedding_service=None)  # type: ignore[arg-type]
        self._action = action
        self._target_id: str | None = None

    def set_target(self, target_id: str) -> None:
        self._target_id = target_id

    async def decide_evolution(self, memory_text: str, existing_evaluation: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {"action": self._action}
        if self._target_id:
            result["target_id"] = self._target_id
        return result

    async def create_memory(self, memory_text: str, metadata: dict[str, Any]) -> dict[str, Any]:
        return {"memory_id": str(uuid.uuid4()), "status": "created"}

    async def update_memory(self, memory_id: str, new_text: str, new_metadata: dict[str, Any]) -> dict[str, Any]:
        return {"memory_id": memory_id, "status": "updated"}

    async def archive_memory(self, memory_id: str) -> dict[str, Any]:
        return {"memory_id": memory_id, "status": "archived"}

    async def get_memory_document(self, memory_id: str) -> str | None:
        return "Existing memory content."


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class ImmediateMemoryProcessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embeddings = FakeEmbeddingService()
        self.chroma = FakeChromaService()
        self.memory_service = FakeMemoryService(self.chroma, self.embeddings)
        self.evaluator = FakeMemoryEvaluator(store=True, category="Fact", importance=0.3)
        self.evolution = FakeMemoryEvolution(action="CREATE")

        self.processor = ImmediateMemoryProcessor(
            memory_service=self.memory_service,
            memory_evaluator=self.evaluator,
            memory_evolution=self.evolution,
            chroma_service=self.chroma,
            embedding_service=self.embeddings,
        )

    # --------------------------------------------------------------
    # Empty / blank messages
    # --------------------------------------------------------------
    def test_empty_message_returns_skip(self) -> None:
        result = self._process("")
        self.assertEqual(result.action, MemoryActionType.SKIP)
        self.assertTrue(result.success)

    def test_whitespace_message_returns_skip(self) -> None:
        result = self._process("   ")
        self.assertEqual(result.action, MemoryActionType.SKIP)
        self.assertTrue(result.success)

    # --------------------------------------------------------------
    # Explicit delete
    # --------------------------------------------------------------
    def test_delete_memory(self) -> None:
        mid = str(uuid.uuid4())
        self.chroma.add_memory(mid, [1.0, 0.0, 0.0], "I like cats", {"source": "user"})

        result = self._process("forget that cat memory")
        self.assertEqual(result.action, MemoryActionType.DELETE)
        self.assertTrue(result.success)
        self.assertEqual(result.memory_id, mid)

    def test_delete_memory_no_match(self) -> None:
        # Embedding far from the "forget" query embedding (0.5,0.3,0.2)
        # gives a cosine similarity well below the 0.5 threshold.
        self.chroma.add_memory(str(uuid.uuid4()), [0.1, 0.1, 0.1], "Something unrelated", {"source": "user"})

        result = self._process("forget something that doesn't exist")
        self.assertEqual(result.action, MemoryActionType.SKIP)
        self.assertTrue(result.success)

    # --------------------------------------------------------------
    # Explicit archive
    # --------------------------------------------------------------
    def test_archive_memory(self) -> None:
        mid = str(uuid.uuid4())
        self.chroma.add_memory(mid, [1.0, 0.0, 0.0], "I like cats", {"source": "user"})

        result = self._process("archive that cat memory")
        self.assertEqual(result.action, MemoryActionType.ARCHIVE)
        self.assertTrue(result.success)
        self.assertEqual(result.memory_id, mid)

    # --------------------------------------------------------------
    # Evaluation skip
    # --------------------------------------------------------------
    def test_evaluation_skip(self) -> None:
        self.evaluator = FakeMemoryEvaluator(store=False)
        self.processor = ImmediateMemoryProcessor(
            memory_service=self.memory_service,
            memory_evaluator=self.evaluator,
            memory_evolution=self.evolution,
            chroma_service=self.chroma,
            embedding_service=self.embeddings,
        )

        result = self._process("Just a random greeting")
        self.assertEqual(result.action, MemoryActionType.SKIP)
        self.assertTrue(result.success)

    # --------------------------------------------------------------
    # CREATE action
    # --------------------------------------------------------------
    def test_create_memory(self) -> None:
        self.evolution = FakeMemoryEvolution(action="CREATE")
        self.processor = ImmediateMemoryProcessor(
            memory_service=self.memory_service,
            memory_evaluator=self.evaluator,
            memory_evolution=self.evolution,
            chroma_service=self.chroma,
            embedding_service=self.embeddings,
        )

        result = self._process("I love programming in Python")
        self.assertEqual(result.action, MemoryActionType.CREATE)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.memory_id)

    # --------------------------------------------------------------
    # UPDATE action
    # --------------------------------------------------------------
    def test_update_memory(self) -> None:
        mid = str(uuid.uuid4())
        self.chroma.add_memory(mid, [1.0, 0.0, 0.0], "I like Java", {"source": "user"})

        self.evolution = FakeMemoryEvolution(action="UPDATE")
        self.evolution.set_target(mid)
        self.processor = ImmediateMemoryProcessor(
            memory_service=self.memory_service,
            memory_evaluator=self.evaluator,
            memory_evolution=self.evolution,
            chroma_service=self.chroma,
            embedding_service=self.embeddings,
        )

        result = self._process("I love Java")
        self.assertEqual(result.action, MemoryActionType.UPDATE)
        self.assertTrue(result.success)
        self.assertEqual(result.memory_id, mid)

    def test_update_without_target_id_returns_error(self) -> None:
        self.evolution = FakeMemoryEvolution(action="UPDATE")
        self.processor = ImmediateMemoryProcessor(
            memory_service=self.memory_service,
            memory_evaluator=self.evaluator,
            memory_evolution=self.evolution,
            chroma_service=self.chroma,
            embedding_service=self.embeddings,
        )

        result = self._process("I love Java")
        self.assertEqual(result.action, MemoryActionType.ERROR)
        self.assertFalse(result.success)

    # --------------------------------------------------------------
    # MERGE action
    # --------------------------------------------------------------
    def test_merge_memory(self) -> None:
        mid = str(uuid.uuid4())
        self.chroma.add_memory(mid, [1.0, 0.0, 0.0], "Existing memory content.", {"source": "user"})

        self.evolution = FakeMemoryEvolution(action="MERGE")
        self.evolution.set_target(mid)
        self.processor = ImmediateMemoryProcessor(
            memory_service=self.memory_service,
            memory_evaluator=self.evaluator,
            memory_evolution=self.evolution,
            chroma_service=self.chroma,
            embedding_service=self.embeddings,
        )

        result = self._process("Additional detail about cats")
        self.assertEqual(result.action, MemoryActionType.MERGE)
        self.assertTrue(result.success)
        self.assertEqual(result.memory_id, mid)

    def test_merge_without_target_id_returns_error(self) -> None:
        self.evolution = FakeMemoryEvolution(action="MERGE")
        self.processor = ImmediateMemoryProcessor(
            memory_service=self.memory_service,
            memory_evaluator=self.evaluator,
            memory_evolution=self.evolution,
            chroma_service=self.chroma,
            embedding_service=self.embeddings,
        )

        result = self._process("Additional detail")
        self.assertEqual(result.action, MemoryActionType.ERROR)
        self.assertFalse(result.success)

    # --------------------------------------------------------------
    # Unknown evolution action -> SKIP
    # --------------------------------------------------------------
    def test_unknown_evolution_action_returns_skip(self) -> None:
        self.evolution = FakeMemoryEvolution(action="SOMETHING_ELSE")
        self.processor = ImmediateMemoryProcessor(
            memory_service=self.memory_service,
            memory_evaluator=self.evaluator,
            memory_evolution=self.evolution,
            chroma_service=self.chroma,
            embedding_service=self.embeddings,
        )

        result = self._process("I like cats")
        self.assertEqual(result.action, MemoryActionType.SKIP)
        self.assertTrue(result.success)

    # --------------------------------------------------------------
    # Error handling — evolution service raises
    # --------------------------------------------------------------
    def test_evolution_error_returns_error_result(self) -> None:
        broken = BrokenMemoryEvolution()
        self.processor = ImmediateMemoryProcessor(
            memory_service=self.memory_service,
            memory_evaluator=self.evaluator,
            memory_evolution=broken,
            chroma_service=self.chroma,
            embedding_service=self.embeddings,
        )

        result = self._process("I like cats")
        self.assertEqual(result.action, MemoryActionType.ERROR)
        self.assertFalse(result.success)

    # --------------------------------------------------------------
    # Error handling — execution failure
    # --------------------------------------------------------------
    def test_execution_error_returns_error_result(self) -> None:
        self.evolution = FakeMemoryEvolution(action="CREATE")
        self.evolution.create_memory = None  # type: ignore[method-assign]

        async def broken_create(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("Creation failed")

        self.evolution.create_memory = broken_create  # type: ignore[assignment]

        self.processor = ImmediateMemoryProcessor(
            memory_service=self.memory_service,
            memory_evaluator=self.evaluator,
            memory_evolution=self.evolution,
            chroma_service=self.chroma,
            embedding_service=self.embeddings,
        )

        result = self._process("I like cats")
        self.assertEqual(result.action, MemoryActionType.ERROR)
        self.assertFalse(result.success)
        self.assertIn("Creation failed", result.error or "")

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------
    def _process(self, msg: str) -> ImmediateMemoryResult:
        import asyncio
        return asyncio.run(self.processor.process_message(msg))


class BrokenMemoryEvolution(MemoryEvolutionService):
    def __init__(self) -> None:
        super().__init__(memory_service=None, chroma_service=None, embedding_service=None)  # type: ignore[arg-type]

    async def decide_evolution(self, memory_text: str, existing_evaluation: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("Evolution service unavailable")

    async def create_memory(self, memory_text: str, metadata: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("Unavailable")

    async def update_memory(self, memory_id: str, new_text: str, new_metadata: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("Unavailable")

    async def archive_memory(self, memory_id: str) -> dict[str, Any]:
        raise RuntimeError("Unavailable")


if __name__ == "__main__":
    unittest.main()
