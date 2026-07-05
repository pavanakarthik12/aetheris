"""Tests for the Memory Resolution Engine (single-value attribute conflict resolution)."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

import unittest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.memory_resolver import (
    SINGLE_VALUE_ATTRIBUTES,
    extract_attribute,
    resolve_conflict,
)


class FakeChromaService:
    """Minimal in-memory ChromaService fake that supports metadata lookups."""

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

    def update_memory(
        self,
        memory_id: str,
        embedding: list[float],
        document: str,
        metadata: dict[str, Any],
    ) -> None:
        if memory_id not in self._records:
            raise RuntimeError("Not found")
        self._records[memory_id] = {
            "id": memory_id,
            "embedding": embedding,
            "document": document,
            "metadata": metadata,
        }

    def get_memory_by_id(self, memory_id: str) -> dict[str, Any] | None:
        rec = self._records.get(memory_id)
        if not rec:
            return None
        return {"id": rec["id"], "document": rec["document"], "metadata": rec["metadata"]}

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

    def search_memory(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        results = []
        for rec in self._records.values():
            results.append({
                "id": rec["id"],
                "document": rec["document"],
                "score": 0.5,
                "metadata": rec["metadata"],
            })
        return results[:top_k]

    def list_all_memories(self) -> list[dict[str, Any]]:
        return [
            {"id": r["id"], "document": r["document"], "metadata": r["metadata"]}
            for r in self._records.values()
        ]

    def delete_memory(self, memory_id: str) -> None:
        if memory_id not in self._records:
            raise RuntimeError("Not found")
        del self._records[memory_id]


class ExtractAttributeTests(unittest.TestCase):
    """Test attribute extraction from memory text."""

    def test_extract_name(self) -> None:
        self.assertEqual(extract_attribute("My name is John Paul."), "name")
        self.assertEqual(extract_attribute("My name is Creator."), "name")
        self.assertEqual(extract_attribute("my name is alice"), "name")

    def test_extract_age(self) -> None:
        self.assertEqual(extract_attribute("I am 25 years old."), "age")
        self.assertEqual(extract_attribute("i'm 30 yo"), "age")

    def test_extract_birthday(self) -> None:
        self.assertEqual(extract_attribute("My birthday is Jan 1st."), "birthday")
        self.assertEqual(extract_attribute("i was born on 1990-05-15"), "birthday")

    def test_extract_favorite_language(self) -> None:
        self.assertEqual(
            extract_attribute("My favorite programming language is Rust."),
            "favorite_programming_language",
        )
        self.assertEqual(
            extract_attribute("my favorite language is Python"),
            "favorite_programming_language",
        )

    def test_extract_favorite_subject(self) -> None:
        self.assertEqual(
            extract_attribute("My favorite subject is Mathematics."),
            "favorite_subject",
        )

    def test_extract_favorite_food(self) -> None:
        self.assertEqual(
            extract_attribute("My favorite food is Pizza."),
            "favorite_food",
        )

    def test_extract_city(self) -> None:
        self.assertEqual(extract_attribute("My city is New York."), "city")

    def test_extract_country(self) -> None:
        self.assertEqual(extract_attribute("My country is Canada."), "country")

    def test_extract_occupation(self) -> None:
        self.assertEqual(extract_attribute("I work as an engineer."), "occupation")
        self.assertEqual(extract_attribute("My occupation is teacher."), "occupation")

    def test_extract_school(self) -> None:
        self.assertEqual(extract_attribute("My school is Lincoln High."), "school")
        self.assertEqual(extract_attribute("i go to Westside School"), "school")

    def test_extract_university(self) -> None:
        self.assertEqual(extract_attribute("My university is MIT."), "university")
        self.assertEqual(extract_attribute("i study at Stanford"), "university")

    def test_extract_company(self) -> None:
        self.assertEqual(extract_attribute("My company is Acme Corp."), "company")
        self.assertEqual(extract_attribute("i work at Google"), "company")

    def test_extract_relationship_status(self) -> None:
        self.assertEqual(
            extract_attribute("I am married."),
            "relationship_status",
        )
        self.assertEqual(
            extract_attribute("My relationship status is single."),
            "relationship_status",
        )

    def test_no_attribute_detected(self) -> None:
        self.assertIsNone(extract_attribute("What is the weather like?"))
        self.assertIsNone(extract_attribute("Hello, how are you?"))
        self.assertIsNone(extract_attribute("Search my memories for Java."))
        self.assertIsNone(extract_attribute(""))

    def test_name_does_not_match_other_phrases(self) -> None:
        """'my name is' should not match in unrelated context."""
        self.assertIsNone(extract_attribute("What is the name of that movie?"))


class ResolveConflictTests(unittest.TestCase):
    """Test the conflict resolution logic directly."""

    def setUp(self) -> None:
        self.chroma = FakeChromaService()
        self.archived_ids: list[str] = []

    async def _archive_fn(self, memory_id: str) -> None:
        rec = self.chroma.get_memory_by_id(memory_id)
        if rec is None:
            raise RuntimeError(f"Memory {memory_id} not found")
        meta = dict(rec["metadata"])
        meta["status"] = "archived"
        self.chroma.update_memory(
            memory_id=memory_id,
            embedding=[0.0, 0.0, 0.0],
            document=rec["document"],
            metadata=meta,
        )
        self.archived_ids.append(memory_id)

    async def _seed(
        self,
        text: str,
        attribute: str | None = None,
        status: str = "active",
    ) -> str:
        mem_id = str(uuid.uuid4())
        metadata: dict[str, Any] = {
            "status": status,
            "version": 1,
            "created_at": "2026-01-01T00:00:00Z",
        }
        if attribute:
            metadata["attribute"] = attribute
        self.chroma.add_memory(mem_id, [0.5, 0.5, 0.5], text, metadata)
        return mem_id

    async def _resolve(self, text: str) -> dict[str, Any]:
        return await resolve_conflict(
            memory_text=text,
            chroma_service=self.chroma,
            archive_fn=self._archive_fn,
        )

    # --------------------------------------------------------------
    # No conflict cases
    # --------------------------------------------------------------

    def test_no_attribute_no_conflict(self) -> None:
        import asyncio
        result = asyncio.run(self._resolve("What is the weather?"))
        self.assertFalse(result["conflict_detected"])
        self.assertIsNone(result["archived_memory_id"])
        self.assertIsNone(result["attribute"])
        self.assertTrue(result["verified"])

    def test_no_existing_memory_no_conflict(self) -> None:
        import asyncio
        result = asyncio.run(self._resolve("My name is Alice."))
        self.assertFalse(result["conflict_detected"])
        self.assertEqual(result["attribute"], "name")
        self.assertTrue(result["verified"])

    def test_existing_archived_no_conflict(self) -> None:
        import asyncio
        asyncio.run(self._seed("John", attribute="name", status="archived"))
        result = asyncio.run(self._resolve("My name is Alice."))
        self.assertFalse(result["conflict_detected"])
        self.assertEqual(result["attribute"], "name")
        self.assertTrue(result["verified"])

    # --------------------------------------------------------------
    # Conflict cases
    # --------------------------------------------------------------

    def test_conflict_archives_previous_active(self) -> None:
        import asyncio
        prev_id = asyncio.run(self._seed("My name is John Paul.", attribute="name"))
        result = asyncio.run(self._resolve("My name is Creator."))
        self.assertTrue(result["conflict_detected"])
        self.assertEqual(result["archived_memory_id"], prev_id)
        self.assertEqual(result["attribute"], "name")
        self.assertTrue(result["verified"])
        # Confirm previous memory is now archived
        prev = asyncio.run(
            asyncio.to_thread(self.chroma.get_memory_by_id, prev_id)
        )
        self.assertIsNotNone(prev)
        self.assertEqual(prev["metadata"].get("status"), "archived")

    def test_conflict_only_archives_one(self) -> None:
        import asyncio
        prev_id = asyncio.run(self._seed("John", attribute="name"))
        other_id = asyncio.run(self._seed("Python", attribute="favorite_programming_language"))
        result = asyncio.run(self._resolve("My name is Bob."))
        self.assertTrue(result["conflict_detected"])
        self.assertEqual(result["archived_memory_id"], prev_id)
        self.assertTrue(result["verified"])
        # Other memory should remain active
        other = asyncio.run(
            asyncio.to_thread(self.chroma.get_memory_by_id, other_id)
        )
        self.assertEqual(other["metadata"].get("status"), "active")

    def test_different_attributes_no_conflict(self) -> None:
        import asyncio
        asyncio.run(self._seed("Java", attribute="favorite_programming_language"))
        result = asyncio.run(self._resolve("My name is Alice."))
        self.assertFalse(result["conflict_detected"])
        self.assertEqual(result["attribute"], "name")

    def test_favorite_language_conflict(self) -> None:
        import asyncio
        prev_id = asyncio.run(self._seed("Java", attribute="favorite_programming_language"))
        result = asyncio.run(
            self._resolve("My favorite programming language is Python.")
        )
        self.assertTrue(result["conflict_detected"])
        self.assertEqual(result["archived_memory_id"], prev_id)
        self.assertEqual(result["attribute"], "favorite_programming_language")
        self.assertTrue(result["verified"])

    def test_favorite_food_conflict(self) -> None:
        import asyncio
        prev_id = asyncio.run(self._seed("Pizza", attribute="favorite_food"))
        result = asyncio.run(self._resolve("My favorite food is Sushi."))
        self.assertTrue(result["conflict_detected"])
        self.assertEqual(result["archived_memory_id"], prev_id)
        self.assertTrue(result["verified"])

    def test_city_conflict(self) -> None:
        import asyncio
        prev_id = asyncio.run(self._seed("New York", attribute="city"))
        result = asyncio.run(self._resolve("My city is San Francisco."))
        self.assertTrue(result["conflict_detected"])
        self.assertEqual(result["archived_memory_id"], prev_id)
        self.assertTrue(result["verified"])

    def test_occupation_conflict(self) -> None:
        import asyncio
        prev_id = asyncio.run(self._seed("Engineer", attribute="occupation"))
        result = asyncio.run(self._resolve("I work as a designer."))
        self.assertTrue(result["conflict_detected"])
        self.assertEqual(result["archived_memory_id"], prev_id)
        self.assertTrue(result["verified"])

    # --------------------------------------------------------------
    # Scenario-based tests
    # --------------------------------------------------------------

    def test_scenario1_name_change(self) -> None:
        """Store 'My name is John Paul', then 'My name is Creator'.
        Expected: Creator is active, John Paul is archived.
        """
        import asyncio

        # First name
        result1 = asyncio.run(self._resolve("My name is John Paul."))
        self.assertFalse(result1["conflict_detected"])

        # Store the first memory
        first_id = asyncio.run(self._seed("My name is John Paul.", attribute="name"))

        # Second name — should archive the first
        result2 = asyncio.run(self._resolve("My name is Creator."))
        self.assertTrue(result2["conflict_detected"])
        self.assertEqual(result2["archived_memory_id"], first_id)
        self.assertTrue(result2["verified"])

        # Verify only one active name
        active = asyncio.run(
            asyncio.to_thread(
                self.chroma.get_memories_by_metadata,
                {"attribute": "name", "status": "active"},
            )
        )
        self.assertEqual(len(active), 0, "New memory isn't stored yet — only resolved")

    def test_scenario2_favorite_language_change(self) -> None:
        """Store 'My favorite language is Java', then 'I now prefer Python'
        (not captured by attribute patterns but by evolution update).
        The resolver focuses on the explicit attribute case.
        """
        import asyncio
        asyncio.run(self._seed("Java", attribute="favorite_programming_language"))
        result = asyncio.run(
            self._resolve("My favorite programming language is Python.")
        )
        self.assertTrue(result["conflict_detected"])
        self.assertTrue(result["verified"])

    def test_scenario3_delete_active_name(self) -> None:
        """Delete the active name.  No active memories should exist for name."""
        import asyncio
        mem_id = asyncio.run(self._seed("John", attribute="name"))

        # Delete the active memory
        asyncio.run(asyncio.to_thread(self.chroma.delete_memory, mem_id))

        # Verify no active name exists
        active = asyncio.run(
            asyncio.to_thread(
                self.chroma.get_memories_by_metadata,
                {"attribute": "name", "status": "active"},
            )
        )
        self.assertEqual(len(active), 0)


class MultipleSameAttributeTests(unittest.TestCase):
    """Test that only one active memory exists per attribute."""

    def setUp(self) -> None:
        self.chroma = FakeChromaService()

    def _add(self, text: str, attr: str, status: str = "active") -> str:
        mid = str(uuid.uuid4())
        self.chroma.add_memory(mid, [0.5, 0.5, 0.5], text, {
            "attribute": attr,
            "status": status,
            "version": 1,
        })
        return mid

    def test_archive_via_update(self) -> None:
        """The resolver should not find two active memories for same attribute."""
        mid1 = self._add("John", "name")
        mid2 = self._add("Alice", "name", status="archived")

        active = self.chroma.get_memories_by_metadata(
            {"attribute": "name", "status": "active"},
        )
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["id"], mid1)


class MemoryEvolutionServiceResolutionTests(unittest.TestCase):
    """Integration test: resolution through MemoryEvolutionService.create_memory."""

    def setUp(self) -> None:
        self.chroma = FakeChromaService()

        class FakeEmbeddingService:
            async def embed_text(self, text: str) -> list[float]:
                return [0.5, 0.5, 0.5]

        class FakeMemoryService:
            def __init__(self, chroma: FakeChromaService, emb: FakeEmbeddingService):
                self._chroma = chroma
                self._embeddings = emb

            async def save_memory(
                self,
                memory_text: str,
                metadata: dict[str, Any] | None = None,
            ) -> dict[str, Any]:
                mem_id = str(uuid.uuid4())
                emb = await self._embeddings.embed_text(memory_text)
                self._chroma.add_memory(mem_id, emb, memory_text, metadata or {})
                return {"memory_id": mem_id, "status": "saved", "version": 1}

            async def search_memory(
                self,
                query: str,
                top_k: int = 5,
                include_archived: bool = False,
            ) -> list[dict[str, Any]]:
                results = self._chroma.search_memory([0.5, 0.5, 0.5], top_k=top_k)
                if not include_archived:
                    from backend.app.services.memory_service import _filter_active
                    results = _filter_active(results)
                return results

            def list_memories(self, include_archived: bool = False) -> list[dict[str, Any]]:
                mems = self._chroma.list_all_memories()
                if not include_archived:
                    from backend.app.services.memory_service import _filter_active
                    mems = _filter_active(mems)
                return mems

        self.emb_service = FakeEmbeddingService()
        self.mem_service = FakeMemoryService(self.chroma, self.emb_service)

        from backend.app.services.memory_evolution_service import MemoryEvolutionService

        self.evolution = MemoryEvolutionService(
            memory_service=self.mem_service,
            chroma_service=self.chroma,
            embedding_service=self.emb_service,
        )

    async def _create(self, text: str, category: str = "Fact") -> dict[str, Any]:
        return await self.evolution.create_memory(text, metadata={
            "source": "chat",
            "category": category,
            "importance": 0.5,
            "reason": "test",
        })

    def test_create_name_twice_archives_first(self) -> None:
        import asyncio

        # First: create "My name is John Paul"
        first = asyncio.run(self._create("My name is John Paul."))
        first_id = first["memory_id"]

        # Verify it's stored as active with attribute=name
        first_rec = asyncio.run(
            asyncio.to_thread(self.chroma.get_memory_by_id, first_id)
        )
        self.assertIsNotNone(first_rec)
        self.assertEqual(first_rec["metadata"].get("status"), "active")
        self.assertEqual(first_rec["metadata"].get("attribute"), "name")

        # Second: create "My name is Creator"
        second = asyncio.run(self._create("My name is Creator."))
        second_id = second["memory_id"]

        # Verify first is now archived
        first_rec2 = asyncio.run(
            asyncio.to_thread(self.chroma.get_memory_by_id, first_id)
        )
        self.assertEqual(first_rec2["metadata"].get("status"), "archived")

        # Verify second is active
        second_rec = asyncio.run(
            asyncio.to_thread(self.chroma.get_memory_by_id, second_id)
        )
        self.assertEqual(second_rec["metadata"].get("status"), "active")
        self.assertEqual(second_rec["metadata"].get("attribute"), "name")

        # Verify only ONE active name
        active = self.chroma.get_memories_by_metadata(
            {"attribute": "name", "status": "active"},
        )
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["id"], second_id)

    def test_create_different_attributes_no_conflict(self) -> None:
        import asyncio

        first = asyncio.run(self._create("My name is John."))
        second = asyncio.run(self._create("My favorite food is Pizza."))

        for mid in (first["memory_id"], second["memory_id"]):
            rec = asyncio.run(
                asyncio.to_thread(self.chroma.get_memory_by_id, mid)
            )
            self.assertEqual(rec["metadata"].get("status"), "active")

    def test_list_memories_excludes_archived(self) -> None:
        import asyncio
        asyncio.run(self._create("My name is John."))
        asyncio.run(self._create("My name is Creator."))  # archives John

        # list_memories should only return active (Creator)
        all_mems = self.mem_service.list_memories()
        self.assertEqual(len(all_mems), 1)
        self.assertIn("Creator", all_mems[0]["document"])


if __name__ == "__main__":
    unittest.main()
