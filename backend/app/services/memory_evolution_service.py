"""Memory Evolution Engine — decides CREATE / UPDATE / MERGE / ARCHIVE.

Replaces the flat "always-save" pattern with an intelligent pipeline that
preserves version history and prevents duplicate or contradictory memories.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from .chroma_service import ChromaService, ChromaServiceError
from .embedding_service import EmbeddingService, EmbeddingServiceError
from .memory_service import MemoryService

logger = logging.getLogger(__name__)


class MemoryEvolutionServiceError(RuntimeError):
    """Raised when an evolution operation cannot be completed."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class MemoryEvolutionService:
    """Memory Evolution Engine — intelligently manage memory lifecycle.

    Every memory passes through this engine which decides whether to
    CREATE, UPDATE, MERGE, or ARCHIVE based on semantic similarity and
    conflict analysis.  Version history is preserved indefinitely.
    """

    def __init__(
        self,
        memory_service: MemoryService,
        chroma_service: ChromaService,
        embedding_service: EmbeddingService,
    ) -> None:
        self._memory_service = memory_service
        self._chroma_service = chroma_service
        self._embedding_service = embedding_service

    # ------------------------------------------------------------------
    # Public API — required methods
    # ------------------------------------------------------------------

    async def detect_related_memories(
        self,
        memory_text: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return semantically related memories via vector search.

        Args:
            memory_text: Text to search with.
            top_k:       Maximum results (default 5).

        Returns:
            List of memory dicts from ``MemoryService.search_memory``.
        """
        return await self._memory_service.search_memory(memory_text, top_k)

    async def detect_conflicts(
        self,
        incoming_text: str,
        existing_memories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Classify each existing memory's relationship to the incoming text.

        Args:
            incoming_text:      The new message to classify.
            existing_memories:  Results from ``detect_related_memories``.

        Returns:
            List of dicts with keys:
            ``memory_id``, ``relationship`` (reinforces / updates /
            contradicts / unrelated), ``score``, ``explanation``.
        """
        conflicts: list[dict[str, Any]] = []
        incoming_lower = incoming_text.lower()

        for mem in existing_memories:
            existing_text = mem.get("document", "")
            score = mem.get("score", 0.0)
            existing_meta = mem.get("metadata", {})

            # Skip archived
            if existing_meta.get("status") == "archived":
                continue

            relationship, explanation = self._classify_relationship(
                incoming_lower, existing_text, score, existing_meta,
            )

            conflicts.append({
                "memory_id": mem["id"],
                "relationship": relationship,
                "score": score,
                "explanation": explanation,
            })

        logger.debug("Conflict detection complete | candidates=%d", len(conflicts))
        return conflicts

    async def decide_evolution(
        self,
        memory_text: str,
        existing_evaluation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """End-to-end evolution pipeline: search, analyse, decide.

        Args:
            memory_text:        The incoming message.
            existing_evaluation:  Optional result from MemoryEvaluator
                                  (must contain ``store``, ``category``,
                                   ``importance``, ``reason``).

        Returns:
            dict with keys:
            ``action`` (CREATE / UPDATE / MERGE / SKIP),
            ``target_id`` (str or None),
            ``version`` (int),
            ``explanation`` (str).
        """
        # If evaluator says skip, honour it
        if existing_evaluation and not existing_evaluation.get("store", True):
            return {
                "action": "SKIP",
                "target_id": None,
                "version": 0,
                "explanation": existing_evaluation.get("reason", "Skipped by evaluator."),
            }

        related = await self.detect_related_memories(memory_text, top_k=5)
        conflicts = await self.detect_conflicts(memory_text, related)

        active_conflicts = [c for c in conflicts if c["relationship"] != "unrelated"]

        if not active_conflicts:
            return {
                "action": "CREATE",
                "target_id": None,
                "version": 1,
                "explanation": "No related memories found. Creating a new entry.",
            }

        # Priority 1: contradiction → UPDATE
        for conflict in active_conflicts:
            if conflict["relationship"] == "contradicts":
                target = self._resolve_memory(related, conflict["memory_id"])
                old_ver = target.get("metadata", {}).get("version", 1) if target else 1
                return {
                    "action": "UPDATE",
                    "target_id": conflict["memory_id"],
                    "version": old_ver + 1,
                    "explanation": conflict["explanation"],
                }

        # Priority 2: updates (change indicators) → UPDATE
        for conflict in active_conflicts:
            if conflict["relationship"] == "updates":
                target = self._resolve_memory(related, conflict["memory_id"])
                old_ver = target.get("metadata", {}).get("version", 1) if target else 1
                return {
                    "action": "UPDATE",
                    "target_id": conflict["memory_id"],
                    "version": old_ver + 1,
                    "explanation": conflict["explanation"],
                }

        # Priority 3: reinforces with high similarity (score > 0.7) → MERGE
        for conflict in active_conflicts:
            if conflict["relationship"] == "reinforces" and conflict["score"] > 0.7:
                return {
                    "action": "MERGE",
                    "target_id": conflict["memory_id"],
                    "version": 0,
                    "explanation": conflict["explanation"],
                }

        # Default: create
        return {
            "action": "CREATE",
            "target_id": None,
            "version": 1,
            "explanation": "No strong conflict or update detected. Creating a new memory.",
        }

    async def create_memory(
        self,
        memory_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a brand-new memory with version 1.

        Args:
            memory_text: Text to store.
            metadata:    Optional metadata (version/status/history are set
                         automatically).

        Returns:
            dict with ``memory_id``, ``status``, ``version``, ``created_at``.
        """
        resolved = dict(metadata or {})
        resolved.setdefault("version", 1)
        resolved.setdefault("status", "active")
        resolved.setdefault("history", "[]")
        resolved["updated_at"] = datetime.now(tz=timezone.utc).isoformat()

        result = await self._memory_service.save_memory(memory_text, metadata=resolved)
        result["version"] = 1
        return result

    async def update_memory(
        self,
        memory_id: str,
        new_text: str,
        new_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update an existing memory, preserving the previous version in history.

        Args:
            memory_id:   UUID of the memory to update.
            new_text:    New text content.
            new_metadata: Optional metadata overrides.

        Returns:
            dict with ``memory_id``, ``version``, ``status``.

        Raises:
            MemoryEvolutionServiceError: If memory not found or update fails.
        """
        existing = self._chroma_service.get_memory_by_id(memory_id)
        if existing is None:
            raise MemoryEvolutionServiceError(
                f"Memory '{memory_id}' not found.", status_code=404,
            )

        old_doc = existing["document"]
        old_meta = dict(existing.get("metadata", {}))
        old_version = old_meta.get("version", 1)
        old_history_raw = old_meta.get("history", "[]")

        # Parse existing history
        old_history: list[dict[str, Any]] = []
        if isinstance(old_history_raw, str):
            try:
                old_history = json.loads(old_history_raw)
            except (json.JSONDecodeError, TypeError):
                old_history = []
        elif isinstance(old_history_raw, list):
            old_history = old_history_raw

        # Prepend current version as a history entry
        clean_old_meta = {k: v for k, v in old_meta.items() if k != "history"}
        old_history.insert(0, {
            "version": old_version,
            "text": old_doc,
            "metadata": clean_old_meta,
            "archived_at": datetime.now(tz=timezone.utc).isoformat(),
        })

        # Build new metadata
        new_meta = dict(new_metadata or {})
        new_meta["version"] = old_version + 1
        new_meta["status"] = "active"
        new_meta["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
        new_meta["history"] = json.dumps(old_history)

        # Preserve fields from old metadata unless explicitly overridden
        for key in ("created_at", "source", "category", "importance", "tags"):
            if key in old_meta and key not in new_meta:
                new_meta.setdefault(key, old_meta[key])

        # Generate new embedding
        try:
            embedding = await self._embedding_service.embed_text(new_text)
        except EmbeddingServiceError as exc:
            raise MemoryEvolutionServiceError(
                f"Failed to generate embedding: {exc}",
                status_code=exc.status_code,
            ) from exc

        # Persist
        try:
            self._chroma_service.update_memory(
                memory_id=memory_id,
                embedding=embedding,
                document=new_text,
                metadata=new_meta,
            )
        except ChromaServiceError as exc:
            raise MemoryEvolutionServiceError(
                str(exc), status_code=exc.status_code,
            ) from exc

        logger.info(
            "Memory updated | id=%s | old_version=%d | new_version=%d | history_entries=%d",
            memory_id, old_version, old_version + 1, len(old_history),
        )

        return {"memory_id": memory_id, "version": old_version + 1, "status": "updated"}

    async def merge_memory(
        self,
        target_id: str,
        source_id: str,
        merged_text: str,
        merged_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Merge two memories: archive the source, update the target.

        The target receives the *merged_text* and inherits metadata from
        both records, with the target's values taking precedence.

        Args:
            target_id:       UUID of the surviving memory.
            source_id:       UUID of the memory to archive.
            merged_text:     Combined text for the target.
            merged_metadata: Optional metadata overrides.

        Returns:
            dict with ``target_id``, ``source_id``, ``version``, ``status``.
        """
        target = self._chroma_service.get_memory_by_id(target_id)
        if target is None:
            raise MemoryEvolutionServiceError(
                f"Target memory '{target_id}' not found.", status_code=404,
            )

        source = self._chroma_service.get_memory_by_id(source_id)
        if source is None:
            raise MemoryEvolutionServiceError(
                f"Source memory '{source_id}' not found.", status_code=404,
            )

        # Archive the source first
        await self.archive_memory(source_id)

        # Build merged metadata
        target_meta = dict(target.get("metadata", {}))
        source_meta = dict(source.get("metadata", {}))

        meta = dict(merged_metadata or {})
        for key in ("source", "category", "tags"):
            if key not in meta:
                meta[key] = target_meta.get(key, source_meta.get(key, ""))
        if "importance" not in meta:
            meta["importance"] = max(
                target_meta.get("importance", 0.5),
                source_meta.get("importance", 0.5),
            )

        result = await self.update_memory(
            memory_id=target_id,
            new_text=merged_text,
            new_metadata=meta,
        )

        logger.info(
            "Memory merge complete | target=%s v%d | source=%s archived",
            target_id, result["version"], source_id,
        )

        return {
            "target_id": target_id,
            "source_id": source_id,
            "version": result["version"],
            "status": "merged",
        }

    async def archive_memory(self, memory_id: str) -> dict[str, str]:
        """Mark a memory as archived (excluded from normal retrieval).

        The original text and metadata are preserved — only the ``status``
        field changes.  The embedding is re-generated so the vector store
        stays consistent.

        Args:
            memory_id: UUID of the memory to archive.

        Returns:
            dict with ``memory_id`` and ``status``.
        """
        existing = self._chroma_service.get_memory_by_id(memory_id)
        if existing is None:
            raise MemoryEvolutionServiceError(
                f"Memory '{memory_id}' not found.", status_code=404,
            )

        old_meta = dict(existing.get("metadata", {}))
        old_meta["status"] = "archived"
        old_meta["updated_at"] = datetime.now(tz=timezone.utc).isoformat()

        try:
            embedding = await self._embedding_service.embed_text(existing["document"])
        except EmbeddingServiceError as exc:
            raise MemoryEvolutionServiceError(
                f"Failed to generate embedding: {exc}",
                status_code=exc.status_code,
            ) from exc

        try:
            self._chroma_service.update_memory(
                memory_id=memory_id,
                embedding=embedding,
                document=existing["document"],
                metadata=old_meta,
            )
        except ChromaServiceError as exc:
            raise MemoryEvolutionServiceError(
                str(exc), status_code=exc.status_code,
            ) from exc

        logger.info("Memory archived | id=%s", memory_id)
        return {"memory_id": memory_id, "status": "archived"}

    async def get_history(self, memory_id: str) -> dict[str, Any]:
        """Return the full version history for a memory.

        Args:
            memory_id: UUID of the memory.

        Returns:
            dict with ``memory_id``, ``current_version``, ``current_text``,
            ``current_metadata``, ``history`` (list of previous versions).
        """
        existing = self._chroma_service.get_memory_by_id(memory_id)
        if existing is None:
            raise MemoryEvolutionServiceError(
                f"Memory '{memory_id}' not found.", status_code=404,
            )

        meta = dict(existing.get("metadata", {}))
        history_raw = meta.get("history", "[]")

        history: list[dict[str, Any]] = []
        if isinstance(history_raw, str):
            try:
                history = json.loads(history_raw)
            except (json.JSONDecodeError, TypeError):
                history = []
        elif isinstance(history_raw, list):
            history = history_raw

        current_meta = {k: v for k, v in meta.items() if k != "history"}

        return {
            "memory_id": memory_id,
            "current_version": meta.get("version", 1),
            "current_text": existing["document"],
            "current_metadata": current_meta,
            "history": history,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_relationship(
        self,
        incoming_lower: str,
        existing_text: str,
        score: float,
        existing_meta: dict[str, Any],
    ) -> tuple[str, str]:
        """Classify the relationship between incoming and existing text.

        Returns a tuple of (relationship_label, explanation_string).
        """
        existing_lower = existing_text.lower()

        if score < 0.25:
            return ("unrelated", f"Low similarity (score={score:.2f}).")

        change_keywords = [
            "now prefer", "instead of", "used to", "formerly",
            "changed my", "switched to", "no longer", "not anymore",
            "completed", "finished", "achieved", "accomplished",
            "i now", "i currently", "these days", "anymore",
        ]
        has_change = any(kw in incoming_lower for kw in change_keywords)

        common_tokens = set(incoming_lower.split()) & set(existing_lower.split())
        total = max(len(set(existing_lower.split())), 1)
        overlap_ratio = len(common_tokens) / total

        # Contradiction via known replacement pairs
        contradiction_pairs = [
            ("java", "rust"), ("python", "java"), ("javascript", "typescript"),
            ("like", "dislike"), ("love", "hate"), ("good", "bad"),
            ("windows", "linux"), ("mac", "windows"), ("mac", "linux"),
            ("frontend", "backend"), ("react", "vue"), ("angular", "react"),
        ]
        for a, b in contradiction_pairs:
            if (a in existing_lower and b in incoming_lower) or \
               (b in existing_lower and a in incoming_lower):
                return (
                    "contradicts",
                    f"New information contradicts existing memory ({a} vs {b}).",
                )

        # Change keywords + some overlap → UPDATE
        if has_change and overlap_ratio > 0.1:
            return (
                "updates",
                f"New information replaces or updates existing content (overlap={overlap_ratio:.0%}).",
            )

        # High similarity + significant overlap → reinforces (MERGE candidate)
        if score > 0.7 and overlap_ratio > 0.25:
            return (
                "reinforces",
                f"New information complements existing memory (score={score:.2f}, overlap={overlap_ratio:.0%}).",
            )

        return ("unrelated", f"No actionable relationship (score={score:.2f}).")

    @staticmethod
    def _resolve_memory(
        memories: list[dict[str, Any]],
        memory_id: str,
    ) -> dict[str, Any] | None:
        """Look up a memory dict by ID from a list."""
        for mem in memories:
            if mem["id"] == memory_id:
                return mem
        return None
