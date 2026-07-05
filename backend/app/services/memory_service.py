"""Memory service — the single orchestration point for Phase 3.

Coordinates the EmbeddingService and ChromaService to store, retrieve,
list, and delete semantic memories.  No LLM calls are made here.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from .chroma_service import ChromaService, ChromaServiceError
from .embedding_service import EmbeddingService, EmbeddingServiceError

logger = logging.getLogger(__name__)


class MemoryServiceError(RuntimeError):
    """Raised when a memory operation cannot be completed."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


# Sentinel so callers can distinguish "not provided" from an explicit None
_MISSING = object()


class MemoryService:
    """Orchestrate semantic memory storage and retrieval.

    All public methods are independent and reusable — none calls another
    public method internally, making them safe to invoke from any context.

    Args:
        embedding_service: Provides dense vector generation.
        chroma_service:    Provides persistent vector storage.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        chroma_service: ChromaService,
    ) -> None:
        self._embeddings = embedding_service
        self._chroma = chroma_service

    # ------------------------------------------------------------------
    # save_memory
    # ------------------------------------------------------------------

    async def save_memory(
        self,
        memory_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Embed *memory_text* and persist it in ChromaDB.

        A UUID is generated automatically so every memory is globally unique.
        Sensible defaults are merged into *metadata* for any keys not supplied
        by the caller.

        Args:
            memory_text: The raw text to store as a memory.
            metadata:    Optional key/value metadata.  Recognised keys:
                         ``source``, ``tags``, ``importance``.
                         ``created_at`` is always set automatically.

        Returns:
            dict with ``memory_id``, ``status``, ``created_at``.

        Raises:
            MemoryServiceError: On embedding or storage failure.
        """
        if not memory_text or not memory_text.strip():
            raise MemoryServiceError("memory_text must not be empty.", status_code=422)

        memory_id = str(uuid.uuid4())
        created_at = datetime.now(tz=timezone.utc).isoformat()

        # Build metadata — merge caller-supplied values over defaults
        resolved_meta = _build_metadata(metadata, created_at)

        logger.info("Saving memory | id=%s | text_length=%d", memory_id, len(memory_text))

        # 1. Generate embedding
        try:
            embedding = await self._embeddings.embed_text(memory_text)
            logger.info(
                "Embedding generated for memory | id=%s | dim=%d | text=%s",
                memory_id,
                len(embedding),
                memory_text,
            )
            logger.info("Memory metadata | id=%s | metadata=%s", memory_id, resolved_meta)
        except EmbeddingServiceError as exc:
            logger.exception("Embedding failed during save_memory | id=%s", memory_id)
            raise MemoryServiceError(
                f"Failed to generate embedding: {exc}", status_code=exc.status_code
            ) from exc

        # 2. Persist in ChromaDB
        try:
            self._chroma.add_memory(
                memory_id=memory_id,
                embedding=embedding,
                document=memory_text,
                metadata=resolved_meta,
            )
        except ChromaServiceError as exc:
            logger.exception("ChromaDB write failed during save_memory | id=%s", memory_id)
            raise MemoryServiceError(str(exc), status_code=exc.status_code) from exc

        logger.info("Memory saved successfully | id=%s | text=%.80r", memory_id, memory_text)
        return {
            "memory_id": memory_id,
            "status": "saved",
            "created_at": created_at,
        }

    # ------------------------------------------------------------------
    # search_memory
    # ------------------------------------------------------------------

    async def search_memory(
        self,
        query: str,
        top_k: int = 5,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        """Semantically search stored memories for *query*.

        No LLM is invoked.  Results are ranked by cosine similarity.
        Archived memories are excluded by default.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of results to return (1–100).
            include_archived: If *True*, archived memories are also returned.

        Returns:
            List of dicts ordered by descending similarity, each with:
            ``id``, ``document``, ``score``, ``metadata``.

        Raises:
            MemoryServiceError: On embedding or retrieval failure.
        """
        if not query or not query.strip():
            raise MemoryServiceError("query must not be empty.", status_code=422)

        if top_k < 1:
            top_k = 1
        elif top_k > 100:
            top_k = 100

        logger.info("Searching memories | query_length=%d | top_k=%d", len(query), top_k)
        logger.info("Search query | %s", query)

        # 1. Embed the query
        try:
            query_embedding = await self._embeddings.embed_text(query)
        except EmbeddingServiceError as exc:
            logger.exception("Embedding failed during search_memory")
            raise MemoryServiceError(
                f"Failed to generate query embedding: {exc}", status_code=exc.status_code
            ) from exc

        # 2. Vector search in ChromaDB
        try:
            results = self._chroma.search_memory(
                query_embedding=query_embedding,
                top_k=top_k,
            )
        except ChromaServiceError as exc:
            logger.exception("ChromaDB query failed during search_memory")
            raise MemoryServiceError(str(exc), status_code=exc.status_code) from exc

        # 3. Filter out archived unless explicitly requested
        if not include_archived:
            results = _filter_active(results)

        logger.info("Memory search returned %d result(s) | query=%.60r", len(results), query)
        for i, r in enumerate(results):
            logger.info(
                "  result[%d] | score=%.4f | text=%.80r",
                i, r.get("score", 0), r.get("document", ""),
            )
        return results

    # ------------------------------------------------------------------
    # delete_memory
    # ------------------------------------------------------------------

    def delete_memory(self, memory_id: str) -> dict[str, str]:
        """Delete the memory identified by *memory_id*.

        Args:
            memory_id: UUID of the memory to delete.

        Returns:
            dict with ``memory_id`` and ``status``.

        Raises:
            MemoryServiceError: If the memory is not found or deletion fails.
        """
        if not memory_id or not memory_id.strip():
            raise MemoryServiceError("memory_id must not be empty.", status_code=422)

        logger.info("Deleting memory | id=%s", memory_id)

        try:
            self._chroma.delete_memory(memory_id)
        except ChromaServiceError as exc:
            logger.exception("ChromaDB delete failed | id=%s", memory_id)
            raise MemoryServiceError(str(exc), status_code=exc.status_code) from exc

        logger.info("Memory deleted | id=%s", memory_id)
        return {"memory_id": memory_id, "status": "deleted"}

    # ------------------------------------------------------------------
    # list_memories
    # ------------------------------------------------------------------

    def list_memories(
        self,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        """Return stored memory records.

        Archived memories are excluded by default.

        Args:
            include_archived: If *True*, archived memories are also returned.

        Returns:
            List of dicts, each with ``id``, ``document``, ``metadata``.

        Raises:
            MemoryServiceError: If the retrieval fails.
        """
        logger.info("Listing all memories")

        try:
            memories = self._chroma.list_all_memories()
        except ChromaServiceError as exc:
            logger.exception("ChromaDB list failed")
            raise MemoryServiceError(str(exc), status_code=exc.status_code) from exc

        if not include_archived:
            memories = _filter_active(memories)

        logger.info("Listed %d memory record(s)", len(memories))
        return memories

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_metadata(
    caller_meta: dict[str, Any] | None,
    created_at: str,
) -> dict[str, Any]:
    """Merge caller-supplied metadata over sensible defaults.

    ChromaDB only accepts str / int / float / bool values in metadata.
    Lists are serialised to comma-separated strings automatically.
    """
    defaults: dict[str, Any] = {
        "created_at": created_at,
        "source": "user",
        "tags": "",
        "importance": 0.5,
        "memory_strength": 0.60,
        "status": "active",
        "version": 1,
    }


def _filter_active(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only records whose status is not ``archived``.

    Records without a ``status`` metadata field are treated as active
    for backward compatibility.
    """
    filtered: list[dict[str, Any]] = []
    for r in records:
        if r is None:
            continue
        meta = r.get("metadata")
        if meta is None:
            r = dict(r)
            r["metadata"] = {}
            filtered.append(r)
        elif meta.get("status") != "archived":
            filtered.append(r)
    return filtered

    if caller_meta:
        for key, value in caller_meta.items():
            if isinstance(value, list):
                # ChromaDB does not support list values — flatten to CSV string
                defaults[key] = ", ".join(str(v) for v in value)
            elif isinstance(value, (str, int, float, bool)):
                defaults[key] = value
            else:
                # Coerce anything else to string
                defaults[key] = str(value)

    # created_at is always overwritten with our server-side timestamp
    defaults["created_at"] = created_at
    return defaults
