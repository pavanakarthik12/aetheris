"""ChromaDB service — collection management, vector storage, and retrieval."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "aetheris_memory"


class ChromaServiceError(RuntimeError):
    """Raised when a ChromaDB operation cannot be completed."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class ChromaService:
    """Manage the ChromaDB persistent client and the aetheris_memory collection.

    The client and collection are initialised lazily on first access so the
    service is cheap to construct during application startup.

    Args:
        settings: Optional settings override; falls back to the shared
                  application settings when omitted.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._storage_path = Path(self._settings.chroma_db_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._client = None       # chromadb.PersistentClient — loaded lazily
        self._collection = None   # chromadb.Collection     — loaded lazily

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def storage_path(self) -> Path:
        """Return the filesystem location used for ChromaDB persistence."""
        return self._storage_path

    # --- write ---------------------------------------------------------

    def add_memory(
        self,
        memory_id: str,
        embedding: list[float],
        document: str,
        metadata: dict[str, Any],
    ) -> None:
        """Persist a single memory record in the collection.

        Args:
            memory_id: Globally unique identifier for this memory (UUID).
            embedding:  Dense vector produced by the embedding service.
            document:   Raw text of the memory.
            metadata:   Arbitrary key/value pairs (must be str/int/float/bool).

        Raises:
            ChromaServiceError: If the write fails or the ID already exists.
        """
        collection = self._get_collection()
        try:
            collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata],
            )
            stored = collection.get(ids=[memory_id], include=["documents", "metadatas"])
            if not stored.get("ids"):
                raise ChromaServiceError(
                    f"ChromaDB did not confirm insertion for memory '{memory_id}'.",
                    status_code=500,
                )

            logger.info("Memory Saved")
            logger.info("Memory ID | %s", memory_id)
            logger.info("Memory Text | %s", document)
            logger.info("Memory stored | id=%s | doc_length=%d", memory_id, len(document))
        except Exception as exc:
            # ChromaDB raises a plain Exception (or a subclass) for duplicate IDs
            error_msg = str(exc).lower()
            if "already exists" in error_msg or "duplicate" in error_msg:
                raise ChromaServiceError(
                    f"Memory with id '{memory_id}' already exists.",
                    status_code=409,
                ) from exc
            logger.exception("ChromaDB add failed | id=%s", memory_id)
            raise ChromaServiceError(f"Failed to store memory: {exc}") from exc

    # --- read ----------------------------------------------------------

    def search_memory(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the *top_k* most semantically similar memories."""
        collection = self._get_collection()
        logger.info("Query | top_k=%d | dim=%d", top_k, len(query_embedding))

        # Always ask ChromaDB for the live count — never use a cached value.
        # A stale count of 0 would cause an early return even when data exists.
        try:
            live_count = collection.count()
        except Exception:
            live_count = 0

        if live_count == 0:
            logger.info("search_memory | collection is empty, returning []")
            return []

        effective_k = min(top_k, live_count)
        logger.info("search_memory | live_count=%d | effective_k=%d", live_count, effective_k)

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=effective_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.exception("ChromaDB query failed | top_k=%d", top_k)
            raise ChromaServiceError(f"Memory search failed: {exc}") from exc

        records: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for mem_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            score = round(1.0 - dist, 6)
            records.append(
                {
                    "id": mem_id,
                    "document": doc,
                    "score": score,
                    "metadata": meta or {},
                }
            )

        logger.info("Retrieved Memories | count=%d", len(records))
        for record in records:
            logger.info(
                "Similarity Score | id=%s | score=%.6f | text=%s",
                record["id"],
                record["score"],
                record["document"],
            )
        logger.info("Memory search complete | top_k=%d | results=%d", top_k, len(records))
        return records

    def delete_memory(self, memory_id: str) -> None:
        """Remove a single memory record by its UUID.

        Args:
            memory_id: The UUID of the memory to delete.

        Raises:
            ChromaServiceError: If the ID does not exist or deletion fails.
        """
        collection = self._get_collection()
        try:
            # Verify existence before deletion so we can return a 404 cleanly.
            existing = collection.get(ids=[memory_id])
            if not existing["ids"]:
                raise ChromaServiceError(
                    f"Memory '{memory_id}' not found.", status_code=404
                )
            collection.delete(ids=[memory_id])
            logger.info("Memory deleted | id=%s", memory_id)
        except ChromaServiceError:
            raise
        except Exception as exc:
            logger.exception("ChromaDB delete failed | id=%s", memory_id)
            raise ChromaServiceError(f"Failed to delete memory: {exc}") from exc

    def list_all_memories(self) -> list[dict[str, Any]]:
        """Return every stored memory record (useful for debugging).

        Returns:
            List of dicts, each containing ``id``, ``document``, ``metadata``.

        Raises:
            ChromaServiceError: If the retrieval fails.
        """
        collection = self._get_collection()
        try:
            results = collection.get(include=["documents", "metadatas"])
        except Exception as exc:
            logger.exception("ChromaDB list failed")
            raise ChromaServiceError(f"Failed to list memories: {exc}") from exc

        records: list[dict[str, Any]] = []
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        for mem_id, doc, meta in zip(ids, documents, metadatas):
            records.append(
                {
                    "id": mem_id,
                    "document": doc,
                    "metadata": meta or {},
                }
            )

        logger.info("Memory list retrieved | total=%d", len(records))
        return records

    def update_memory(
        self,
        memory_id: str,
        embedding: list[float],
        document: str,
        metadata: dict[str, Any],
    ) -> None:
        """Replace an existing memory record in-place.

        Uses ChromaDB's native ``update`` which requires the ID to already
        exist.  Embeddings, document, and metadata are all replaced atomically.

        Args:
            memory_id: UUID of the memory to update.
            embedding:  New dense vector.
            document:   New text content.
            metadata:   New metadata dict.

        Raises:
            ChromaServiceError: If the ID does not exist or the update fails.
        """
        collection = self._get_collection()
        try:
            existing = collection.get(ids=[memory_id])
            if not existing["ids"]:
                raise ChromaServiceError(
                    f"Memory '{memory_id}' not found.", status_code=404,
                )
            collection.update(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata],
            )
            logger.info("Memory updated | id=%s | doc_length=%d", memory_id, len(document))
        except ChromaServiceError:
            raise
        except Exception as exc:
            logger.exception("ChromaDB update failed | id=%s", memory_id)
            raise ChromaServiceError(f"Failed to update memory: {exc}") from exc

    def get_memory_by_id(self, memory_id: str) -> dict[str, Any] | None:
        """Fetch a single memory by its UUID, or return None if not found.

        Args:
            memory_id: UUID of the memory to fetch.

        Raises:
            ChromaServiceError: If the lookup fails.
        """
        collection = self._get_collection()
        try:
            results = collection.get(ids=[memory_id], include=["documents", "metadatas"])
        except Exception as exc:
            logger.exception("ChromaDB get_by_id failed | id=%s", memory_id)
            raise ChromaServiceError(f"Failed to fetch memory: {exc}") from exc

        if not results["ids"]:
            return None

        return {
            "id": results["ids"][0],
            "document": results["documents"][0],
            "metadata": results["metadatas"][0] or {},
        }

    def get_memories_by_metadata(
        self,
        where: dict[str, Any],
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return memories matching a metadata filter.

        Args:
            where: ChromaDB ``where`` filter dict (e.g. ``{"attribute": "name",
                   "status": "active"}``).
            limit:  Maximum number of results (default: no limit).

        Returns:
            List of dicts, each with ``id``, ``document``, ``metadata``.

        Raises:
            ChromaServiceError: If the lookup fails.
        """
        collection = self._get_collection()
        try:
            kwargs: dict[str, Any] = {
                "where": where,
                "include": ["documents", "metadatas"],
            }
            if limit is not None:
                kwargs["limit"] = limit
            results = collection.get(**kwargs)
        except Exception as exc:
            logger.exception("ChromaDB metadata lookup failed | where=%s", where)
            raise ChromaServiceError(f"Failed to query memories by metadata: {exc}") from exc

        records: list[dict[str, Any]] = []
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        for mem_id, doc, meta in zip(ids, documents, metadatas):
            records.append({
                "id": mem_id,
                "document": doc,
                "metadata": dict(meta) if meta else {},
            })

        logger.info(
            "Metadata lookup | where=%s | results=%d", where, len(records),
        )
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        """Return the persistent ChromaDB client, initialising it on first call."""
        if self._client is None:
            try:
                import chromadb  # type: ignore[import]

                self._client = chromadb.PersistentClient(path=str(self._storage_path))
                logger.info("ChromaDB client initialised | path=%s", self._storage_path)
            except ImportError as exc:
                raise ChromaServiceError(
                    "chromadb is not installed. Run: pip install chromadb"
                ) from exc
            except Exception as exc:
                raise ChromaServiceError(
                    f"Failed to initialise ChromaDB client: {exc}"
                ) from exc
        return self._client

    def _get_collection(self):
        """Return the aetheris_memory collection, creating it if absent."""
        if self._collection is None:
            client = self._get_client()
            try:
                self._collection = client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(
                    "ChromaDB collection ready | name=%s | count=%d",
                    COLLECTION_NAME,
                    self._collection.count(),
                )
            except Exception as exc:
                raise ChromaServiceError(
                    f"Failed to get or create collection '{COLLECTION_NAME}': {exc}"
                ) from exc
        return self._collection

    def _collection_count(self) -> int:
        """Return the live number of records in the collection."""
        try:
            return self._get_collection().count()
        except Exception:
            return 0
