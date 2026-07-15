"""In-memory cache for memory search results with TTL and auto-invalidation."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS: float = 30.0


class MemorySearchCache:
    """Cache recent memory search results to avoid redundant ChromaDB queries.

    Automatically invalidated on any memory write (create, update, delete).
    """

    def __init__(self, ttl: float = _CACHE_TTL_SECONDS) -> None:
        self._ttl = ttl
        self._cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._valid = True
        self._hits = 0
        self._misses = 0
        self._invalidations = 0

    def get(self, query: str) -> list[dict[str, Any]] | None:
        now = time.monotonic()
        key = query.strip().lower()
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        ts, results = entry
        if now - ts > self._ttl:
            del self._cache[key]
            self._misses += 1
            return None
        if not self._valid:
            self._misses += 1
            return None
        self._hits += 1
        logger.debug("Memory cache HIT | query=%.60r | results=%d", query, len(results))
        return results

    def set(self, query: str, results: list[dict[str, Any]]) -> None:
        key = query.strip().lower()
        self._cache[key] = (time.monotonic(), results)
        logger.debug("Memory cache SET | query=%.60r | results=%d", query, len(results))

    def invalidate(self) -> None:
        self._cache.clear()
        self._valid = True
        self._invalidations += 1
        logger.debug("Memory cache invalidated")

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "invalidations": self._invalidations,
            "hit_rate": round(self.hit_rate, 4),
            "size": len(self._cache),
        }
