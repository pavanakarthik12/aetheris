"""Lightweight in-memory metrics collector for the request pipeline."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

_WINDOW_SIZE = 1000


class MetricsCollector:
    """Collect and expose request-level performance metrics."""

    def __init__(self) -> None:
        self._durations: deque[float] = deque(maxlen=_WINDOW_SIZE)
        self._retrieval_times: deque[float] = deque(maxlen=_WINDOW_SIZE)
        self._llm_times: deque[float] = deque(maxlen=_WINDOW_SIZE)
        self._prompt_sizes: deque[int] = deque(maxlen=_WINDOW_SIZE)
        self._completion_sizes: deque[int] = deque(maxlen=_WINDOW_SIZE)
        self._dedup_count: int = 0
        self._cache_hits: int = 0
        self._retry_count: int = 0
        self._total_requests: int = 0

    def record_request(
        self,
        duration_ms: float,
        retrieval_ms: float = 0.0,
        llm_ms: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        dedup_saved: int = 0,
        cache_hit: bool = False,
        retries: int = 0,
    ) -> None:
        self._total_requests += 1
        self._durations.append(duration_ms)
        self._retrieval_times.append(retrieval_ms)
        self._llm_times.append(llm_ms)
        self._prompt_sizes.append(prompt_tokens)
        self._completion_sizes.append(completion_tokens)
        self._dedup_count += dedup_saved
        self._retry_count += retries
        if cache_hit:
            self._cache_hits += 1

    def _avg(self, values: deque[float]) -> float:
        return round(sum(values) / len(values), 2) if values else 0.0

    def snapshot(self) -> dict[str, Any]:
        total = self._total_requests
        return {
            "total_requests": total,
            "average_latency_ms": self._avg(self._durations),
            "average_retrieval_ms": self._avg(self._retrieval_times),
            "average_llm_ms": self._avg(self._llm_times),
            "average_prompt_tokens": round(self._avg(self._prompt_sizes)),
            "average_completion_tokens": round(self._avg(self._completion_sizes)),
            "duplicates_prevented": self._dedup_count,
            "retry_count": self._retry_count,
            "cache_hits": self._cache_hits,
            "window_size": len(self._durations),
        }
