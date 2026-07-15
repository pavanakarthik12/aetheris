from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

_METRICS_WINDOW_SIZE: int = 100


class ProviderHealthMonitor:
    def __init__(self, provider: str, model: str) -> None:
        self._provider = provider
        self._model = model
        self._success_count: int = 0
        self._failure_count: int = 0
        self._count_429: int = 0
        self._count_402: int = 0
        self._count_timeout: int = 0
        self._count_retry: int = 0
        self._latencies: deque[float] = deque(maxlen=_METRICS_WINDOW_SIZE)
        self._failures: list[dict[str, Any]] = []
        self._last_success_time: float = 0.0
        self._last_failure_time: float = 0.0
        self._logger = logging.getLogger(f"{__name__}.{provider}")

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    def record_success(self, latency_ms: float) -> None:
        self._success_count += 1
        self._latencies.append(latency_ms)
        self._last_success_time = time.time()

    def record_failure(
        self,
        error_code: str,
        http_status: int | None = None,
        retry_count: int = 0,
    ) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()
        entry: dict[str, Any] = {
            "error_code": error_code,
            "http_status": http_status,
            "retry_count": retry_count,
            "timestamp": self._last_failure_time,
        }
        self._failures.append(entry)

        if http_status == 429:
            self._count_429 += 1
        elif http_status == 402:
            self._count_402 += 1

        if error_code == "timeout":
            self._count_timeout += 1

    def record_retry(self) -> None:
        self._count_retry += 1

    def snapshot(self) -> dict[str, Any]:
        total = self._success_count + self._failure_count
        if total == 0:
            status = "unknown"
            healthy = False
            failure_rate = 0.0
        else:
            failure_rate = self._failure_count / total * 100
            healthy = failure_rate < 50.0
            status = "healthy" if healthy else "degraded"
        avg_latency = (
            sum(self._latencies) / len(self._latencies)
            if self._latencies
            else 0.0
        )
        return {
            "provider": self._provider,
            "model": self._model,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "total_requests": total,
            "failure_rate_percent": round(failure_rate, 2),
            "average_latency_ms": round(avg_latency, 2),
            "count_429": self._count_429,
            "count_402": self._count_402,
            "count_timeout": self._count_timeout,
            "count_retry": self._count_retry,
            "last_success_time": self._last_success_time,
            "last_failure_time": self._last_failure_time,
            "healthy": healthy,
            "current_status": status,
        }
