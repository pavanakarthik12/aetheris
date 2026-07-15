from __future__ import annotations

import logging
import time
from typing import Any

from ..circuit_breaker import CircuitBreaker
from ..exceptions import (
    LLMQuotaExceeded,
    LLMRateLimited,
    LLMServiceError,
    ProviderConnectionError,
    ProviderServerError,
    ProviderTimeout,
    ProviderUnavailable,
)
from ..provider_health import ProviderHealthMonitor
from .provider_interface import LLMProvider

_RETRYABLE_FOR_FAILOVER: set[type[LLMServiceError]] = {
    LLMRateLimited,
    LLMQuotaExceeded,
    ProviderServerError,
    ProviderUnavailable,
    ProviderTimeout,
    ProviderConnectionError,
}

logger = logging.getLogger(__name__)


class ManagedProvider:

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self.circuit_breaker = CircuitBreaker(
            provider=provider.provider_name,
            model=provider.model_name,
        )
        self.health = ProviderHealthMonitor(
            provider=provider.provider_name,
            model=provider.model_name,
        )

    @property
    def name(self) -> str:
        return self.provider.provider_name

    def snapshot(self) -> dict[str, Any]:
        cb = self.circuit_breaker.snapshot()
        h = self.health.snapshot()
        return {
            "provider": self.name,
            "model": self.provider.model_name,
            "healthy": h["healthy"],
            "current_status": h["current_status"],
            "circuit_open": cb["is_open"],
            "circuit_state": cb["state"],
            "circuit_failure_count": cb["failure_count"],
            "average_latency_ms": h["average_latency_ms"],
            "failure_rate_percent": h["failure_rate_percent"],
            "success_count": h["success_count"],
            "failure_count": h["failure_count"],
            "total_requests": h["total_requests"],
            "retry_count": h["count_retry"],
            "count_429": h["count_429"],
            "count_402": h["count_402"],
            "count_timeout": h["count_timeout"],
            "last_success_time": h["last_success_time"],
            "last_failure_time": h["last_failure_time"],
            "available": self.provider.is_available(),
        }


class ProviderManager:

    def __init__(
        self,
        providers: list[LLMProvider],
        enable_failover: bool = True,
        enable_circuit_breaker: bool = True,
    ) -> None:
        self._enable_failover = enable_failover
        self._enable_circuit_breaker = enable_circuit_breaker
        self._managed: list[ManagedProvider] = [
            ManagedProvider(p) for p in providers
        ]
        self._primary_index: int = 0
        self._active_index: int = 0
        self._fallback_used: bool = False
        self._last_switch_time: float = 0.0
        self._logger = logging.getLogger(__name__)

    @property
    def current_provider(self) -> ManagedProvider:
        return self._managed[self._active_index]

    @property
    def primary_provider(self) -> ManagedProvider:
        return self._managed[self._primary_index]

    @property
    def fallback_used(self) -> bool:
        return self._fallback_used

    @property
    def active_provider_name(self) -> str:
        return self.current_provider.name

    @property
    def active_model_name(self) -> str:
        return self.current_provider.provider.model_name

    def _is_retryable_for_failover(self, exc: LLMServiceError) -> bool:
        return any(isinstance(exc, t) for t in _RETRYABLE_FOR_FAILOVER)

    def _switch_to_fallback(self) -> None:
        if self._active_index == self._primary_index and len(self._managed) > 1:
            self._active_index = 1
            self._fallback_used = True
            self._last_switch_time = time.monotonic()
            self._logger.warning(
                "Failover triggered | primary=%s | fallback=%s",
                self._managed[self._primary_index].name,
                self.current_provider.name,
            )

    def _switch_to_primary(self) -> None:
        if self._active_index != self._primary_index:
            self._active_index = self._primary_index
            self._fallback_used = False
            self._logger.info(
                "Restored primary | provider=%s",
                self.primary_provider.name,
            )

    def _try_restore_primary(self) -> None:
        if self._active_index != self._primary_index:
            elapsed = time.monotonic() - self._last_switch_time
            if elapsed >= 30.0:
                self._switch_to_primary()

    async def generate(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        if self._enable_failover:
            self._try_restore_primary()

        for attempt_index in range(len(self._managed)):
            mp = self._managed[self._active_index]

            if self._enable_circuit_breaker and not mp.circuit_breaker.allow_request():
                mp.health.record_failure(error_code="circuit_open", http_status=None)
                self._logger.warning(
                    "Circuit open for %s, skipping", mp.name,
                )
                if self._enable_failover and self._active_index == self._primary_index:
                    self._switch_to_fallback()
                    continue
                raise ProviderUnavailable(
                    f"{mp.name} is temporarily unavailable due to repeated failures. "
                    "Please try again in a moment."
                )

            started_at = time.perf_counter()
            try:
                result = await mp.provider.generate(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                duration_ms = (time.perf_counter() - started_at) * 1000
                mp.health.record_success(duration_ms)
                mp.circuit_breaker.record_success()

                if self._enable_failover and self._active_index != self._primary_index:
                    self._logger.info(
                        "Fallback provider succeeded | provider=%s | latency=%.2fms",
                        mp.name, duration_ms,
                    )

                return result

            except LLMServiceError as exc:
                duration_ms = (time.perf_counter() - started_at) * 1000
                mp.health.record_failure(
                    error_code=type(exc).__name__,
                    http_status=getattr(exc, "status_code", None),
                )
                mp.circuit_breaker.record_failure()

                self._logger.warning(
                    "Provider %s failed | error=%s | status_code=%s | duration_ms=%.2f | retryable=%s | active_index=%d | total_providers=%d",
                    mp.name, type(exc).__name__,
                    getattr(exc, "status_code", "N/A"),
                    duration_ms,
                    self._is_retryable_for_failover(exc),
                    self._active_index, len(self._managed),
                )

                if self._enable_failover and self._is_retryable_for_failover(exc):
                    if self._active_index == self._primary_index:
                        self._switch_to_fallback()
                        continue
                    if attempt_index < len(self._managed) - 1:
                        continue

                if self._active_index != self._primary_index:
                    self._switch_to_primary()

                raise

        raise ProviderUnavailable(
            "All AI providers are currently unavailable. "
            "Please try again later."
        )

    def all_snapshots(self) -> list[dict[str, Any]]:
        return [mp.snapshot() for mp in self._managed]
