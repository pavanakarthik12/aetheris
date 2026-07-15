from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any


class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    _COOLDOWN_SECONDS: float = 30.0
    _FAILURE_THRESHOLD: int = 3

    def __init__(self, provider: str, model: str) -> None:
        self._provider = provider
        self._model = model
        self._state = CircuitState.CLOSED
        self._failure_count: int = 0
        self._last_failure_time: float = 0.0
        self._last_test_time: float = 0.0
        self._logger = logging.getLogger(f"{__name__}.{provider}")

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self._FAILURE_THRESHOLD and self._state == CircuitState.CLOSED:
            self._state = CircuitState.OPEN
            self._logger.warning(
                "Circuit opened | provider=%s | model=%s | failures=%d | cooldown=%.0fs",
                self._provider, self._model, self._failure_count, self._COOLDOWN_SECONDS,
            )

    def record_success(self) -> None:
        self._failure_count = 0
        if self._state != CircuitState.CLOSED:
            old = self._state.value
            self._state = CircuitState.CLOSED
            self._logger.info(
                "Circuit closed | provider=%s | model=%s | was=%s",
                self._provider, self._model, old,
            )

    def allow_request(self) -> bool:
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._COOLDOWN_SECONDS:
                self._state = CircuitState.HALF_OPEN
                self._logger.info(
                    "Circuit half-opened | provider=%s | model=%s | cooldown_elapsed=%.0fs",
                    self._provider, self._model, elapsed,
                )
                return True
            return False
        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "provider": self._provider,
            "model": self._model,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "is_open": self.is_open,
            "cooldown_seconds": self._COOLDOWN_SECONDS,
        }
