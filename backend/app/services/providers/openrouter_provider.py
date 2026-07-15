from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from ..exceptions import (
    LLMQuotaExceeded,
    LLMRateLimited,
    LLMServiceError,
    ProviderBadRequest,
    ProviderConnectionError,
    ProviderMalformedResponse,
    ProviderServerError,
    ProviderTimeout,
    ProviderUnauthorized,
    ProviderUnavailable,
)
from .provider_interface import LLMProvider

_RETRYABLE_STATUSES: set[int] = {429, 500, 502, 503, 504}
_MAX_RETRIES: int = 2
_RETRY_BASE_DELAY: float = 1.0


class OpenRouterProvider(LLMProvider):

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._logger = logging.getLogger(f"{__name__}.openrouter")

    @property
    def provider_name(self) -> str:
        return "openrouter"

    @property
    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def health_check(self) -> bool:
        if not self.is_available():
            return False
        try:
            client = self._get_client()
            response = await client.get("/models", timeout=httpx.Timeout(5.0))
            return response.is_success
        except Exception:
            return False

    async def generate(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        client = self._get_client()
        started_at = time.perf_counter()

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            if attempt > 0:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                self._logger.warning(
                    "OpenRouter retry %d/%d | waiting %.1fs",
                    attempt, _MAX_RETRIES, delay,
                )
                await self._sleep(delay)

            try:
                response = await client.post(
                    "/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                status_code = exc.response.status_code
                detail = self._extract_error(exc.response)

                if status_code not in _RETRYABLE_STATUSES:
                    raise self._classify(status_code, detail) from exc

                if attempt < _MAX_RETRIES:
                    continue
                raise self._classify(status_code, detail) from exc
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    continue
                raise ProviderTimeout(
                    "The AI service took too long to respond. Please try again."
                ) from exc
            except httpx.RequestError as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    continue
                raise ProviderConnectionError(
                    f"Unable to reach OpenRouter: {exc}"
                ) from exc

        if last_exc:
            raise ProviderServerError(
                f"OpenRouter request failed after {_MAX_RETRIES + 1} attempts."
            ) from last_exc

        duration_ms = (time.perf_counter() - started_at) * 1000

        payload_response = response.json()
        try:
            content = payload_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderMalformedResponse(
                "OpenRouter returned an unexpected payload."
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise ProviderMalformedResponse("OpenRouter returned an empty response.")

        self._logger.info(
            "OpenRouter success | model=%s | duration_ms=%.2f",
            self._model, duration_ms,
        )

        return content.strip()

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        return self._client

    async def _sleep(self, delay: float) -> None:
        import asyncio
        await asyncio.sleep(delay)

    @staticmethod
    def _extract_error(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text[:200] or f"HTTP {response.status_code}"
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                msg = error.get("message")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
            msg = payload.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
        return f"HTTP {response.status_code}"

    @staticmethod
    def _classify(status_code: int, detail: str) -> LLMServiceError:
        mapping: dict[int, type[LLMServiceError]] = {
            400: ProviderBadRequest,
            401: ProviderUnauthorized,
            402: LLMQuotaExceeded,
            403: ProviderUnauthorized,
            404: ProviderBadRequest,
            408: ProviderTimeout,
            429: LLMRateLimited,
            500: ProviderServerError,
            502: ProviderServerError,
            503: ProviderUnavailable,
            504: ProviderTimeout,
        }
        cls = mapping.get(status_code, LLMServiceError)
        return cls(detail)
