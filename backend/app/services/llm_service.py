"""Isolated LLM service boundary for Qwen communication."""

from __future__ import annotations

import logging
import time
from urllib.parse import urlparse

import httpx

from ..config.settings import Settings, get_settings


class LLMServiceError(RuntimeError):
    """Raised when the LLM boundary cannot complete a request."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMService:
    """Encapsulate provider selection, model identity, and Qwen access."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._logger = logging.getLogger(__name__)
        self._client: httpx.AsyncClient | None = None

    @property
    def provider(self) -> str:
        """Return the configured LLM provider name."""

        return self._settings.llm_provider

    @property
    def model_name(self) -> str:
        """Return the configured LLM model name."""

        return self._settings.llm_model

    @property
    def api_key(self) -> str:
        """Return the configured provider API key."""

        return self._settings.qwen_api_key

    @property
    def base_url(self) -> str:
        """Return the configured provider base URL."""

        return self._settings.qwen_base_url.rstrip("/")

    @property
    def _uses_openrouter(self) -> bool:
        hostname = urlparse(self.base_url).hostname or ""
        return "openrouter.ai" in hostname

    @property
    def _uses_dashscope(self) -> bool:
        hostname = urlparse(self.base_url).hostname or ""
        return "dashscope.aliyuncs.com" in hostname

    def _ensure_ready(self) -> None:
        if self.provider.lower() != "qwen":
            raise LLMServiceError("Unsupported LLM provider configured.", status_code=500)

        if not self.api_key:
            raise LLMServiceError("QWEN_API_KEY is not configured.", status_code=500)

        if self._uses_dashscope and self.api_key.startswith("sk-or-v1-"):
            raise LLMServiceError(
                "QWEN_API_KEY appears to be an OpenRouter key, but QWEN_BASE_URL points to DashScope. "
                "Use a DashScope API key or set QWEN_BASE_URL=https://openrouter.ai/api/v1.",
                status_code=500,
            )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(30.0, connect=10.0),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )

        return self._client

    async def aclose(self) -> None:
        """Close the reusable HTTP client."""

        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def generate_text(self, prompt: str) -> str:
        """Send a single-message request to Qwen and return the assistant reply."""

        self._ensure_ready()
        client = self._get_client()
        started_at = time.perf_counter()

        self._logger.info(
            "LLM request | provider=%s | model=%s | prompt_length=%s",
            self.provider,
            self.model_name,
            len(prompt),
        )

        try:
            response = await client.post(
                "/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 512,
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            detail = self._extract_error_message(exc.response)
            if status_code == 401:
                raise LLMServiceError(
                    "Authentication failed for the configured LLM endpoint. "
                    "Check QWEN_API_KEY and QWEN_BASE_URL. "
                    f"Provider detail: {detail}",
                    status_code=401,
                ) from exc

            raise LLMServiceError(f"Qwen API returned an error: {detail}") from exc
        except httpx.RequestError as exc:
            raise LLMServiceError("Unable to reach the Qwen API.") from exc

        duration = time.perf_counter() - started_at
        self._logger.info(
            "LLM response time | provider=%s | model=%s | duration_ms=%.2f",
            self.provider,
            self.model_name,
            duration * 1000,
        )

        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMServiceError("Qwen API returned an unexpected payload.") from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMServiceError("Qwen API returned an empty response.")

        self._logger.info("LLM response received | response_length=%s", len(content))
        return content.strip()

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text[:200] or f"HTTP {response.status_code}"

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message.strip():
                    return message.strip()
            message = payload.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()

        return f"HTTP {response.status_code}"