"""Isolated LLM service boundary for Qwen communication."""

from __future__ import annotations

import logging
import time
from typing import Any
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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Close the reusable HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    async def generate_text(self, prompt: str) -> str:
        """Send a single user message to Qwen and return the assistant reply.

        Preserved for backward compatibility.  New callers should prefer
        ``generate_with_context()`` which handles system prompts and memory.

        Args:
            prompt: Raw text sent as a single user message.

        Returns:
            The assistant's reply as a plain string.
        """
        return await self._chat_completion(
            messages=[{"role": "user", "content": prompt}],
        )

    async def generate_with_context(
        self,
        user_message: str,
        system_prompt: str,
        memory_context: str = "",
    ) -> str:
        """Send a structured request to Qwen with system prompt and memory context.

        Builds the OpenAI-compatible messages array in this order:
          1. ``system`` — Aetheris identity and behaviour instructions.
          2. ``user``   — optional memory context block prepended to the
                          user's message in a single turn.

        Combining memory and user text into one user turn avoids mid-conversation
        system messages that some providers handle inconsistently.

        Args:
            user_message:   The raw text from the user.
            system_prompt:  Identity/instruction text for the ``system`` role.
            memory_context: Pre-formatted memory block from ContextBuilderService.
                            Pass ``""`` when no memories are available — the
                            method handles that gracefully.

        Returns:
            The assistant's reply as a plain string.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

        if memory_context.strip():
            user_turn = (
                f"{memory_context.strip()}\n\n"
                f"Current User Message:\n{user_message.strip()}"
            )
        else:
            user_turn = user_message.strip()

        messages.append({"role": "user", "content": user_turn})

        self._logger.info(
            "generate_with_context | memory_present=%s | user_length=%d",
            bool(memory_context.strip()),
            len(user_message),
        )

        return await self._chat_completion(messages=messages)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_ready(self) -> None:
        """Raise LLMServiceError if the service is not properly configured."""
        if self.provider.lower() != "qwen":
            raise LLMServiceError(
                "Unsupported LLM provider configured.", status_code=500
            )
        if not self.api_key:
            raise LLMServiceError(
                "QWEN_API_KEY is not configured.", status_code=500
            )
        if self._uses_dashscope and self.api_key.startswith("sk-or-v1-"):
            raise LLMServiceError(
                "QWEN_API_KEY appears to be an OpenRouter key, but QWEN_BASE_URL "
                "points to DashScope. Use a DashScope API key or set "
                "QWEN_BASE_URL=https://openrouter.ai/api/v1.",
                status_code=500,
            )

    def _get_client(self) -> httpx.AsyncClient:
        """Return the shared async HTTP client, creating it on first call."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(30.0, connect=10.0),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        return self._client

    async def _chat_completion(self, messages: list[dict[str, Any]]) -> str:
        """Send a messages array to the Qwen chat completions endpoint.

        Every public generation method funnels through here so error
        normalisation and timing logs live in exactly one place.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.

        Returns:
            The assistant reply content as a stripped string.

        Raises:
            LLMServiceError: On auth failure, network error, or bad payload.
        """
        self._ensure_ready()
        client = self._get_client()
        started_at = time.perf_counter()

        self._logger.info(
            "LLM request | provider=%s | model=%s | message_count=%d",
            self.provider,
            self.model_name,
            len(messages),
        )

        try:
            response = await client.post(
                "/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": messages,
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
            raise LLMServiceError(
                f"Qwen API returned an error: {detail}"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMServiceError("Unable to reach the Qwen API.") from exc

        duration_ms = (time.perf_counter() - started_at) * 1000
        self._logger.info(
            "LLM response received | provider=%s | model=%s | duration_ms=%.2f",
            self.provider,
            self.model_name,
            duration_ms,
        )

        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMServiceError(
                "Qwen API returned an unexpected payload."
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMServiceError("Qwen API returned an empty response.")

        self._logger.info("LLM reply length=%d", len(content))
        return content.strip()

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        """Pull a human-readable error message out of an error response."""
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
