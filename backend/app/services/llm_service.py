from __future__ import annotations

import logging
from typing import Any

from ..config.settings import Settings, get_settings
from .circuit_breaker import CircuitBreaker
from .conversation_context_filter import filter_conversation
from .prompt_builder import PromptBuilder
from .exceptions import (
    LLMServiceError,
    LLMQuotaExceeded,
    LLMRateLimited,
    ProviderBadRequest,
    ProviderConnectionError,
    ProviderMalformedResponse,
    ProviderServerError,
    ProviderTimeout,
    ProviderUnauthorized,
    ProviderUnavailable,
)
from .provider_health import ProviderHealthMonitor
from .providers.groq_provider import GroqProvider
from .providers.provider_interface import LLMProvider
from .providers.provider_manager import ProviderManager


class LLMService:
    _DEFAULT_MAX_TOKENS: int = 512
    _DEFAULT_TEMPERATURE: float = 0.7

    _MAX_CONVERSATION_TURNS: int = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._logger = logging.getLogger(__name__)
        self._manager: ProviderManager | None = None
        self._conversation_history: list[dict[str, Any]] = []
        self._validate_providers()

    def _validate_providers(self) -> None:
        errors: list[str] = []

        provider = self._settings.llm_provider
        if provider == "groq":
            if not self._settings.groq_api_key:
                errors.append("GROQ_API_KEY is not configured")
            if not self._settings.groq_base_url.startswith("http"):
                errors.append(f"GROQ_BASE_URL is invalid: {self._settings.groq_base_url}")
            if not self._settings.groq_model:
                errors.append("GROQ_MODEL is not configured")

        if errors:
            for err in errors:
                self._logger.warning("Provider config warning: %s", err)

        self._logger.info(
            "Provider loaded | provider=%s | model=%s | circuit_breaker=%s",
            self._settings.llm_provider,
            self._settings.groq_model,
            self._settings.enable_circuit_breaker,
        )

    def _build_manager(self) -> ProviderManager:
        if self._manager is not None:
            return self._manager

        providers: list[LLMProvider] = [GroqProvider(
            api_key=self._settings.groq_api_key,
            base_url=self._settings.groq_base_url,
            model=self._settings.groq_model,
            timeout=self._settings.llm_timeout,
        )]

        self._manager = ProviderManager(
            providers=providers,
            enable_failover=False,
            enable_circuit_breaker=self._settings.enable_circuit_breaker,
        )
        return self._manager

    @property
    def _provider_manager(self) -> ProviderManager:
        return self._build_manager()

    @property
    def provider(self) -> str:
        return self._provider_manager.active_provider_name

    @property
    def model_name(self) -> str:
        return self._provider_manager.active_model_name

    @property
    def api_key(self) -> str:
        return self._settings.groq_api_key

    @property
    def base_url(self) -> str:
        return self._settings.groq_base_url.rstrip("/")

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._provider_manager.primary_provider.circuit_breaker

    @property
    def health_monitor(self) -> ProviderHealthMonitor:
        return self._provider_manager.primary_provider.health

    @property
    def provider_manager(self) -> ProviderManager:
        return self._provider_manager

    # ------------------------------------------------------------------
    # Conversation history management
    # ------------------------------------------------------------------

    @property
    def conversation_history(self) -> list[dict[str, Any]]:
        return list(self._conversation_history)

    def store_exchange(self, user_message: str, assistant_response: str) -> None:
        self._conversation_history.append({
            "role": "user",
            "content": user_message.strip(),
        })
        self._conversation_history.append({
            "role": "assistant",
            "content": assistant_response.strip(),
        })
        if len(self._conversation_history) > self._MAX_CONVERSATION_TURNS * 2:
            self._conversation_history = self._conversation_history[-(self._MAX_CONVERSATION_TURNS * 2):]

    def clear_conversation(self) -> None:
        self._conversation_history.clear()

    def get_filtered_conversation(self, query: str) -> list[dict[str, Any]]:
        if not self._conversation_history:
            return []
        result = filter_conversation(query, self._conversation_history)
        self._logger.info(
            "Conversation filter | query_type=%s | before=%d | after=%d | discarded=%d | %.2fms",
            result.query_type.value,
            result.total_before,
            result.total_after,
            len(result.discarded),
            result.execution_time_ms,
        )
        return result.filtered_history

    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        pass

    async def generate_text(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        return await self._chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def generate_with_context(
        self,
        user_message: str,
        system_prompt: str,
        memory_context: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

        # Section order: User Facts → Conversation → Current Message
        content_parts: list[str] = []

        if memory_context.strip():
            content_parts.append(memory_context.strip())

        filtered_conv = self.get_filtered_conversation(user_message)
        if filtered_conv:
            conv_block = PromptBuilder.conversation_block(filtered_conv)
            if conv_block:
                content_parts.append(conv_block)

        content_parts.append(f"Current User Message:\n{user_message.strip()}")

        user_turn = "\n\n".join(content_parts)
        messages.append({"role": "user", "content": user_turn})

        self._logger.info(
            "generate_with_context | memory_present=%s | conv_turns=%d | total_messages=%d | user_length=%d",
            bool(memory_context.strip()),
            len(filtered_conv),
            len(messages),
            len(user_message),
        )

        return await self._chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def _chat_completion(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        effective_max_tokens = max_tokens if max_tokens is not None else self._DEFAULT_MAX_TOKENS
        effective_temperature = temperature if temperature is not None else self._DEFAULT_TEMPERATURE

        self._logger.info(
            "LLM request | provider=%s | model=%s | message_count=%d | max_tokens=%d | temperature=%.2f",
            self.provider,
            self.model_name,
            len(messages),
            effective_max_tokens,
            effective_temperature,
        )

        return await self._provider_manager.generate(
            messages=messages,
            max_tokens=effective_max_tokens,
            temperature=effective_temperature,
        )
