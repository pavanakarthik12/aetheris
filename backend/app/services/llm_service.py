from __future__ import annotations

import logging
from typing import Any

from ..config.settings import Settings, get_settings
from .circuit_breaker import CircuitBreaker
from .exceptions import (
    LLMServiceError,
    LLMQuotaExceeded,
    LLMRateLimited,
    ProviderBadRequest,
    ProviderConflict,
    ProviderConnectionError,
    ProviderForbidden,
    ProviderMalformedResponse,
    ProviderNotFound,
    ProviderServerError,
    ProviderTimeout,
    ProviderUnauthorized,
    ProviderUnavailable,
)
from .provider_health import ProviderHealthMonitor
from .providers.groq_provider import GroqProvider
from .providers.openrouter_provider import OpenRouterProvider
from .providers.provider_interface import LLMProvider
from .providers.provider_manager import ProviderManager


class LLMService:
    _DEFAULT_MAX_TOKENS: int = 512
    _DEFAULT_TEMPERATURE: float = 0.7

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._logger = logging.getLogger(__name__)
        self._manager: ProviderManager | None = None
        self._validate_providers()

    def _validate_providers(self) -> None:
        errors: list[str] = []

        if not self._settings.openrouter_api_key:
            errors.append("OPENROUTER_API_KEY is not configured")

        if not self._settings.groq_api_key:
            errors.append("GROQ_API_KEY is not configured")

        if not self._settings.openrouter_base_url.startswith("http"):
            errors.append(f"OPENROUTER_BASE_URL is invalid: {self._settings.openrouter_base_url}")

        if not self._settings.groq_base_url.startswith("http"):
            errors.append(f"GROQ_BASE_URL is invalid: {self._settings.groq_base_url}")

        if not self._settings.openrouter_model:
            errors.append("OPENROUTER_MODEL is not configured")

        if not self._settings.groq_model:
            errors.append("GROQ_MODEL is not configured")

        if errors:
            for err in errors:
                self._logger.warning("Provider config warning: %s", err)

        self._logger.info(
            "Providers loaded | primary=%s (%s) | secondary=%s (%s) | failover=%s | circuit_breaker=%s",
            self._settings.primary_provider,
            self._settings.openrouter_model,
            self._settings.secondary_provider,
            self._settings.groq_model,
            self._settings.enable_provider_failover,
            self._settings.enable_circuit_breaker,
        )

    def _build_manager(self) -> ProviderManager:
        if self._manager is not None:
            return self._manager

        providers: list[LLMProvider] = []

        primary_name = self._settings.primary_provider.lower()
        secondary_name = self._settings.secondary_provider.lower()

        if primary_name == "openrouter":
            providers.append(OpenRouterProvider(
                api_key=self._settings.openrouter_api_key,
                base_url=self._settings.openrouter_base_url,
                model=self._settings.openrouter_model,
                timeout=self._settings.llm_timeout,
            ))
        else:
            providers.append(OpenRouterProvider(
                api_key=self._settings.qwen_api_key,
                base_url=self._settings.qwen_base_url,
                model=self._settings.llm_model,
                timeout=self._settings.llm_timeout,
            ))

        if secondary_name == "groq":
            providers.append(GroqProvider(
                api_key=self._settings.groq_api_key,
                base_url=self._settings.groq_base_url,
                model=self._settings.groq_model,
                timeout=self._settings.llm_timeout,
            ))

        self._manager = ProviderManager(
            providers=providers,
            enable_failover=self._settings.enable_provider_failover,
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
        return self._settings.qwen_api_key

    @property
    def base_url(self) -> str:
        return self._settings.qwen_base_url.rstrip("/")

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._provider_manager.primary_provider.circuit_breaker

    @property
    def health_monitor(self) -> ProviderHealthMonitor:
        return self._provider_manager.primary_provider.health

    @property
    def provider_manager(self) -> ProviderManager:
        return self._provider_manager

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
