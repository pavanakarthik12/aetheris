"""Provider Manager for External Knowledge Providers.

Initializes providers, validates configuration, and exposes
the active provider. The rest of Aetheris interacts only
with this manager — never with concrete provider classes.
"""

from __future__ import annotations

import logging
from typing import Any

from ...schemas.external_knowledge import SearchResponse
from .base_provider import ExternalKnowledgeProvider
from .tavily_provider import TavilyProvider

logger = logging.getLogger(__name__)


class ExternalKnowledgeManager:

    def __init__(
        self,
        tavily_api_key: str = "",
    ) -> None:
        self._providers: dict[str, ExternalKnowledgeProvider] = {}
        self._active_provider: ExternalKnowledgeProvider | None = None
        self._initialized = False
        self._init_error: str | None = None

        self._init_providers(tavily_api_key)

    def _init_providers(self, tavily_api_key: str) -> None:
        if tavily_api_key:
            tavily = TavilyProvider(api_key=tavily_api_key)
            self._providers["tavily"] = tavily
            self._active_provider = tavily
            logger.info("ExternalKnowledge | Tavily provider initialized")
        else:
            logger.warning(
                "ExternalKnowledge | TAVILY_API_KEY not set — "
                "external knowledge search is disabled."
            )
            self._init_error = "TAVILY_API_KEY is not configured"

        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def init_error(self) -> str | None:
        return self._init_error

    @property
    def active_provider(self) -> ExternalKnowledgeProvider | None:
        return self._active_provider

    @property
    def active_provider_name(self) -> str:
        if self._active_provider is not None:
            return self._active_provider.provider_name
        return "none"

    @property
    def available_providers(self) -> list[str]:
        return list(self._providers.keys())

    def is_available(self) -> bool:
        return self._active_provider is not None and self._active_provider.is_available()

    async def health_check(self) -> bool:
        if self._active_provider is None:
            return False
        try:
            return await self._active_provider.health_check()
        except Exception:
            return False

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> SearchResponse:
        if self._active_provider is None:
            logger.warning("Search called but no external knowledge provider is active")
            return SearchResponse(
                total_results=0,
                provider="none",
            )
        return await self._active_provider.search(query, max_results=max_results)

    def snapshot(self) -> dict[str, Any]:
        return {
            "initialized": self._initialized,
            "active_provider": self.active_provider_name,
            "available_providers": self.available_providers,
            "is_available": self.is_available(),
            "init_error": self._init_error,
        }
