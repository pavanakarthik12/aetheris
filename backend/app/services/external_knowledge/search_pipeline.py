"""External Knowledge Search Pipeline — MVP.

Validates queries, executes searches through the Provider Manager,
and returns normalized results. Never interacts with providers directly.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from ...schemas.external_knowledge import (
    NormalizedSearchResultItem,
    SearchDebugResponse,
)
from .provider_manager import ExternalKnowledgeManager

_QUERY_MAX_LENGTH: int = 500

logger = logging.getLogger(__name__)


class SearchPipelineError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class SearchPipeline:

    def __init__(self, manager: ExternalKnowledgeManager) -> None:
        self._manager = manager

    async def execute(
        self,
        query: str,
        max_results: int = 5,
    ) -> SearchDebugResponse:
        started = perf_counter()

        validation_error = self._validate(query)
        if validation_error:
            elapsed = (perf_counter() - started) * 1000
            logger.warning(
                "Search validation failed | reason=%s | query=%.50s",
                validation_error, query,
            )
            return SearchDebugResponse(
                provider=self._manager.active_provider_name,
                query=query,
                success=False,
                execution_time_ms=elapsed,
                result_count=0,
                results=[],
                error=validation_error,
            )

        if not self._manager.is_available():
            elapsed = (perf_counter() - started) * 1000
            init_err = self._manager.init_error or "No external knowledge provider is available"
            logger.warning("Search unavailable | reason=%s", init_err)
            error_message = "Missing API Key" if "API_KEY" in init_err.upper() else init_err
            return SearchDebugResponse(
                provider=self._manager.active_provider_name,
                query=query,
                success=False,
                execution_time_ms=elapsed,
                result_count=0,
                results=[],
                error=error_message,
            )

        provider_name = self._manager.active_provider_name

        try:
            response = await self._manager.search(query, max_results=max_results)
        except Exception as exc:
            elapsed = (perf_counter() - started) * 1000
            logger.error(
                "Search execution failed | provider=%s | error=%s | query=%.50s",
                provider_name, exc, query,
            )
            return SearchDebugResponse(
                provider=provider_name,
                query=query,
                success=False,
                execution_time_ms=elapsed,
                result_count=0,
                results=[],
                error=f"Search execution failed: {exc}",
            )

        elapsed = (perf_counter() - started) * 1000
        normalized = [
            NormalizedSearchResultItem(
                title=r.title,
                url=r.url,
                snippet=r.snippet,
                content=r.content,
                source=r.source,
                score=r.score,
            )
            for r in response.results
        ]

        success = response.total_results > 0 or not self._is_error_response(response)

        logger.info(
            "Search completed | provider=%s | query=%.50s | results=%d | duration_ms=%.2f | success=%s",
            provider_name, query, len(normalized), elapsed, success,
        )

        return SearchDebugResponse(
            provider=provider_name,
            query=query,
            success=success,
            execution_time_ms=elapsed,
            result_count=len(normalized),
            results=normalized,
            error=None,
        )

    @staticmethod
    def _validate(query: str) -> str | None:
        if not query or not query.strip():
            return "Query cannot be empty"
        stripped = query.strip()
        if len(stripped) > _QUERY_MAX_LENGTH:
            return f"Query exceeds maximum length of {_QUERY_MAX_LENGTH} characters"
        return None

    @staticmethod
    def _is_error_response(response: Any) -> bool:
        return hasattr(response, "total_results") and response.total_results == 0
