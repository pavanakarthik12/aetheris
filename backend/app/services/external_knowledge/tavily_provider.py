"""Tavily search provider implementation.

Communicates with the Tavily Search API using Bearer token
authentication and normalizes responses into Aetheris'
internal SearchResult format.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import httpx

from ...schemas.external_knowledge import SearchResponse, SearchResult
from .base_provider import ExternalKnowledgeProvider

_TAVILY_API_URL: str = "https://api.tavily.com/search"
_DEFAULT_TIMEOUT: float = 15.0

logger = logging.getLogger(__name__)


class TavilyProvider(ExternalKnowledgeProvider):

    def __init__(
        self,
        api_key: str,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def provider_name(self) -> str:
        return "tavily"

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def health_check(self) -> bool:
        if not self.is_available():
            return False
        try:
            client = self._get_client()
            response = await client.post(
                _TAVILY_API_URL,
                json={"query": "health", "max_results": 1},
                timeout=httpx.Timeout(5.0),
            )
            return response.is_success
        except Exception:
            return False

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> SearchResponse:
        started = perf_counter()
        client = self._get_client()

        payload: dict[str, Any] = {
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "include_images": False,
            "include_raw_content": False,
            "max_results": max_results,
        }

        try:
            response = await client.post(
                _TAVILY_API_URL,
                json=payload,
                timeout=httpx.Timeout(self._timeout),
            )
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException:
            logger.warning("Tavily search timed out | query=%.50s", query)
            return SearchResponse(
                total_results=0,
                duration_ms=(perf_counter() - started) * 1000,
                provider=self.provider_name,
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            body = self._extract_error_body(exc.response)
            logger.error(
                "Tavily search failed | status=%d | body=%s | query=%.50s",
                status, body, query,
            )
            return SearchResponse(
                total_results=0,
                duration_ms=(perf_counter() - started) * 1000,
                provider=self.provider_name,
            )
        except httpx.RequestError as exc:
            logger.error("Tavily request error | error=%s | query=%.50s", exc, query)
            return SearchResponse(
                total_results=0,
                duration_ms=(perf_counter() - started) * 1000,
                provider=self.provider_name,
            )

        raw_results = data.get("results", [])
        results: list[SearchResult] = []
        for r in raw_results:
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", "")[:300] if r.get("content") else "",
                content=r.get("raw_content") or r.get("content", ""),
                source=self.provider_name,
                score=r.get("score", 0.0),
            ))

        elapsed = (perf_counter() - started) * 1000
        logger.info(
            "Tavily search | query=%.50s | results=%d | duration_ms=%.2f",
            query, len(results), elapsed,
        )

        return SearchResponse(
            results=results,
            answer=data.get("answer", ""),
            total_results=len(results),
            duration_ms=elapsed,
            provider=self.provider_name,
        )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        return self._client

    @staticmethod
    def _extract_error_body(response: httpx.Response) -> str:
        try:
            return response.text[:500]
        except Exception:
            return f"HTTP {response.status_code}"
