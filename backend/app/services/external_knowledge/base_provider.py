"""Abstract base class for all external knowledge providers.

Every search provider must implement this interface.
The rest of Aetheris communicates only through this abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ...schemas.external_knowledge import SearchResponse


class ExternalKnowledgeProvider(ABC):

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> SearchResponse:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...
