"""Schemas for the External Knowledge Provider layer and search pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


# ── Internal dataclasses (provider layer) ──────────────────────────

@dataclass
class SearchResult:
    title: str = ""
    url: str = ""
    snippet: str = ""
    content: str = ""
    source: str = ""
    score: float = 0.0


@dataclass
class SearchResponse:
    results: list[SearchResult] = field(default_factory=list)
    answer: str = ""
    total_results: int = 0
    duration_ms: float = 0.0
    provider: str = ""


# ── API response models (search pipeline) ──────────────────────────

class NormalizedSearchResultItem(BaseModel):
    title: str = Field(default="", description="Result title.")
    url: str = Field(default="", description="Result URL.")
    snippet: str = Field(default="", description="Short text snippet.")
    content: str = Field(default="", description="Full content if available.")
    source: str = Field(default="", description="Provider name.")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Relevance score.")


class SearchQueryRequest(BaseModel):
    q: str = Field(min_length=1, max_length=500, description="Search query.")


class SearchDebugResponse(BaseModel):
    provider: str = Field(default="", description="Provider that executed the search.")
    query: str = Field(default="", description="Original search query.")
    success: bool = Field(default=False, description="Whether the search succeeded.")
    execution_time_ms: float = Field(default=0.0, description="Execution time in milliseconds.")
    result_count: int = Field(default=0, description="Number of results returned.")
    results: list[NormalizedSearchResultItem] = Field(default_factory=list, description="Search results.")
    error: str | None = Field(default=None, description="Error message if search failed.")
