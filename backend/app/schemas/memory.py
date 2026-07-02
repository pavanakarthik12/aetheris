"""Pydantic schemas for the Memory API (Phase 3)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared sub-schema
# ---------------------------------------------------------------------------


class MemoryMetadata(BaseModel):
    """Optional metadata that can be attached to a memory record."""

    source: str = Field(default="user", description="Origin of the memory (e.g. 'user', 'system').")
    tags: str = Field(default="", description="Comma-separated topic tags.")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance score between 0 and 1.")

    class Config:
        extra = "allow"  # accept arbitrary additional keys


# ---------------------------------------------------------------------------
# POST /api/memory/save
# ---------------------------------------------------------------------------


class SaveMemoryRequest(BaseModel):
    """Payload for saving a new memory."""

    memory_text: str = Field(
        min_length=1,
        max_length=8000,
        description="The text content to store as a memory.",
    )
    metadata: MemoryMetadata | None = Field(
        default=None,
        description="Optional metadata to attach to the memory.",
    )


class SaveMemoryResponse(BaseModel):
    """Response returned after a memory is successfully saved."""

    memory_id: str = Field(description="Auto-generated UUID for the stored memory.")
    status: str = Field(description="Always 'saved' on success.")
    created_at: str = Field(description="ISO-8601 UTC timestamp of when the memory was stored.")


# ---------------------------------------------------------------------------
# POST /api/memory/search
# ---------------------------------------------------------------------------


class SearchMemoryRequest(BaseModel):
    """Payload for semantic memory search."""

    query: str = Field(
        min_length=1,
        max_length=2000,
        description="Natural-language query to search against stored memories.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of results to return.",
    )


class MemorySearchResult(BaseModel):
    """A single result entry returned by the semantic search."""

    id: str = Field(description="UUID of the matching memory.")
    document: str = Field(description="The stored text content.")
    score: float = Field(description="Cosine similarity score (0–1; higher is more similar).")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Associated metadata.")


class SearchMemoryResponse(BaseModel):
    """Response returned by the memory search endpoint."""

    query: str = Field(description="The original query string.")
    top_k: int = Field(description="The requested result limit.")
    results: list[MemorySearchResult] = Field(description="Ranked list of matching memories.")
    total: int = Field(description="Number of results actually returned.")


# ---------------------------------------------------------------------------
# GET /api/memory/list
# ---------------------------------------------------------------------------


class MemoryListItem(BaseModel):
    """A single record in the full memory listing."""

    id: str = Field(description="UUID of the memory.")
    document: str = Field(description="The stored text content.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Associated metadata.")


class ListMemoriesResponse(BaseModel):
    """Response returned by the list memories endpoint."""

    memories: list[MemoryListItem] = Field(description="All stored memories.")
    total: int = Field(description="Total number of stored memories.")


# ---------------------------------------------------------------------------
# DELETE /api/memory/{memory_id}
# ---------------------------------------------------------------------------


class DeleteMemoryResponse(BaseModel):
    """Response returned after a memory is successfully deleted."""

    memory_id: str = Field(description="UUID of the deleted memory.")
    status: str = Field(description="Always 'deleted' on success.")
