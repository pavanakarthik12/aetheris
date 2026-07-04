"""Pydantic schemas for the Memory Evolution API (Phase 6)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------

EvolutionAction = Literal["CREATE", "UPDATE", "MERGE", "ARCHIVE", "SKIP"]


# ---------------------------------------------------------------------------
# POST /api/memory/evolve
# ---------------------------------------------------------------------------


class EvolveRequest(BaseModel):
    """Payload for the end-to-end evolution pipeline."""

    memory_text: str = Field(
        min_length=1,
        max_length=8000,
        description="The text content to evaluate and evolve.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata to attach to the new/updated memory.",
    )


class EvolveResponse(BaseModel):
    """Response returned after the evolution pipeline completes."""

    action: EvolutionAction = Field(description="The action that was taken.")
    memory_id: str = Field(description="UUID of the affected memory (new or existing).")
    version: int = Field(default=1, description="The current version of the memory.")
    explanation: str = Field(default="", description="Human-readable explanation of the decision.")


# ---------------------------------------------------------------------------
# POST /api/memory/update
# ---------------------------------------------------------------------------


class UpdateMemoryRequest(BaseModel):
    """Payload for updating an existing memory."""

    memory_id: str = Field(description="UUID of the memory to update.")
    memory_text: str = Field(
        min_length=1,
        max_length=8000,
        description="The new text content.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata to merge into the existing metadata.",
    )


class UpdateMemoryResponse(BaseModel):
    """Response returned after a successful update."""

    memory_id: str = Field(description="UUID of the updated memory.")
    version: int = Field(description="New version number.")
    status: str = Field(description="Always 'updated' on success.")


# ---------------------------------------------------------------------------
# POST /api/memory/merge
# ---------------------------------------------------------------------------


class MergeMemoryRequest(BaseModel):
    """Payload for merging two memories into one."""

    target_id: str = Field(description="UUID of the memory that will receive the merged content.")
    source_id: str = Field(description="UUID of the memory that will be archived after merge.")
    merged_text: str = Field(
        min_length=1,
        max_length=8000,
        description="The combined text content for the target memory.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata to merge into the target's metadata.",
    )


class MergeMemoryResponse(BaseModel):
    """Response returned after a successful merge."""

    target_id: str = Field(description="UUID of the surviving memory.")
    source_id: str = Field(description="UUID of the archived source memory.")
    version: int = Field(description="New version of the target memory.")
    status: str = Field(description="Always 'merged' on success.")


# ---------------------------------------------------------------------------
# POST /api/memory/archive
# ---------------------------------------------------------------------------


class ArchiveMemoryRequest(BaseModel):
    """Payload for archiving a memory."""

    memory_id: str = Field(description="UUID of the memory to archive.")


class ArchiveMemoryResponse(BaseModel):
    """Response returned after a successful archive."""

    memory_id: str = Field(description="UUID of the archived memory.")
    status: str = Field(description="Always 'archived' on success.")


# ---------------------------------------------------------------------------
# GET /api/memory/history/{memory_id}
# ---------------------------------------------------------------------------


class HistoryEntry(BaseModel):
    """A single entry in the version history of a memory."""

    version: int = Field(description="Version number.")
    text: str = Field(description="The text content at this version.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata snapshot at this version.",
    )
    archived_at: str | None = Field(
        default=None,
        description="ISO-8601 timestamp if this version was archived.",
    )


class MemoryHistoryResponse(BaseModel):
    """Response containing the full version history of a memory."""

    memory_id: str = Field(description="UUID of the memory.")
    current_version: int = Field(description="Current version number.")
    current_text: str = Field(description="Current text content.")
    current_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Current metadata.",
    )
    history: list[HistoryEntry] = Field(
        default_factory=list,
        description="Previous versions in descending version order (newest first).",
    )
