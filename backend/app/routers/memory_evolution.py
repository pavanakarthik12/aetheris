"""Memory Evolution router — Phase 6: intelligent memory lifecycle management.

Endpoints for testing and debugging the evolution engine directly.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_memory_evolution_service
from ..schemas.evolution import (
    ArchiveMemoryRequest,
    ArchiveMemoryResponse,
    EvolveRequest,
    EvolveResponse,
    MemoryHistoryResponse,
    MergeMemoryRequest,
    MergeMemoryResponse,
    UpdateMemoryRequest,
    UpdateMemoryResponse,
)
from ..services.memory_evolution_service import (
    MemoryEvolutionService,
    MemoryEvolutionServiceError,
)

router = APIRouter(prefix="/api/memory", tags=["memory-evolution"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# POST /api/memory/evolve  —  end-to-end evolution pipeline
# ---------------------------------------------------------------------------


@router.post(
    "/evolve",
    response_model=EvolveResponse,
    summary="Run the evolution pipeline on a message",
    description=(
        "Search for related memories, detect conflicts, and decide "
        "whether to CREATE, UPDATE, MERGE, or SKIP.  If the action is "
        "CREATE or UPDATE it is executed immediately."
    ),
)
async def evolve(
    request: EvolveRequest,
    evolution: MemoryEvolutionService = Depends(get_memory_evolution_service),
) -> EvolveResponse:
    """Run the full evolution pipeline and persist the result."""

    logger.info(
        "POST /api/memory/evolve | text_length=%d",
        len(request.memory_text),
    )

    try:
        decision = await evolution.decide_evolution(
            memory_text=request.memory_text,
        )
    except MemoryEvolutionServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    action = decision["action"]
    memory_id: str | None = None
    version: int = 1

    try:
        if action == "CREATE":
            result = await evolution.create_memory(
                memory_text=request.memory_text,
                metadata=request.metadata,
            )
            memory_id = result["memory_id"]
            version = result.get("version", 1)

        elif action == "UPDATE":
            target_id = decision.get("target_id")
            if not target_id:
                raise HTTPException(status_code=500, detail="UPDATE decision missing target_id.")
            result = await evolution.update_memory(
                memory_id=target_id,
                new_text=request.memory_text,
                new_metadata=request.metadata,
            )
            memory_id = result["memory_id"]
            version = result["version"]

        elif action == "MERGE":
            target_id = decision.get("target_id")
            if not target_id:
                raise HTTPException(status_code=500, detail="MERGE decision missing target_id.")

            existing_text = await evolution.get_memory_document(target_id) or ""

            merged_text = f"{existing_text}\n{request.memory_text}"
            result = await evolution.update_memory(
                memory_id=target_id,
                new_text=merged_text,
                new_metadata=request.metadata,
            )
            memory_id = result["memory_id"]
            version = result["version"]

        else:
            memory_id = ""
            version = 0

    except MemoryEvolutionServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return EvolveResponse(
        action=action,
        memory_id=memory_id or "",
        version=version,
        explanation=decision.get("explanation", ""),
    )


# ---------------------------------------------------------------------------
# POST /api/memory/update
# ---------------------------------------------------------------------------


@router.post(
    "/update",
    response_model=UpdateMemoryResponse,
    summary="Update an existing memory (with versioning)",
    description=(
        "Replace the text and optionally the metadata of an existing memory. "
        "The previous version is preserved in the version history."
    ),
)
async def update_memory(
    request: UpdateMemoryRequest,
    evolution: MemoryEvolutionService = Depends(get_memory_evolution_service),
) -> UpdateMemoryResponse:
    """Update a memory by ID with version tracking."""

    logger.info(
        "POST /api/memory/update | id=%s | text_length=%d",
        request.memory_id,
        len(request.memory_text),
    )

    try:
        result = await evolution.update_memory(
            memory_id=request.memory_id,
            new_text=request.memory_text,
            new_metadata=request.metadata,
        )
    except MemoryEvolutionServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return UpdateMemoryResponse(**result)


# ---------------------------------------------------------------------------
# POST /api/memory/merge
# ---------------------------------------------------------------------------


@router.post(
    "/merge",
    response_model=MergeMemoryResponse,
    summary="Merge two memories into one",
    description=(
        "Archives the source memory and updates the target with the "
        "supplied merged text.  The target's version is incremented."
    ),
)
async def merge_memory(
    request: MergeMemoryRequest,
    evolution: MemoryEvolutionService = Depends(get_memory_evolution_service),
) -> MergeMemoryResponse:
    """Merge source into target, then archive the source."""

    logger.info(
        "POST /api/memory/merge | target=%s | source=%s",
        request.target_id,
        request.source_id,
    )

    try:
        result = await evolution.merge_memory(
            target_id=request.target_id,
            source_id=request.source_id,
            merged_text=request.merged_text,
            merged_metadata=request.metadata,
        )
    except MemoryEvolutionServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return MergeMemoryResponse(**result)


# ---------------------------------------------------------------------------
# POST /api/memory/archive
# ---------------------------------------------------------------------------


@router.post(
    "/archive",
    response_model=ArchiveMemoryResponse,
    summary="Archive a memory",
    description=(
        "Mark a memory as archived.  Archived memories are excluded from "
        "normal retrieval but remain accessible for historical queries."
    ),
)
async def archive_memory(
    request: ArchiveMemoryRequest,
    evolution: MemoryEvolutionService = Depends(get_memory_evolution_service),
) -> ArchiveMemoryResponse:
    """Archive a memory by ID."""

    logger.info("POST /api/memory/archive | id=%s", request.memory_id)

    try:
        result = await evolution.archive_memory(request.memory_id)
    except MemoryEvolutionServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return ArchiveMemoryResponse(**result)


# ---------------------------------------------------------------------------
# GET /api/memory/history/{memory_id}
# ---------------------------------------------------------------------------


@router.get(
    "/history/{memory_id}",
    response_model=MemoryHistoryResponse,
    summary="Get version history for a memory",
    description="Return all previous versions of a memory, including metadata snapshots.",
)
async def get_memory_history(
    memory_id: str,
    evolution: MemoryEvolutionService = Depends(get_memory_evolution_service),
) -> MemoryHistoryResponse:
    """Return version history for the given memory."""

    logger.info("GET /api/memory/history/%s", memory_id)

    try:
        result = await evolution.get_history(memory_id)
    except MemoryEvolutionServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return MemoryHistoryResponse(**result)
