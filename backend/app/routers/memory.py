"""Memory router — Phase 3 semantic memory API."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_memory_service
from ..schemas.memory import (
    DeleteMemoryResponse,
    ListMemoriesResponse,
    MemoryListItem,
    MemorySearchResult,
    SaveMemoryRequest,
    SaveMemoryResponse,
    SearchMemoryRequest,
    SearchMemoryResponse,
)
from ..services.memory_service import MemoryService, MemoryServiceError

router = APIRouter(prefix="/api/memory", tags=["memory"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# POST /api/memory/save
# ---------------------------------------------------------------------------


@router.post(
    "/save",
    response_model=SaveMemoryResponse,
    status_code=201,
    summary="Save a memory",
    description=(
        "Embed the supplied text and persist it in ChromaDB as a new memory record. "
        "A UUID is generated automatically. Metadata is optional; sensible defaults are applied."
    ),
)
async def save_memory(
    request: SaveMemoryRequest,
    memory_service: MemoryService = Depends(get_memory_service),
) -> SaveMemoryResponse:
    """Embed *memory_text* and store it as a new memory."""

    logger.info("POST /api/memory/save | text_length=%d", len(request.memory_text))

    # Flatten MemoryMetadata to a plain dict (None → empty dict for defaults)
    meta: dict[str, Any] = {}
    if request.metadata is not None:
        if hasattr(request.metadata, "model_dump"):
            meta = request.metadata.model_dump(exclude_none=True)
        else:
            meta = request.metadata.dict(exclude_none=True)

    try:
        result = memory_service.save_memory(
            memory_text=request.memory_text,
            metadata=meta or None,
        )
    except MemoryServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return SaveMemoryResponse(**result)


# ---------------------------------------------------------------------------
# POST /api/memory/search
# ---------------------------------------------------------------------------


@router.post(
    "/search",
    response_model=SearchMemoryResponse,
    summary="Search memories semantically",
    description=(
        "Generate an embedding for *query* and retrieve the top-k most semantically similar memories. "
        "No LLM is invoked — results come directly from ChromaDB vector similarity."
    ),
)
async def search_memory(
    request: SearchMemoryRequest,
    memory_service: MemoryService = Depends(get_memory_service),
) -> SearchMemoryResponse:
    """Return the most semantically similar memories for *query*."""

    logger.info(
        "POST /api/memory/search | query_length=%d | top_k=%d",
        len(request.query),
        request.top_k,
    )

    try:
        raw_results = memory_service.search_memory(
            query=request.query,
            top_k=request.top_k,
        )
    except MemoryServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    results = [MemorySearchResult(**r) for r in raw_results]

    return SearchMemoryResponse(
        query=request.query,
        top_k=request.top_k,
        results=results,
        total=len(results),
    )


# ---------------------------------------------------------------------------
# GET /api/memory/list
# ---------------------------------------------------------------------------


@router.get(
    "/list",
    response_model=ListMemoriesResponse,
    summary="List all stored memories",
    description="Return every memory record in the ChromaDB collection. Useful for debugging.",
)
async def list_memories(
    memory_service: MemoryService = Depends(get_memory_service),
) -> ListMemoriesResponse:
    """Return every stored memory."""

    logger.info("GET /api/memory/list")

    try:
        memories = memory_service.list_memories()
    except MemoryServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    items = [MemoryListItem(**m) for m in memories]

    return ListMemoriesResponse(memories=items, total=len(items))


# ---------------------------------------------------------------------------
# DELETE /api/memory/{memory_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/{memory_id}",
    response_model=DeleteMemoryResponse,
    summary="Delete a memory by ID",
    description="Permanently remove the memory identified by *memory_id* from ChromaDB.",
)
async def delete_memory(
    memory_id: str,
    memory_service: MemoryService = Depends(get_memory_service),
) -> DeleteMemoryResponse:
    """Delete the memory with the given UUID."""

    logger.info("DELETE /api/memory/%s", memory_id)

    try:
        result = memory_service.delete_memory(memory_id)
    except MemoryServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return DeleteMemoryResponse(**result)
