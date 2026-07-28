"""Temporary debug endpoint for External Knowledge Search Pipeline.

DEV ONLY — this endpoint exists solely for testing the search pipeline.
Do not use in production. Will be replaced by integration with the
Cognitive Reasoning Engine in a future phase.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_search_pipeline
from ..schemas.external_knowledge import SearchDebugResponse
from ..services.external_knowledge.search_pipeline import SearchPipeline

router = APIRouter(prefix="/api/debug", tags=["debug"])
logger = logging.getLogger(__name__)


@router.get(
    "/search",
    response_model=SearchDebugResponse,
    summary="Execute an external knowledge search (DEV ONLY)",
    description=(
        "Temporary development endpoint for testing the External Knowledge "
        "Search Pipeline. Validates the query, executes a search through "
        "the active provider, and returns normalized results.\n\n"
        "This endpoint is for development and testing only — it will be "
        "removed once the search pipeline is integrated with the Cognitive "
        "Reasoning Engine."
    ),
)
async def debug_search(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    max_results: int = Query(default=5, ge=1, le=20, description="Maximum number of results"),
    pipeline: SearchPipeline = Depends(get_search_pipeline),
) -> SearchDebugResponse:
    logger.info(
        "Debug search requested | query=%.50s | max_results=%d",
        q, max_results,
    )
    return await pipeline.execute(query=q, max_results=max_results)
