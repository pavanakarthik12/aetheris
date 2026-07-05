"""Reflection router — debug endpoints for the Reflection Engine."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from ..dependencies import get_reflection_service
from ..schemas.reflection import (
    RecentReflectionsResponse,
    ReflectionDetailResponse,
    ReflectionListResponse,
    ReflectionStatistics,
)
from ..services.reflection_service import ReflectionService

router = APIRouter(prefix="/api/reflections", tags=["reflections"])
logger = logging.getLogger(__name__)


@router.get("", response_model=ReflectionListResponse)
async def list_reflections(
    reflection_service: ReflectionService = Depends(get_reflection_service),
) -> ReflectionListResponse:
    """Return all stored reflection records for debugging."""
    reflections = reflection_service.list_reflections()
    return ReflectionListResponse(reflections=reflections, total=len(reflections))


@router.get("/history", response_model=ReflectionListResponse)
async def list_reflections_history(
    reflection_service: ReflectionService = Depends(get_reflection_service),
) -> ReflectionListResponse:
    """Alias for listing all reflections (debug endpoint)."""
    reflections = reflection_service.list_reflections()
    return ReflectionListResponse(reflections=reflections, total=len(reflections))


@router.get("/statistics", response_model=ReflectionStatistics)
async def get_reflection_statistics(
    reflection_service: ReflectionService = Depends(get_reflection_service),
) -> ReflectionStatistics:
    """Return aggregated statistics about all stored reflections."""
    return reflection_service.statistics()


@router.get("/recent", response_model=RecentReflectionsResponse)
async def get_recent_reflections(
    limit: int = Query(default=10, ge=1, le=100),
    reflection_service: ReflectionService = Depends(get_reflection_service),
) -> RecentReflectionsResponse:
    """Return the most recent N reflection records."""
    reflections = reflection_service.recent(limit=limit)
    return RecentReflectionsResponse(reflections=reflections, total=len(reflections))


@router.get("/{reflection_id}", response_model=ReflectionDetailResponse)
async def get_reflection(
    reflection_id: str,
    reflection_service: ReflectionService = Depends(get_reflection_service),
) -> ReflectionDetailResponse:
    """Return a single reflection record by ID for debugging."""
    record = reflection_service.get_reflection(reflection_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Reflection not found")
    return ReflectionDetailResponse(reflection=record)
