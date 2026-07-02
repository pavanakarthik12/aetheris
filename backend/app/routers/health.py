"""Health check router for infrastructure readiness."""

from fastapi import APIRouter

from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return a minimal health response for uptime checks."""

    return HealthResponse(status="ok", service="aetheris-backend")