"""Shared API schemas used by the foundation routes and handlers."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response payload for the health endpoint."""

    status: str = Field(default="ok")
    service: str = Field(default="aetheris")


class ErrorResponse(BaseModel):
    """Generic error payload used by the error handling boundary."""

    type: str
    message: str
    request_id: str | None = None