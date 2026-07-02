"""Chat request and response schemas for the Phase 1 API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Single-message chat request payload."""

    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    """Single-message chat response payload."""

    response: str