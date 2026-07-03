"""Chat request and response schemas for the Aetheris chat API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Single-message chat request payload."""

    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    """Single-message chat response payload."""

    response: str = Field(description="The assistant's reply.")
    memory_count: int = Field(
        default=0,
        description=(
            "Number of memories injected into the prompt for this response. "
            "Zero means the reply was generated without memory context."
        ),
    )
