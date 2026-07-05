"""Chat request and response schemas for the Aetheris chat API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class MemoryActionType(str, Enum):
    """The type of memory action performed by the ImmediateMemoryProcessor."""

    CREATE = "CREATE"
    UPDATE = "UPDATE"
    MERGE = "MERGE"
    DELETE = "DELETE"
    ARCHIVE = "ARCHIVE"
    SKIP = "SKIP"
    ERROR = "ERROR"


class ImmediateMemoryResult(BaseModel):
    """Result returned by the ImmediateMemoryProcessor after handling a message."""

    action: MemoryActionType = Field(description="The action that was taken.")
    success: bool = Field(description="Whether the operation completed successfully.")
    memory_id: str | None = Field(default=None, description="The ID of the affected memory, if any.")
    error: str | None = Field(default=None, description="Error message if the operation failed.")


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
    memory_action: MemoryActionType = Field(
        default=MemoryActionType.SKIP,
        description="The memory action taken for this message (CREATE / UPDATE / MERGE / DELETE / etc.).",
    )
    memory_success: bool = Field(
        default=True,
        description="Whether the memory operation completed successfully.",
    )
    memory_error: str | None = Field(
        default=None,
        description="Error message if the memory operation failed.",
    )
