"""Pydantic models for the Reflection Engine."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ReflectionAction(str, Enum):
    CREATE_MEMORY = "CREATE_MEMORY"
    UPDATE_MEMORY = "UPDATE_MEMORY"
    MERGE_MEMORY = "MERGE_MEMORY"
    STRENGTHEN_MEMORY = "STRENGTHEN_MEMORY"
    NO_ACTION = "NO_ACTION"


class ReflectionOutput(BaseModel):
    new_memory: bool = False
    update_memory: bool = False
    memory_strengthened: bool = False
    assistant_mistake: bool = False
    user_corrected_ai: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    future_behavior_change: bool = False
    reflection_summary: str = ""
    actions: list[ReflectionAction] = [ReflectionAction.NO_ACTION]


class ReflectionRecord(BaseModel):
    id: str
    timestamp: str
    user_message: str
    assistant_response: str
    reflection: ReflectionOutput


class ReflectionListResponse(BaseModel):
    reflections: list[dict[str, Any]]
    total: int


class ReflectionDetailResponse(BaseModel):
    reflection: dict[str, Any]
