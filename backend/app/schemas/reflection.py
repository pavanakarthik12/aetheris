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
    DELETE_MEMORY = "DELETE_MEMORY"
    ARCHIVE_MEMORY = "ARCHIVE_MEMORY"
    NO_ACTION = "NO_ACTION"


class AssistantQuality(BaseModel):
    correctness: float = Field(default=1.0, ge=0.0, le=1.0)
    completeness: float = Field(default=1.0, ge=0.0, le=1.0)
    relevance: float = Field(default=1.0, ge=0.0, le=1.0)
    clarity: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    hallucination_risk: float = Field(default=0.0, ge=0.0, le=1.0)


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
    action: ReflectionAction = ReflectionAction.NO_ACTION
    memory_strength_delta: float = 0.0
    consistency: bool = True
    assistant_quality: AssistantQuality = Field(default_factory=AssistantQuality)
    user_preferences_detected: list[str] = []
    reasoning: str = ""
    requires_manual_review: bool = False


class ReflectionRecord(BaseModel):
    id: str
    timestamp: str
    user_message: str
    assistant_response: str
    reflection: ReflectionOutput
    processing_time_ms: float = 0.0


class ReflectionListResponse(BaseModel):
    reflections: list[dict[str, Any]]
    total: int


class ReflectionDetailResponse(BaseModel):
    reflection: dict[str, Any]


class ReflectionStatistics(BaseModel):
    total_reflections: int
    most_common_action: str
    action_counts: dict[str, int]
    average_confidence: float
    average_quality: AssistantQuality = Field(default_factory=AssistantQuality)
    average_processing_time_ms: float


class RecentReflectionsResponse(BaseModel):
    reflections: list[dict[str, Any]]
    total: int
