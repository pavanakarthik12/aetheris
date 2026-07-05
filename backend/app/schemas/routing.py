"""Pydantic models for the Cognitive Request Router."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .chat import MemoryActionType


class IntentType(str, Enum):
    """All supported intent classes the router can dispatch."""

    NORMAL_CHAT = "NORMAL_CHAT"
    CREATE_MEMORY = "CREATE_MEMORY"
    UPDATE_MEMORY = "UPDATE_MEMORY"
    DELETE_MEMORY = "DELETE_MEMORY"
    MERGE_MEMORY = "MERGE_MEMORY"
    SEARCH_MEMORY = "SEARCH_MEMORY"
    WEB_SEARCH = "WEB_SEARCH"
    SYSTEM_QUERY = "SYSTEM_QUERY"
    MULTI_ACTION = "MULTI_ACTION"
    UNKNOWN = "UNKNOWN"


class IntentClassification(BaseModel):
    """Result of the intent classifier for a single user message."""

    primary_intent: IntentType = Field(description="The detected primary intent.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the classification.")
    sub_intents: list[IntentType] = Field(
        default_factory=list,
        description="Additional intents for MULTI_ACTION requests.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra info (target memory ID, search query, etc.).",
    )
    classifier_source: str = Field(
        default="rule",
        description="Which classifier module produced this result (rule / llm / hybrid).",
    )


class RouteStep(BaseModel):
    """A single step executed by the router."""

    subsystem: str = Field(description="Name of the subsystem invoked.")
    action: str = Field(description="Action performed.")
    duration_ms: float = Field(default=0.0, description="Time taken in milliseconds.")
    success: bool = Field(default=True, description="Whether the step succeeded.")
    detail: str = Field(default="", description="Result or error detail.")


class RouterDebugInfo(BaseModel):
    """Debug information exposed for dashboard / monitoring."""

    detected_intent: IntentType = Field(
        default=IntentType.UNKNOWN, description="The classified intent.",
    )
    confidence: float = Field(default=0.0, description="Classification confidence.")
    route_taken: str = Field(default="", description="Human-readable route description.")
    steps: list[RouteStep] = Field(default_factory=list, description="All execution steps.")
    total_duration_ms: float = Field(default=0.0, description="Total router execution time.")
    subsystems_used: list[str] = Field(default_factory=list, description="Subsystems invoked.")
    memory_action: MemoryActionType = Field(
        default=MemoryActionType.SKIP,
        description="Memory action performed (if any).",
    )
    memory_operation_count: int = Field(default=0, description="Number of memory operations.")
    reflection_triggered: bool = Field(default=False, description="Whether reflection was scheduled.")
    internet_used: bool = Field(default=False, description="Whether web search was invoked.")


class RouterResult(BaseModel):
    """Complete result returned by the Cognitive Request Router."""

    response: str = Field(description="The final response text.")
    memory_count: int = Field(default=0, description="Number of memories injected into context.")
    memory_action: MemoryActionType = Field(
        default=MemoryActionType.SKIP,
        description="Memory action performed.",
    )
    memory_success: bool = Field(default=True, description="Whether memory operation succeeded.")
    memory_error: str | None = Field(default=None, description="Error detail if memory op failed.")
    debug: RouterDebugInfo = Field(
        default_factory=RouterDebugInfo,
        description="Debug info for dashboard / monitoring.",
    )
