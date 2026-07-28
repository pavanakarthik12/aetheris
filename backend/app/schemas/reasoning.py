"""Pydantic models for the Cognitive Reasoning Engine."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SemanticIntentType(str, Enum):
    SIMPLE_QUESTION = "SIMPLE_QUESTION"
    PROGRAMMING = "PROGRAMMING"
    MEMORY_RETRIEVAL = "MEMORY_RETRIEVAL"
    CONVERSATION_CONTINUATION = "CONVERSATION_CONTINUATION"
    EXPLANATION = "EXPLANATION"
    PLANNING = "PLANNING"
    CREATIVE_WRITING = "CREATIVE_WRITING"
    CODE_GENERATION = "CODE_GENERATION"
    DEBUGGING = "DEBUGGING"
    REASONING = "REASONING"
    MATHEMATICS = "MATHEMATICS"
    GENERAL_CONVERSATION = "GENERAL_CONVERSATION"
    GREETING = "GREETING"
    UNKNOWN = "UNKNOWN"


class ComplexityLevel(str, Enum):
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class MemorySourceDecision(BaseModel):
    conversation_memory: bool = Field(default=False)
    long_term_memory: bool = Field(default=False)
    system_memory: bool = Field(default=False)
    reflection: bool = Field(default=False)


class DecomposedTask(BaseModel):
    description: str = Field(description="Short description of the sub-task.")
    semantic_intent: SemanticIntentType = Field(default=SemanticIntentType.UNKNOWN)


class ReasoningPlan(BaseModel):
    semantic_intent: SemanticIntentType = Field(
        default=SemanticIntentType.UNKNOWN,
        description="What the user is semantically asking about.",
    )
    complexity: ComplexityLevel = Field(
        default=ComplexityLevel.SIMPLE,
        description="How difficult the request is.",
    )
    tasks: list[DecomposedTask] = Field(
        default_factory=list,
        description="Decomposed sub-tasks if multi-objective.",
    )
    memory_sources: MemorySourceDecision = Field(
        default_factory=MemorySourceDecision,
        description="Which memory sources are needed.",
    )
    needs_clarification: bool = Field(default=False)
    clarification_question: str = Field(default="")
    needs_planning: bool = Field(default=False)
    planning_steps: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.HIGH)


class CognitiveTrace(BaseModel):
    semantic_intent: SemanticIntentType = Field(default=SemanticIntentType.UNKNOWN)
    complexity: ComplexityLevel = Field(default=ComplexityLevel.SIMPLE)
    task_count: int = Field(default=0)
    memory_conversation: bool = Field(default=False)
    memory_long_term: bool = Field(default=False)
    memory_system: bool = Field(default=False)
    memory_reflection: bool = Field(default=False)
    needs_clarification: bool = Field(default=False)
    needs_planning: bool = Field(default=False)
    planning_step_count: int = Field(default=0)
    verification_passed: bool = Field(default=True)
    verification_detail: str = Field(default="")
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.HIGH)
    reasoning_duration_ms: float = Field(default=0.0)


class VerificationResult(BaseModel):
    passed: bool = Field(default=True)
    detail: str = Field(default="")
