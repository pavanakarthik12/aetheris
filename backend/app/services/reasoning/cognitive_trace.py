"""Module 10 — Internal Cognitive Trace.

Records the reasoning path for debugging purposes.
Never shown to users unless debug mode is enabled.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from ...schemas.reasoning import (
    CognitiveTrace,
    ComplexityLevel,
    ConfidenceLevel,
    SemanticIntentType,
)


class CognitiveTracer:
    def __init__(self) -> None:
        self._started: float = 0.0
        self._trace = CognitiveTrace()

    def start(self) -> None:
        self._started = perf_counter()

    def set_intent(self, intent: SemanticIntentType) -> None:
        self._trace.semantic_intent = intent

    def set_complexity(self, complexity: ComplexityLevel) -> None:
        self._trace.complexity = complexity

    def set_task_count(self, count: int) -> None:
        self._trace.task_count = count

    def set_memory_sources(
        self,
        conversation: bool,
        long_term: bool,
        system: bool,
        reflection: bool,
    ) -> None:
        self._trace.memory_conversation = conversation
        self._trace.memory_long_term = long_term
        self._trace.memory_system = system
        self._trace.memory_reflection = reflection

    def set_clarification(self, needed: bool) -> None:
        self._trace.needs_clarification = needed

    def set_planning(self, needed: bool, step_count: int = 0) -> None:
        self._trace.needs_planning = needed
        self._trace.planning_step_count = step_count

    def set_external_knowledge(self, needed: bool) -> None:
        self._trace.needs_external_knowledge = needed

    def set_verification(self, passed: bool, detail: str = "") -> None:
        self._trace.verification_passed = passed
        self._trace.verification_detail = detail

    def set_confidence(self, confidence: ConfidenceLevel) -> None:
        self._trace.confidence = confidence

    def build(self) -> CognitiveTrace:
        elapsed = (perf_counter() - self._started) * 1000
        self._trace.reasoning_duration_ms = round(elapsed, 2)
        return self._trace
