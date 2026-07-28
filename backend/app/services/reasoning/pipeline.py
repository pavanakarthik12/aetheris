"""Module 9 — Reasoning Pipeline.

Orchestrates all reasoning modules in sequence for every request.
Simple requests skip unnecessary stages automatically.
"""

from __future__ import annotations

import logging
from time import perf_counter

from ...schemas.reasoning import (
    CognitiveTrace,
    ComplexityLevel,
    MemorySourceDecision,
    ReasoningPlan,
    SemanticIntentType,
    VerificationResult,
)
from .cognitive_trace import CognitiveTracer
from .complexity_classifier import classify_complexity
from .confidence_estimator import estimate_confidence
from .decision_engine import decide
from .intent_analyzer import analyze_intent
from .planning_engine import create_plan
from .response_verifier import verify
from .task_decomposer import decompose

logger = logging.getLogger(__name__)


class ReasoningPipeline:
    def __init__(self) -> None:
        self._tracer = CognitiveTracer()

    async def run(
        self,
        message: str,
    ) -> tuple[ReasoningPlan, CognitiveTrace]:
        started = perf_counter()
        self._tracer.start()

        semantic_intent = analyze_intent(message)
        self._tracer.set_intent(semantic_intent)

        complexity = classify_complexity(message, semantic_intent)
        self._tracer.set_complexity(complexity)

        tasks = decompose(message)
        self._tracer.set_task_count(len(tasks))
        has_tasks = len(tasks) > 0

        memory_sources, needs_clarification, clarification_q, needs_planning = decide(
            semantic_intent=semantic_intent,
            complexity=complexity,
            message=message,
            has_decomposed_tasks=has_tasks,
        )
        self._tracer.set_memory_sources(
            conversation=memory_sources.conversation_memory,
            long_term=memory_sources.long_term_memory,
            system=memory_sources.system_memory,
            reflection=memory_sources.reflection,
        )
        self._tracer.set_clarification(needs_clarification)

        planning_steps: list[str] = []
        if needs_planning and complexity != ComplexityLevel.SIMPLE:
            planning_steps = create_plan(semantic_intent, complexity, message)
        self._tracer.set_planning(needs_planning, len(planning_steps))

        confidence = estimate_confidence(semantic_intent, complexity, self._tracer.build())
        self._tracer.set_confidence(confidence)

        plan = ReasoningPlan(
            semantic_intent=semantic_intent,
            complexity=complexity,
            tasks=tasks,
            memory_sources=memory_sources,
            needs_clarification=needs_clarification,
            clarification_question=clarification_q,
            needs_planning=needs_planning,
            planning_steps=planning_steps,
            confidence=confidence,
        )

        trace = self._tracer.build()

        elapsed = (perf_counter() - started) * 1000
        logger.info(
            "Reasoning pipeline | intent=%s | complexity=%s | tasks=%d | "
            "clarification=%s | planning=%s | confidence=%s | duration_ms=%.2f",
            semantic_intent.value,
            complexity.value,
            len(tasks),
            needs_clarification,
            needs_planning,
            confidence.value,
            elapsed,
        )

        return plan, trace

    async def verify_response(
        self,
        message: str,
        response: str,
    ) -> VerificationResult:
        result = verify(message, response)
        return result
