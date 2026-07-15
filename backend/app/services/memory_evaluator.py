"""Memory evaluator — decides whether a user message should become a permanent memory."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ..services.llm_service import LLMService, LLMServiceError
from .prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

# Messages matching these patterns skip LLM evaluation (saves one call per greeting).
_SKIP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(hi|hey|hello|yo|sup|howdy)\W*$", re.IGNORECASE),
    re.compile(r"^(good )?(morning|afternoon|evening)\W*$", re.IGNORECASE),
    re.compile(r"^thanks?\W*$", re.IGNORECASE),
    re.compile(r"^bye|goodbye|see ya|later\W*$", re.IGNORECASE),
    re.compile(r"^(ok|okay|k|sure|fine|alright)\W*$", re.IGNORECASE),
    re.compile(r"^(what'?s up|how are you|how'?s it going)\W*$", re.IGNORECASE),
]

_ALLOWED_CATEGORIES = (
    "Preference",
    "Project",
    "Goal",
    "Skill",
    "Relationship",
    "Fact",
    "Achievement",
    "Event",
    "Task",
    "Other",
)


class MemoryEvaluatorServiceError(RuntimeError):
    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class MemoryEvaluation(BaseModel):
    store: bool = Field(description="Whether the message should be persisted.")
    importance: float = Field(ge=0.0, le=1.0, description="Importance score from 0 to 1.")
    category: Literal[
        "Preference",
        "Project",
        "Goal",
        "Skill",
        "Relationship",
        "Fact",
        "Achievement",
        "Event",
        "Task",
        "Other",
    ] = Field(description="The memory category.")
    reason: str = Field(min_length=1, max_length=500, description="Short explanation for the decision.")


class MemoryEvaluatorService:
    """Classify a user message and decide whether it should be stored."""

    def __init__(self, llm_service: LLMService) -> None:
        self._llm = llm_service
        self._logger = logging.getLogger(__name__)

    async def evaluate_memory(self, memory_text: str) -> MemoryEvaluation:
        if not memory_text or not memory_text.strip():
            raise MemoryEvaluatorServiceError("memory_text must not be empty.", status_code=422)

        text = memory_text.strip()

        # Rule-based fast path: skip LLM for obvious non-memories.
        if _is_obviously_not_memory(text):
            self._logger.info("Memory evaluation skipped by rule | text=%r", text)
            return MemoryEvaluation(
                store=False,
                importance=0.0,
                category="Other",
                reason="Not stored (greeting/filler detected by rule).",
            )

        system_prompt = PromptBuilder.memory_evaluation_system()
        user_prompt = PromptBuilder.memory_evaluation_user(text)

        try:
            raw_response = await self._llm.generate_with_context(
                user_message=user_prompt,
                system_prompt=system_prompt,
            )
        except LLMServiceError as exc:
            raise MemoryEvaluatorServiceError(str(exc), status_code=exc.status_code) from exc

        evaluation = self._parse_response(raw_response)
        self._logger.info(
            "Memory evaluation result | store=%s | category=%s | importance=%.2f | reason=%s",
            evaluation.store,
            evaluation.category,
            evaluation.importance,
            evaluation.reason,
        )
        return evaluation

    def _parse_response(self, raw_response: str) -> MemoryEvaluation:
        candidate = raw_response.strip()

        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"\s*```$", "", candidate)

        try:
            payload: Any = json.loads(candidate)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", candidate)
            if not match:
                raise MemoryEvaluatorServiceError(
                    "Memory evaluator returned invalid JSON.",
                    status_code=500,
                )
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise MemoryEvaluatorServiceError(
                    "Memory evaluator returned invalid JSON.",
                    status_code=500,
                ) from exc

        try:
            return MemoryEvaluation.model_validate(payload)
        except ValidationError as exc:
            raise MemoryEvaluatorServiceError(
                f"Memory evaluator returned an invalid decision: {exc}",
                status_code=500,
            ) from exc


def _is_obviously_not_memory(text: str) -> bool:
    """Return True if the text is almost certainly not worth persisting."""
    if len(text.split()) <= 3:
        return True
    if any(p.search(text) for p in _SKIP_PATTERNS):
        return True
    return False
