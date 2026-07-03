"""Memory evaluator — decides whether a user message should become a permanent memory."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ..services.llm_service import LLMService, LLMServiceError

logger = logging.getLogger(__name__)

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
    """Raised when the evaluator cannot produce a valid decision."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class MemoryEvaluation(BaseModel):
    """Validated evaluator output used to control memory storage."""

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
        """Return a structured store/ignore decision for *memory_text*."""
        if not memory_text or not memory_text.strip():
            raise MemoryEvaluatorServiceError("memory_text must not be empty.", status_code=422)

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(memory_text)

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

    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "You are the Aetheris Memory Evaluator. "
            "Analyze only the latest user message and decide whether it should be stored as permanent memory. "
            "Return ONLY valid JSON. No markdown, no code fences, no commentary.\n\n"
            "Required JSON shape:\n"
            '{"store": true, "importance": 0.93, "category": "Project", "reason": "The user shared a long-term project."}\n\n'
            "Rules:\n"
            "- Store long-term goals, user preferences, projects, skills, personal facts, relationships, achievements, important events, and actionable tasks.\n"
            "- Do not store greetings, small talk, one-off calculations, temporary questions, generic acknowledgements, or filler conversation.\n"
            "- Use only these categories: Preference, Project, Goal, Skill, Relationship, Fact, Achievement, Event, Task, Other.\n"
            "- If the message is not worth storing, set store to false, choose category Other, and give a brief reason.\n"
            "- Importance must be a number from 0 to 1.\n"
        )

    @staticmethod
    def _build_user_prompt(memory_text: str) -> str:
        return (
            "Latest user message:\n"
            f"{memory_text.strip()}\n\n"
            "Respond with only the JSON object."
        )

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