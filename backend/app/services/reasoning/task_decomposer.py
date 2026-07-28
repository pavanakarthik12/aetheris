"""Module 3 — Task Decomposer.

Splits multi-objective requests into independent internal tasks.
"""

from __future__ import annotations

import re
from typing import Any

from ...schemas.reasoning import DecomposedTask, SemanticIntentType

_SEPARATOR = re.compile(r"\s+(and\s+also|and\s+then|and|plus|also|additionally|then|furthermore|moreover)\s+", re.IGNORECASE)

_END_OF_TASK = re.compile(r"[.!?]")


def decompose(message: str) -> list[DecomposedTask]:
    parts = _SEPARATOR.split(message.strip())
    tasks: list[DecomposedTask] = []

    for part in parts:
        part = part.strip()
        if not part or part.lower() in ("and", "also", "plus", "then", "additionally", "and also", "and then", "furthermore", "moreover"):
            continue

        sentences = [s.strip() for s in _END_OF_TASK.split(part) if s.strip()]
        for sentence in sentences:
            if sentence:
                tasks.append(DecomposedTask(
                    description=sentence,
                    semantic_intent=_intent_from_text(sentence),
                ))

    if len(tasks) <= 1:
        return []

    return tasks


def _intent_from_text(text: str) -> SemanticIntentType:
    from .intent_analyzer import analyze_intent
    return analyze_intent(text)
