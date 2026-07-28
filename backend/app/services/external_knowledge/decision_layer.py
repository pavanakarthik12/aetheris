"""External Knowledge Decision Layer — lightweight keyword-based classifier.

Determines whether a user request requires external (live) knowledge.
Modular and easily replaceable by a more advanced classifier in future phases.
"""

from __future__ import annotations

import logging
from typing import Any

from ...schemas.reasoning import SemanticIntentType

logger = logging.getLogger(__name__)

_TRIGGER_KEYWORDS: list[str] = [
    "latest", "current", "today", "yesterday",
    "this week", "this month",
    "recent", "news", "update", "updates",
    "release", "releases",
    "documentation", "api changes",
    "version", "live information", "internet lookup",
    "weather", "stock", "price", "forecast",
    "who is the current", "who is the present",
    "what is the latest", "what is the current",
]

_INTENTS_SKIP_SEARCH: set[SemanticIntentType] = {
    SemanticIntentType.GREETING,
    SemanticIntentType.SIMPLE_QUESTION,
    SemanticIntentType.MEMORY_RETRIEVAL,
    SemanticIntentType.PLANNING,
    SemanticIntentType.CREATIVE_WRITING,
    SemanticIntentType.CODE_GENERATION,
    SemanticIntentType.UNKNOWN,
}


class ExternalKnowledgeDecisionLayer:

    def needs_external_knowledge(
        self,
        message: str,
        semantic_intent: SemanticIntentType,
    ) -> bool:
        if semantic_intent in _INTENTS_SKIP_SEARCH:
            logger.debug(
                "ExternalKnowledgeDecision | skip | intent=%s",
                semantic_intent.value,
            )
            return False

        msg_lower = message.strip().lower()
        for keyword in _TRIGGER_KEYWORDS:
            if keyword in msg_lower:
                logger.debug(
                    "ExternalKnowledgeDecision | trigger | keyword=%s | intent=%s",
                    keyword, semantic_intent.value,
                )
                return True

        logger.debug(
            "ExternalKnowledgeDecision | no_match | intent=%s",
            semantic_intent.value,
        )
        return False
