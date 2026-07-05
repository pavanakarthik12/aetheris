"""Intent Classifier — determines user intent from a message.

Uses a modular architecture:
  - RuleIntentClassifier (fast, keyword-based)
  - LLMIntentClassifier (LLM-based for ambiguous cases)
  - IntentClassifier (combined, tries rules first then LLM fallback)

Future classifiers can be added by extending BaseIntentClassifier.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from ..schemas.routing import IntentClassification, IntentType
from .llm_service import LLMService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule-based triggers
# ---------------------------------------------------------------------------

_DELETE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(forget|delete|remove|erase|discard)\b", re.IGNORECASE),
]
_UPDATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(actually|instead of|no longer|not anymore|"
        r"used to|switched to|changed my|update|correct|revise)\b",
        re.IGNORECASE,
    ),
]
_MERGE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(also|additionally|in addition|combine|merge)\b", re.IGNORECASE),
]
_SEARCH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(search|find|look up|retrieve|get|show|list) .* (memory|memories)\b"
        r"|\bwhat (do|did) (i|you) (know|remember|store|have|say|tell|recall)\b"
        r"|\b(do|did) (you|we) (remember|recall|know about)\b"
        r"|\brecall (that|what|my|the|our)\b",
        re.IGNORECASE,
    ),
]
_WEB_SEARCH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(search (the )?(web|internet|online)|"
        r"(look up|find|google|browse) (.* )?(online|web|internet)|"
        r"what('s| is) the (latest|current|recent)|news about|"
        r"tell me (about|more on) (.* )?(latest|current|recent|news))\b",
        re.IGNORECASE,
    ),
]
_SYSTEM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(who (are you|made you|created you)|"
        r"what (are you|can you do|is your name|"
        r"capabilities|version|purpose)|how do you work|"
        r"system (prompt|instruction|config|settings))\b",
        re.IGNORECASE,
    ),
]
_CREATE_MEMORY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(remember that|remember this|save this|store this|"
        r"note that|keep in mind|don't forget)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmy (\w+ )?(is|are|was|were)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi (like|love|enjoy|prefer|want|need|have|use|am)\b",
        re.IGNORECASE,
    ),
]
_MULTI_ACTION_SEPARATOR = re.compile(
    r"\s+(and|also|plus|then|additionally)\s+",
    re.IGNORECASE,
)
_MULTI_ACTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r".*\b(and|also)\b.*\b(search|find|look up|remember|save)\b.*",
        re.IGNORECASE,
    ),
]


class BaseIntentClassifier(ABC):
    """Abstract base for all intent classifier implementations."""

    @abstractmethod
    async def classify(self, message: str) -> IntentClassification:
        """Return the classified intent for *message*."""


class RuleIntentClassifier(BaseIntentClassifier):
    """Fast keyword/pattern-based intent classification.

    Priority order (first match wins):
      1. MULTI_ACTION
      2. DELETE_MEMORY
      3. CREATE_MEMORY
      4. UPDATE_MEMORY
      5. MERGE_MEMORY
      6. SEARCH_MEMORY
      7. WEB_SEARCH
      8. SYSTEM_QUERY
      9. NORMAL_CHAT (default)
    """

    async def classify(self, message: str) -> IntentClassification:
        if not message or not message.strip():
            return IntentClassification(
                primary_intent=IntentType.UNKNOWN,
                confidence=1.0,
                classifier_source="rule",
            )

        # MULTI_ACTION — contains multiple distinct intents
        if any(p.search(message) for p in _MULTI_ACTION_PATTERNS):
            sub_intents = self._detect_sub_intents(message)
            if len(sub_intents) > 1:
                return IntentClassification(
                    primary_intent=IntentType.MULTI_ACTION,
                    confidence=0.85,
                    sub_intents=sub_intents,
                    metadata={"sub_intents": [i.value for i in sub_intents]},
                    classifier_source="rule",
                )

        # DELETE_MEMORY
        if any(p.search(message) for p in _DELETE_PATTERNS):
            return IntentClassification(
                primary_intent=IntentType.DELETE_MEMORY,
                confidence=0.9,
                classifier_source="rule",
            )

        # CREATE_MEMORY
        if any(p.search(message) for p in _CREATE_MEMORY_PATTERNS):
            return IntentClassification(
                primary_intent=IntentType.CREATE_MEMORY,
                confidence=0.8,
                classifier_source="rule",
            )

        # UPDATE_MEMORY
        if any(p.search(message) for p in _UPDATE_PATTERNS):
            return IntentClassification(
                primary_intent=IntentType.UPDATE_MEMORY,
                confidence=0.75,
                classifier_source="rule",
            )

        # MERGE_MEMORY
        if any(p.search(message) for p in _MERGE_PATTERNS):
            return IntentClassification(
                primary_intent=IntentType.MERGE_MEMORY,
                confidence=0.7,
                classifier_source="rule",
            )

        # SEARCH_MEMORY
        if any(p.search(message) for p in _SEARCH_PATTERNS):
            return IntentClassification(
                primary_intent=IntentType.SEARCH_MEMORY,
                confidence=0.8,
                classifier_source="rule",
            )

        # WEB_SEARCH
        if any(p.search(message) for p in _WEB_SEARCH_PATTERNS):
            return IntentClassification(
                primary_intent=IntentType.WEB_SEARCH,
                confidence=0.85,
                classifier_source="rule",
            )

        # SYSTEM_QUERY
        if any(p.search(message) for p in _SYSTEM_PATTERNS):
            return IntentClassification(
                primary_intent=IntentType.SYSTEM_QUERY,
                confidence=0.9,
                classifier_source="rule",
            )

        # Default: normal chat
        return IntentClassification(
            primary_intent=IntentType.NORMAL_CHAT,
            confidence=0.6,
            classifier_source="rule",
        )

    @staticmethod
    def _detect_sub_intents(message: str) -> list[IntentType]:
        segments = _MULTI_ACTION_SEPARATOR.split(message)
        intents: list[IntentType] = []
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            if any(p.search(seg) for p in _CREATE_MEMORY_PATTERNS):
                intents.append(IntentType.CREATE_MEMORY)
            elif any(p.search(seg) for p in _DELETE_PATTERNS):
                intents.append(IntentType.DELETE_MEMORY)
            elif any(p.search(seg) for p in _UPDATE_PATTERNS):
                intents.append(IntentType.UPDATE_MEMORY)
            elif any(p.search(seg) for p in _WEB_SEARCH_PATTERNS):
                intents.append(IntentType.WEB_SEARCH)
            elif any(p.search(seg) for p in _SEARCH_PATTERNS):
                intents.append(IntentType.SEARCH_MEMORY)
            else:
                intents.append(IntentType.NORMAL_CHAT)
        return intents


class LLMIntentClassifier(BaseIntentClassifier):
    """LLM-based intent classifier for ambiguous cases.

    Falls back to the LLM when rule-based classification confidence
    is below threshold.  The prompt asks the LLM to classify intent.
    """

    _CLASSIFY_PROMPT: str = (
        "You are the Aetheris Intent Classifier. "
        "Classify the user's intent from this message. "
        "Return ONLY valid JSON. No markdown, no code fences.\n\n"
        '{"intent": "NORMAL_CHAT", "confidence": 1.0, "reason": ""}\n\n'
        "Choose from:\n"
        "- NORMAL_CHAT: Casual conversation, greetings, opinions, questions about general topics.\n"
        "- CREATE_MEMORY: User wants you to remember or save something.\n"
        "- UPDATE_MEMORY: User corrects or changes previously stated information.\n"
        "- DELETE_MEMORY: User wants to forget or delete a memory.\n"
        "- MERGE_MEMORY: User wants to combine information.\n"
        "- SEARCH_MEMORY: User asks what you remember or requests stored information.\n"
        "- WEB_SEARCH: User asks for current events, news, or online information.\n"
        "- SYSTEM_QUERY: User asks about your identity, capabilities, or configuration.\n"
        "- MULTI_ACTION: User makes multiple distinct requests in one message.\n"
        "- UNKNOWN: Cannot determine intent.\n\n"
        "Rules:\n"
        "- If the message has multiple distinct requests, use MULTI_ACTION.\n"
        "- If in doubt, prefer NORMAL_CHAT over UNKNOWN.\n"
        "- confidence must be a number from 0 to 1.\n"
    )

    def __init__(self, llm_service: LLMService) -> None:
        self._llm = llm_service

    async def classify(self, message: str) -> IntentClassification:
        if not message or not message.strip():
            return IntentClassification(
                primary_intent=IntentType.UNKNOWN,
                confidence=1.0,
                classifier_source="llm",
            )

        prompt = f"{self._CLASSIFY_PROMPT}\n\nUser message:\n{message.strip()}"
        try:
            raw = await self._llm.generate_text(prompt)
            parsed = self._parse_response(raw)
            return IntentClassification(
                primary_intent=parsed.get("intent", IntentType.NORMAL_CHAT),
                confidence=parsed.get("confidence", 0.5),
                metadata={"reason": parsed.get("reason", "")},
                classifier_source="llm",
            )
        except Exception as exc:
            logger.warning("LLM intent classification failed | error=%s", exc)
            return IntentClassification(
                primary_intent=IntentType.NORMAL_CHAT,
                confidence=0.5,
                classifier_source="llm",
            )

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        first_brace = cleaned.find("{")
        if first_brace >= 0:
            cleaned = cleaned[first_brace:]
        last_brace = cleaned.rfind("}")
        if last_brace >= 0:
            cleaned = cleaned[: last_brace + 1]
        import json
        data = json.loads(cleaned)
        intent_str = data.get("intent", "NORMAL_CHAT")
        try:
            intent = IntentType(intent_str)
        except ValueError:
            intent = IntentType.NORMAL_CHAT
        return {
            "intent": intent,
            "confidence": float(data.get("confidence", 0.5)),
            "reason": str(data.get("reason", "")),
        }


class IntentClassifier(BaseIntentClassifier):
    """Combined intent classifier with rule-based fast path + LLM fallback.

    Uses rules first (fast, < 1 ms).  If confidence is below threshold
    or intent is UNKNOWN/NORMAL_CHAT with low confidence, falls back to
    the LLM classifier.  The LLM result replaces the rule result when
    its confidence is higher.
    """

    _RULE_CONFIDENCE_FLOOR: float = 0.6

    def __init__(
        self,
        llm_service: LLMService,
        rule_classifier: RuleIntentClassifier | None = None,
        llm_classifier: LLMIntentClassifier | None = None,
    ) -> None:
        self._rule = rule_classifier or RuleIntentClassifier()
        self._llm_cls = llm_classifier or LLMIntentClassifier(llm_service)

    async def classify(self, message: str) -> IntentClassification:
        rule_result = await self._rule.classify(message)

        if rule_result.confidence >= self._RULE_CONFIDENCE_FLOOR:
            return rule_result

        logger.debug(
            "Rule classifier confidence below threshold (%.2f < %.2f) "
            "— falling back to LLM classifier",
            rule_result.confidence,
            self._RULE_CONFIDENCE_FLOOR,
        )

        llm_result = await self._llm_cls.classify(message)

        if llm_result.confidence > rule_result.confidence:
            return llm_result

        return rule_result
