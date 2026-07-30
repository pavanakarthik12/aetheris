from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import Any

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class FollowUpIntent(str, Enum):
    NEW_TOPIC = "new_topic"
    CONTINUATION = "continuation"
    CLARIFICATION = "clarification"
    COMPARISON = "comparison"
    FOLLOW_UP = "follow_up"
    CORRECTION = "correction"
    REFINEMENT = "refinement"
    REFERENCE = "reference"


@dataclass
class ConversationAnalysis:
    is_follow_up: bool = False
    intent: FollowUpIntent = FollowUpIntent.NEW_TOPIC
    resolved_topic: str = ""
    confidence: float = 0.0
    previous_user_message: str = ""
    previous_assistant_response: str = ""
    matched_follow_up: str = ""
    needs_clarification: bool = False
    possible_topics: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0


_FOLLOW_UP_PHRASES: list[tuple[re.Pattern, FollowUpIntent]] = [
    (re.compile(r"^\s*(continue|go on|carry on|proceed)\s*$", re.IGNORECASE), FollowUpIntent.CONTINUATION),
    (re.compile(r"^\s*continue\s+explaining\b", re.IGNORECASE), FollowUpIntent.CONTINUATION),
    (re.compile(r"^\s*(elaborate|expand|explain\s+further|tell\s+me\s+more)\b", re.IGNORECASE), FollowUpIntent.FOLLOW_UP),
    (re.compile(r"^\s*(more|more\s+details|give\s+more\s+info)\s*$", re.IGNORECASE), FollowUpIntent.FOLLOW_UP),
    (re.compile(r"^\s*expand\s+on\s+that", re.IGNORECASE), FollowUpIntent.FOLLOW_UP),
    (re.compile(r"^\s*explain\s+that\s+part", re.IGNORECASE), FollowUpIntent.CLARIFICATION),
    (re.compile(r"^\s*can\s+you\s+explain\s+that", re.IGNORECASE), FollowUpIntent.CLARIFICATION),
    (re.compile(r"^\s*(why|how|what\s+do\s+you\s+mean)\b", re.IGNORECASE), FollowUpIntent.CLARIFICATION),
    (re.compile(r"^\s*give\s+(me\s+)?examples?\b", re.IGNORECASE), FollowUpIntent.FOLLOW_UP),
    (re.compile(r"^\s*(simplify|explain\s+simply|short\s+version|simple\s+terms)\b", re.IGNORECASE), FollowUpIntent.REFINEMENT),
    (re.compile(r"^\s*(long\s+version|in\s+detail|detailed\s+explanation)\b", re.IGNORECASE), FollowUpIntent.REFINEMENT),
    (re.compile(r"^\s*compare\b", re.IGNORECASE), FollowUpIntent.COMPARISON),
    (re.compile(r"^\s*(difference|what.s\s+the\s+difference)\b", re.IGNORECASE), FollowUpIntent.COMPARISON),
    (re.compile(r"^\s*actually,?\s+.+", re.IGNORECASE), FollowUpIntent.CORRECTION),
    (re.compile(r"^\s*no,?\s+.+", re.IGNORECASE), FollowUpIntent.CORRECTION),
    (re.compile(r"^\s*that.s\s+not\s+(right|correct|what\s+i\s+meant)", re.IGNORECASE), FollowUpIntent.CORRECTION),
    (re.compile(r"^\s*i\s+meant\b", re.IGNORECASE), FollowUpIntent.CORRECTION),
    (re.compile(r"^\s*(refer(en ce)?|about\s+that|regarding)\b", re.IGNORECASE), FollowUpIntent.REFERENCE),
]


class ConversationIntelligenceEngine:

    def __init__(self) -> None:
        self._settings = get_settings()
        self._follow_up_phrases: list[tuple[re.Pattern, FollowUpIntent]] = _FOLLOW_UP_PHRASES

    async def analyze(
        self,
        message: str,
        conversation_history: list[dict[str, Any]],
    ) -> ConversationAnalysis:
        started = perf_counter()
        analysis = ConversationAnalysis()
        cleaned = message.strip()

        if not cleaned or not conversation_history:
            elapsed = (perf_counter() - started) * 1000
            analysis.execution_time_ms = round(elapsed, 2)
            return analysis

        phrase, intent = self._detect_follow_up_phrase(cleaned)
        if phrase is None:
            elapsed = (perf_counter() - started) * 1000
            analysis.execution_time_ms = round(elapsed, 2)
            return analysis

        analysis.is_follow_up = True
        analysis.intent = intent
        analysis.matched_follow_up = phrase

        previous = self._get_previous_exchange(conversation_history)
        analysis.previous_user_message = previous["user"]
        analysis.previous_assistant_response = previous["assistant"]

        resolved_topic, confidence, possible = self._resolve_topic(
            message=cleaned,
            intent=intent,
            conversation_history=conversation_history,
            previous_user=previous["user"],
            previous_assistant=previous["assistant"],
        )
        analysis.resolved_topic = resolved_topic
        analysis.confidence = confidence
        analysis.possible_topics = possible

        threshold = self._settings.followup_confidence_threshold
        if confidence < threshold and len(possible) > 1:
            analysis.needs_clarification = True

        elapsed = (perf_counter() - started) * 1000
        analysis.execution_time_ms = round(elapsed, 2)

        logger.info(
            "ConversationIntelligence | is_follow_up=True | intent=%s | phrase=%.30s | "
            "topic=%.50s | confidence=%.2f | needs_clarification=%s | topics=%s | %.2fms",
            intent.value,
            phrase,
            resolved_topic,
            confidence,
            analysis.needs_clarification,
            possible,
            analysis.execution_time_ms,
        )
        return analysis

    def _detect_follow_up_phrase(
        self,
        message: str,
    ) -> tuple[str | None, FollowUpIntent | None]:
        for pattern, intent in self._follow_up_phrases:
            match = pattern.search(message)
            if match:
                return match.group(0).strip(), intent
        return None, None

    def _get_previous_exchange(
        self,
        history: list[dict[str, Any]],
    ) -> dict[str, str]:
        result: dict[str, str] = {"user": "", "assistant": ""}
        recent = history[-6:] if len(history) > 6 else list(history)
        last_user = ""
        last_assistant = ""
        for msg in recent:
            role = msg.get("role", "")
            content = (msg.get("content") or "").strip()
            if role == "user" and content:
                last_user = content
            elif role == "assistant" and content:
                last_assistant = content
        result["user"] = last_user
        result["assistant"] = last_assistant
        return result

    def _resolve_topic(
        self,
        message: str,
        intent: FollowUpIntent,
        conversation_history: list[dict[str, Any]],
        previous_user: str,
        previous_assistant: str,
    ) -> tuple[str, float, list[str]]:
        possible: list[str] = []
        max_distance = self._settings.max_reference_distance

        relevant = conversation_history[-(max_distance * 2):] if len(conversation_history) > max_distance * 2 else list(conversation_history)
        topics_seen: list[str] = []
        for msg in relevant:
            role = msg.get("role", "")
            content = (msg.get("content") or "").strip()
            if role == "user" and content and content not in topics_seen:
                topics_seen.append(content)

        if intent == FollowUpIntent.COMPARISON:
            other_topic = self._extract_comparison_target(message)
            if other_topic:
                if previous_user:
                    possible.append(previous_user)
                if other_topic:
                    possible.append(other_topic)
                topic = f"{previous_user} vs {other_topic}" if previous_user else other_topic
                return topic, 0.85, possible
            if previous_user:
                possible.append(previous_user)
            return previous_user, 0.75, possible

        if intent == FollowUpIntent.CORRECTION:
            if previous_assistant:
                return previous_assistant, 0.80, [previous_assistant]
            return "", 0.0, []

        if intent == FollowUpIntent.CLARIFICATION:
            if previous_assistant:
                possible.append(previous_assistant)
            if previous_user:
                possible.append(previous_user)
            if len(possible) == 1:
                return possible[0], 0.75, possible
            if len(possible) >= 2:
                return possible[0], 0.70, possible
            return "", 0.0, []

        if intent == FollowUpIntent.REFINEMENT:
            topic = previous_assistant if previous_assistant else previous_user
            return topic, 0.80, [topic] if topic else []

        if intent == FollowUpIntent.REFERENCE:
            ref_topic = self._extract_reference_topic(message)
            if ref_topic:
                return ref_topic, 0.75, [ref_topic]

        topics_seen.reverse()
        for topic in topics_seen:
            if topic not in possible:
                possible.append(topic)

        if previous_user and previous_user not in possible:
            possible.insert(0, previous_user)

        if not possible:
            return "", 0.0, []

        topic = possible[0]
        if intent == FollowUpIntent.FOLLOW_UP:
            confidence = 0.90 if len(topics_seen) <= 3 else 0.70
        elif intent == FollowUpIntent.CONTINUATION:
            confidence = 0.95 if len(topics_seen) <= 3 else 0.80
        else:
            confidence = 0.75

        comparison_intents = {FollowUpIntent.CONTINUATION, FollowUpIntent.FOLLOW_UP, FollowUpIntent.CLARIFICATION, FollowUpIntent.REFINEMENT}
        if intent in comparison_intents:
            shared_keywords = self._keyword_overlap(message, topic) if topic else 0.0
            if shared_keywords < 0.10 and len(possible) > 1:
                confidence = min(confidence, 0.50)

        return topic, confidence, possible

    def _extract_comparison_target(self, message: str) -> str:
        patterns = [
            re.compile(r"compare\s+(it\s+with|it\s+to|this\s+with|that\s+with)\s+(.+)", re.IGNORECASE),
            re.compile(r"(?:compare|difference\s+(?:between|with))\s+(.+)", re.IGNORECASE),
            re.compile(r"(?:vs|versus|or)\s+(.+)$", re.IGNORECASE),
        ]
        for pattern in patterns:
            match = pattern.search(message)
            if match:
                target = match.group(1).strip().rstrip("?.!")
                if target:
                    return target
        return ""

    def _extract_reference_topic(self, message: str) -> str:
        patterns = [
            re.compile(r"(?:about|regarding|refer(?:ring)?\s+to)\s+(.+)", re.IGNORECASE),
            re.compile(r"that\s+(topic|point|part|thing|subject)\s+(?:about|regarding)\s+(.+)", re.IGNORECASE),
        ]
        for pattern in patterns:
            match = pattern.search(message)
            if match:
                groups = [g for g in match.groups() if g]
                if groups:
                    return groups[-1].strip().rstrip("?.!")
        return ""

    @staticmethod
    def _keyword_overlap(a: str, b: str) -> float:
        import re as _re
        tokens_a = set(_re.findall(r"[a-zA-Z0-9_\-]+", a.lower()))
        tokens_b = set(_re.findall(r"[a-zA-Z0-9_\-]+", b.lower()))
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "to", "of", "in", "for", "on", "with",
            "at", "by", "from", "as", "into", "through", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "each",
            "every", "both", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than", "too",
            "very", "just", "about", "up", "what", "which", "who", "whom",
            "this", "that", "these", "those", "am", "it", "its", "my",
            "your", "his", "her", "our", "their", "me", "him", "us", "them", "i",
        }
        tokens_a -= stop_words
        tokens_b -= stop_words
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        jaccard = len(intersection) / len(union) if union else 0.0
        overlap_ratio = len(intersection) / len(tokens_a) if tokens_a else 0.0
        return jaccard * 0.5 + overlap_ratio * 0.5

    def build_follow_up_context(
        self,
        analysis: ConversationAnalysis,
    ) -> str:
        if not analysis.is_follow_up:
            return ""

        parts: list[str] = []

        if analysis.previous_user_message:
            parts.append(f"Previous Question: {analysis.previous_user_message}")
        if analysis.previous_assistant_response:
            parts.append(f"Previous Answer: {analysis.previous_assistant_response}")
        if analysis.resolved_topic:
            parts.append(f"Resolved Topic: {analysis.resolved_topic}")

        parts.append(f"Follow-up Intent: {analysis.intent.value}")
        parts.append(f"Confidence: {analysis.confidence:.2f}")

        return "\n".join(parts)

    def build_clarification_question(
        self,
        analysis: ConversationAnalysis,
    ) -> str:
        topics = analysis.possible_topics
        if not topics or len(topics) < 2:
            return ""

        quoted = [f"'{t[:60]}'" for t in topics[:3]]
        if len(quoted) == 2:
            return f"Would you like me to elaborate on {quoted[0]} or {quoted[1]}?"
        return f"Would you like me to elaborate on {', '.join(quoted[:-1])}, or {quoted[-1]}?"
