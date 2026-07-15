"""Reflection Engine — Aetheris' internal cognitive learning mechanism."""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config.settings import BASE_DIR, get_settings
from ..schemas.reflection import (
    AssistantQuality,
    ReflectionAction,
    ReflectionOutput,
    ReflectionRecord,
    ReflectionStatistics,
)
from .chroma_service import ChromaService
from .embedding_service import EmbeddingService
from .llm_service import (
    LLMQuotaExceeded,
    LLMRateLimited,
    LLMService,
    LLMServiceError,
)
from .memory_service import MemoryService
from .prompt_builder import PromptBuilder

REFLECTIONS_DIR = BASE_DIR / "backend" / "app" / "reflections"
INITIAL_STRENGTH = 0.60
STRENGTH_INCREMENT = 0.15
MAX_STRENGTH = 1.0
_MIN_STRENGTHEN_SCORE = 0.5
MIN_CONFIDENCE_FOR_ACTION = 0.4

# Rule patterns for skipping LLM reflection entirely.
_SKIP_GREETINGS = re.compile(
    r"^(hi|hey|hello|howdy|sup|yo|heya|hey there)\W*$", re.IGNORECASE
)
_SKIP_THANKS = re.compile(
    r"^(thanks?|thank you|ty|thx|appreciate it|much appreciated)\W*$", re.IGNORECASE
)
_SKIP_ACK = re.compile(
    r"^(ok|okay|k|sure|fine|alright|got it|understood|noted|roger)\W*$", re.IGNORECASE
)
_SKIP_BYE = re.compile(
    r"^(bye|goodbye|see ya|later|gn|good night|goodbye|cya|talk later)\W*$", re.IGNORECASE
)
_SKIP_FILLER = re.compile(
    r"^(nice|cool|great|awesome|amazing|perfect|excellent|"
    r"lol|lmao|haha|hehe|yes|no|yep|nah|nope|yeah|ok|sure|"
    r"interesting|good|bad|fine|whatever)\W*$",
    re.IGNORECASE,
)
# Skip short filler questions that need no reflection.
_SKIP_FILLER_QUESTIONS = re.compile(
    r"^(what'?s up|how are you|how'?s it going|what'?s new|"
    r"how'?s everything|how are things)\W*$",
    re.IGNORECASE,
)
# Time-of-day greetings.
_SKIP_TOD = re.compile(
    r"^(good )?(morning|afternoon|evening)\W*$", re.IGNORECASE
)

logger = logging.getLogger(__name__)


class ReflectionServiceError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ReflectionService:
    def __init__(
        self,
        llm_service: LLMService,
        memory_service: MemoryService,
        chroma_service: ChromaService,
        embedding_service: EmbeddingService,
    ) -> None:
        self._llm = llm_service
        self._memory_service = memory_service
        self._chroma_service = chroma_service
        self._embedding_service = embedding_service
        self._reflections_dir = REFLECTIONS_DIR
        self._reflections_dir.mkdir(parents=True, exist_ok=True)
        self._settings = get_settings()

    async def reflect(
        self,
        user_message: str,
        assistant_response: str,
    ) -> ReflectionRecord:
        started_at = datetime.now(tz=timezone.utc)
        logger.info(
            "Reflection started | message_length=%d | response_length=%d",
            len(user_message),
            len(assistant_response),
        )

        try:
            reflection = await self._decide(user_message, assistant_response)

            if reflection.confidence < MIN_CONFIDENCE_FOR_ACTION:
                logger.info(
                    "Confidence below threshold (%.2f < %.2f) — deferring actions",
                    reflection.confidence,
                    MIN_CONFIDENCE_FOR_ACTION,
                )
                reflection.actions = [ReflectionAction.NO_ACTION]

            await self._execute_actions(reflection, user_message)

            elapsed = (datetime.now(tz=timezone.utc) - started_at).total_seconds()
            record = self._save_reflection(
                user_message,
                assistant_response,
                reflection,
                processing_time_ms=round(elapsed * 1000, 2),
            )

            logger.info(
                "Reflection completed | duration_ms=%.0f | actions=%s | summary=%s",
                elapsed * 1000,
                [a.value for a in reflection.actions],
                reflection.reflection_summary,
            )
            return record
        except Exception as exc:
            elapsed = (datetime.now(tz=timezone.utc) - started_at).total_seconds()
            logger.error(
                "Reflection failed | duration_ms=%.0f | error=%s",
                elapsed * 1000,
                exc,
                exc_info=True,
            )
            return self._no_action_record(
                user_message, assistant_response,
                reason=f"Exception: {exc}",
                processing_time_ms=round(elapsed * 1000, 2),
            )

    async def reflect_with_context(
        self,
        user_message: str,
        assistant_response: str,
    ) -> None:
        try:
            await self.reflect(user_message, assistant_response)
        except Exception:
            logger.exception("Background reflection failed — continuing normally")

    async def _decide(
        self,
        user_message: str,
        assistant_response: str,
    ) -> ReflectionOutput:
        text = user_message.strip()

        # Rule-first: skip LLM for trivial interactions.
        skip_reason = self._check_rules(text, assistant_response.strip())
        if skip_reason:
            logger.info(
                "Reflection skipped | reason=%s | text=%r",
                skip_reason,
                text[:60],
            )
            return ReflectionOutput(
                actions=[ReflectionAction.NO_ACTION],
                confidence=1.0,
                reflection_summary=f"Skipped ({skip_reason}).",
            )

        if not self._settings.reflection_enabled:
            logger.info("Reflection disabled by configuration")
            return ReflectionOutput(
                actions=[ReflectionAction.NO_ACTION],
                confidence=1.0,
                reflection_summary="Reflection disabled.",
            )

        return await self._analyze_interaction(text, assistant_response.strip())

    @staticmethod
    def _check_rules(text: str, response: str) -> str | None:
        """Return a skip reason if interaction is trivially skip-worthy, else None."""
        # Both sides extremely short and one matches filler.
        if len(text) < 10 and len(response) < 10:
            if any(
                p.search(text)
                for p in (
                    _SKIP_GREETINGS,
                    _SKIP_THANKS,
                    _SKIP_ACK,
                    _SKIP_BYE,
                    _SKIP_FILLER,
                    _SKIP_FILLER_QUESTIONS,
                    _SKIP_TOD,
                )
            ):
                return "filler_short_both"
            if len(text.split()) <= 2:
                return "too_short_both"

        # Longer user side that is pure filler.
        if any(
            p.search(text)
            for p in (
                _SKIP_GREETINGS,
                _SKIP_THANKS,
                _SKIP_ACK,
                _SKIP_BYE,
                _SKIP_FILLER,
                _SKIP_FILLER_QUESTIONS,
                _SKIP_TOD,
            )
        ):
            return "filler_or_greeting"

        return None

    async def _analyze_interaction(
        self,
        user_message: str,
        assistant_response: str,
    ) -> ReflectionOutput:
        prompt = PromptBuilder.reflection(user_message, assistant_response)
        try:
            raw = await self._llm.generate_text(
                prompt,
                max_tokens=self._settings.reflection_max_tokens,
                temperature=self._settings.reflection_temperature,
            )
            parsed = self._parse_reflection(raw)
            logger.info(
                "Reflection LLM call succeeded | action=%s | confidence=%.2f",
                parsed.action.value if hasattr(parsed.action, 'value') else parsed.action,
                parsed.confidence,
            )
            return parsed
        except LLMQuotaExceeded as exc:
            logger.warning("Reflection fallback (quota) | error=%s", exc)
        except LLMRateLimited as exc:
            logger.warning("Reflection fallback (rate limit) | error=%s", exc)
        except LLMServiceError as exc:
            logger.warning("Reflection fallback (LLM error) | error=%s", exc)
        except Exception as exc:
            logger.warning("Reflection fallback (unexpected) | error=%s", exc)

        return ReflectionOutput(
            actions=[ReflectionAction.NO_ACTION],
            confidence=0.0,
            reflection_summary="Fallback (LLM unavailable).",
        )

    @staticmethod
    def _parse_reflection(raw: str) -> ReflectionOutput:
        cleaned = raw.strip()
        first_brace = cleaned.find("{")
        if first_brace >= 0:
            cleaned = cleaned[first_brace:]
        last_brace = cleaned.rfind("}")
        if last_brace >= 0:
            cleaned = cleaned[: last_brace + 1]
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Reflection parse failed | raw_response=%r", raw[:200])
            raise
        if "actions" in data and isinstance(data["actions"], list):
            data["actions"] = [ReflectionAction(a) for a in data["actions"]]
        if "action" in data and isinstance(data["action"], str):
            data["action"] = ReflectionAction(data["action"])
        return ReflectionOutput(**data)

    async def _execute_actions(
        self,
        reflection: ReflectionOutput,
        user_message: str,
    ) -> None:
        for action in reflection.actions:
            await self._execute_action(action, reflection, user_message)

    async def _execute_action(
        self,
        action: ReflectionAction,
        reflection: ReflectionOutput,
        user_message: str,
    ) -> None:
        if action == ReflectionAction.NO_ACTION:
            return

        if action not in (ReflectionAction.STRENGTHEN_MEMORY,):
            logger.info(
                "Skipping action %s — only STRENGTHEN_MEMORY is supported",
                action.value,
            )
            return

        if reflection.confidence < MIN_CONFIDENCE_FOR_ACTION:
            logger.info(
                "Deferring action %s | confidence=%.2f below threshold=%.2f",
                action.value,
                reflection.confidence,
                MIN_CONFIDENCE_FOR_ACTION,
            )
            return

        logger.info("Executing action | action=%s", action.value)

        if action == ReflectionAction.STRENGTHEN_MEMORY:
            await self._apply_strengthen_memory(user_message)

    async def _apply_strengthen_memory(self, user_message: str) -> None:
        related = await self._memory_service.search_memory(query=user_message, top_k=5)
        for mem in related:
            if mem.get("score", 0) < _MIN_STRENGTHEN_SCORE:
                continue
            meta = dict(mem.get("metadata", {}))
            current = meta.get("memory_strength", INITIAL_STRENGTH)
            updated = min(current + STRENGTH_INCREMENT, MAX_STRENGTH)
            meta["memory_strength"] = updated
            meta["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
            meta["last_strengthened_at"] = meta["updated_at"]
            await self._update_memory_metadata(mem["id"], meta)
            logger.info(
                "Memory strengthened | id=%s | from=%.2f | to=%.2f | score=%.2f",
                mem["id"],
                current,
                updated,
                mem.get("score", 0),
            )

    async def _update_memory_metadata(
        self,
        memory_id: str,
        metadata: dict[str, Any],
    ) -> None:
        try:
            existing = self._chroma_service.get_memory_by_id(memory_id)
            if existing is None:
                logger.warning(
                    "Memory not found for metadata update | id=%s", memory_id
                )
                return
            new_embedding = await self._embedding_service.embed_text(existing["document"])
            self._chroma_service.update_memory(
                memory_id=memory_id,
                embedding=new_embedding,
                document=existing["document"],
                metadata=metadata,
            )
        except Exception as exc:
            logger.error(
                "Failed to update memory metadata | id=%s | error=%s",
                memory_id,
                exc,
            )

    def _no_action_record(
        self,
        user_message: str,
        assistant_response: str,
        reason: str = "",
        processing_time_ms: float = 0.0,
    ) -> ReflectionRecord:
        reflection = ReflectionOutput(
            actions=[ReflectionAction.NO_ACTION],
            confidence=0.0,
            reflection_summary=reason or "Fallback.",
        )
        return self._save_reflection(
            user_message,
            assistant_response,
            reflection,
            processing_time_ms=processing_time_ms,
        )

    def _save_reflection(
        self,
        user_message: str,
        assistant_response: str,
        reflection: ReflectionOutput,
        processing_time_ms: float = 0.0,
    ) -> ReflectionRecord:
        record = ReflectionRecord(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            user_message=user_message,
            assistant_response=assistant_response,
            reflection=reflection,
            processing_time_ms=processing_time_ms,
        )
        file_path = self._reflections_dir / f"{record.id}.json"
        file_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Reflection saved | id=%s", record.id)
        return record

    def list_reflections(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for fpath in sorted(self._reflections_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                records.append(data)
            except Exception as exc:
                logger.warning(
                    "Failed to read reflection | path=%s | error=%s", fpath, exc,
                )
        return records

    def get_reflection(self, reflection_id: str) -> dict[str, Any] | None:
        fpath = self._reflections_dir / f"{reflection_id}.json"
        if not fpath.exists():
            return None
        try:
            return json.loads(fpath.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "Failed to read reflection | id=%s | error=%s", reflection_id, exc,
            )
            return None

    def statistics(self) -> ReflectionStatistics:
        records = self.list_reflections()
        total = len(records)
        if total == 0:
            return ReflectionStatistics(
                total_reflections=0,
                most_common_action="NO_ACTION",
                action_counts={"NO_ACTION": 0},
                average_confidence=0.0,
                average_quality=AssistantQuality(),
                average_processing_time_ms=0.0,
            )

        action_counts: dict[str, int] = {}
        total_confidence = 0.0
        total_quality_correctness = 0.0
        total_quality_completeness = 0.0
        total_quality_relevance = 0.0
        total_quality_clarity = 0.0
        total_quality_confidence = 0.0
        total_quality_hallucination_risk = 0.0
        total_processing_time = 0.0

        for rec in records:
            ref = rec.get("reflection", {})
            actions = ref.get("actions", ["NO_ACTION"])
            for a in actions:
                action_counts[a] = action_counts.get(a, 0) + 1

            total_confidence += ref.get("confidence", 1.0)
            total_processing_time += rec.get("processing_time_ms", 0.0)

            aq = ref.get("assistant_quality", {})
            total_quality_correctness += aq.get("correctness", 1.0)
            total_quality_completeness += aq.get("completeness", 1.0)
            total_quality_relevance += aq.get("relevance", 1.0)
            total_quality_clarity += aq.get("clarity", 1.0)
            total_quality_confidence += aq.get("confidence", 1.0)
            total_quality_hallucination_risk += aq.get("hallucination_risk", 0.0)

        most_common = max(action_counts, key=action_counts.get)

        avg_conf = round(total_confidence / total, 4)
        avg_pt = round(total_processing_time / total, 2)

        avg_q = AssistantQuality(
            correctness=round(total_quality_correctness / total, 4),
            completeness=round(total_quality_completeness / total, 4),
            relevance=round(total_quality_relevance / total, 4),
            clarity=round(total_quality_clarity / total, 4),
            confidence=round(total_quality_confidence / total, 4),
            hallucination_risk=round(total_quality_hallucination_risk / total, 4),
        )

        return ReflectionStatistics(
            total_reflections=total,
            most_common_action=most_common,
            action_counts=action_counts,
            average_confidence=avg_conf,
            average_quality=avg_q,
            average_processing_time_ms=avg_pt,
        )

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        records = self.list_reflections()
        return records[:limit]
