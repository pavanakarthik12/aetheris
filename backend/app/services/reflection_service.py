"""Reflection Engine — Aetheris' internal cognitive learning mechanism.

Runs asynchronously AFTER the user has received a response.  The Reflection
Engine never directly modifies critical user memories.  Memory mutations
(CREATE, UPDATE, MERGE, DELETE, ARCHIVE) are handled by the
ImmediateMemoryProcessor *before* the LLM generates a response.

Reflection is limited to:
  - Self-evaluation and assistant quality analysis
  - Confidence analysis
  - Memory STRENGTHENING (the only metadata-level mutation allowed)
  - Consistency analysis (read-only detection, no automatic weakening)
  - Reflection history and internal learning metrics
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config.settings import BASE_DIR
from ..schemas.reflection import (
    AssistantQuality,
    ReflectionAction,
    ReflectionOutput,
    ReflectionRecord,
    ReflectionStatistics,
)
from .chroma_service import ChromaService
from .embedding_service import EmbeddingService
from .llm_service import LLMService
from .memory_service import MemoryService

REFLECTIONS_DIR = BASE_DIR / "backend" / "app" / "reflections"
INITIAL_STRENGTH = 0.60
STRENGTH_INCREMENT = 0.15
MAX_STRENGTH = 1.0
_MIN_STRENGTHEN_SCORE = 0.5
MIN_CONFIDENCE_FOR_ACTION = 0.4

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
            reflection = await self._analyze_interaction(user_message, assistant_response)

            if reflection.confidence < MIN_CONFIDENCE_FOR_ACTION:
                logger.info(
                    "Confidence below threshold (%.2f < %.2f) — deferring actions",
                    reflection.confidence,
                    MIN_CONFIDENCE_FOR_ACTION,
                )
                reflection.actions = [ReflectionAction.NO_ACTION]

            await self._execute_actions(reflection, user_message)

            elapsed = (datetime.now(tz=timezone.utc) - started_at).total_seconds()
            elapsed_ms = round(elapsed * 1000, 2)
            record = self._save_reflection(
                user_message, assistant_response, reflection, processing_time_ms=elapsed_ms,
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
            raise ReflectionServiceError(str(exc))

    async def reflect_with_context(
        self,
        user_message: str,
        assistant_response: str,
    ) -> None:
        try:
            await self.reflect(user_message, assistant_response)
        except ReflectionServiceError:
            logger.exception("Background reflection failed — continuing normally")

    async def _analyze_interaction(
        self,
        user_message: str,
        assistant_response: str,
    ) -> ReflectionOutput:
        prompt = self._build_reflection_prompt(user_message, assistant_response)
        try:
            raw = await self._llm.generate_text(prompt)
            parsed = self._parse_reflection(raw)
            return parsed
        except Exception as exc:
            logger.warning(
                "Reflection analysis failed, defaulting to NO_ACTION | error=%s", exc,
            )
            return ReflectionOutput()

    def _build_reflection_prompt(
        self,
        user_message: str,
        assistant_response: str,
    ) -> str:
        return (
            "You are Aetheris' Cognitive Reflection Engine. "
            "Analyze the completed interaction below.\n\n"
            f"User Message:\n{user_message}\n\n"
            f"Assistant Response:\n{assistant_response}\n\n"
            "Answer these questions:\n"
            "1. Did this interaction strengthen an existing memory?\n"
            "2. Did the assistant misunderstand the user?\n"
            "3. Did the user correct the assistant?\n"
            "4. Should future responses change because of this interaction?\n\n"
            "Now analyze from these additional perspectives:\n"
            "5. Consistency: Does the new information contradict any existing memory?\n"
            "6. Assistant Quality: Self-evaluate your response — rate correctness, "
            "completeness, relevance, clarity, confidence (0–1), and hallucination_risk (0–1).\n"
            "7. Reasoning: Why did you choose the above action?\n"
            "8. Manual Review: Does this case require human review?\n\n"
            "Return ONLY valid JSON in this exact format:\n"
            '{\n'
            '  "new_memory": false,\n'
            '  "update_memory": false,\n'
            '  "memory_strengthened": false,\n'
            '  "assistant_mistake": false,\n'
            '  "user_corrected_ai": false,\n'
            '  "confidence": 0.95,\n'
            '  "future_behavior_change": false,\n'
            '  "reflection_summary": "...",\n'
            '  "actions": ["NO_ACTION"],\n'
            '  "action": "NO_ACTION",\n'
            '  "consistency": true,\n'
            '  "assistant_quality": {\n'
            '    "correctness": 1.0,\n'
            '    "completeness": 1.0,\n'
            '    "relevance": 1.0,\n'
            '    "clarity": 1.0,\n'
            '    "confidence": 1.0,\n'
            '    "hallucination_risk": 0.0\n'
            '  },\n'
            '  "user_preferences_detected": [],\n'
            '  "reasoning": "...",\n'
            '  "requires_manual_review": false\n'
            '}\n\n'
            "Actions must be from: NO_ACTION, STRENGTHEN_MEMORY"
        )

    @staticmethod
    def _parse_reflection(raw: str) -> ReflectionOutput:
        cleaned = raw.strip()
        first_brace = cleaned.find("{")
        if first_brace >= 0:
            cleaned = cleaned[first_brace:]
        last_brace = cleaned.rfind("}")
        if last_brace >= 0:
            cleaned = cleaned[:last_brace + 1]
        data = json.loads(cleaned)
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
                "Skipping action %s — only STRENGTHEN_MEMORY is supported by Reflection Engine",
                action.value,
            )
            return

        if reflection.confidence < MIN_CONFIDENCE_FOR_ACTION:
            logger.info(
                "Deferring action %s | confidence=%.2f below threshold=%.2f",
                action.value, reflection.confidence, MIN_CONFIDENCE_FOR_ACTION,
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
                mem["id"], current, updated, mem.get("score", 0),
            )

    async def _update_memory_metadata(
        self,
        memory_id: str,
        metadata: dict[str, Any],
    ) -> None:
        try:
            existing = self._chroma_service.get_memory_by_id(memory_id)
            if existing is None:
                logger.warning("Memory not found for metadata update | id=%s", memory_id)
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
                memory_id, exc,
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
