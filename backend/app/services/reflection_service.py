"""Reflection Engine — Aetheris' internal cognitive learning mechanism.

Runs asynchronously after every completed conversation to analyze the
interaction, extract lessons, and update the memory system accordingly.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config.settings import BASE_DIR
from ..schemas.reflection import ReflectionAction, ReflectionOutput, ReflectionRecord
from .chroma_service import ChromaService
from .embedding_service import EmbeddingService
from .llm_service import LLMService
from .memory_service import MemoryService

REFLECTIONS_DIR = BASE_DIR / "backend" / "app" / "reflections"
INITIAL_STRENGTH = 0.60
STRENGTH_INCREMENT = 0.15
MAX_STRENGTH = 1.0
_MIN_STRENGTHEN_SCORE = 0.5

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
            await self._execute_actions(reflection, user_message)
            record = self._save_reflection(user_message, assistant_response, reflection)

            elapsed = (datetime.now(tz=timezone.utc) - started_at).total_seconds()
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
            "1. Did the user reveal new long-term information?\n"
            "2. Should an existing memory be updated?\n"
            "3. Did this interaction strengthen an existing memory?\n"
            "4. Did the assistant misunderstand the user?\n"
            "5. Did the assistant answer with low confidence?\n"
            "6. Did the user correct the assistant?\n"
            "7. Should future responses change because of this interaction?\n\n"
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
            '  "actions": ["NO_ACTION"]\n'
            '}\n\n'
            "Actions must be from: CREATE_MEMORY, UPDATE_MEMORY, "
            "MERGE_MEMORY, STRENGTHEN_MEMORY, NO_ACTION"
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

        logger.info("Executing action | action=%s", action.value)

        if action == ReflectionAction.STRENGTHEN_MEMORY:
            await self._apply_strengthen_memory(user_message)
        elif action == ReflectionAction.CREATE_MEMORY:
            await self._apply_create_memory(reflection)
        elif action == ReflectionAction.UPDATE_MEMORY:
            await self._apply_update_memory(user_message)
        elif action == ReflectionAction.MERGE_MEMORY:
            await self._apply_merge_memory(reflection)

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

    async def _apply_create_memory(self, reflection: ReflectionOutput) -> None:
        await self._memory_service.save_memory(
            memory_text=reflection.reflection_summary,
            metadata={
                "source": "reflection",
                "memory_strength": INITIAL_STRENGTH,
                "importance": reflection.confidence,
                "category": "Reflection",
            },
        )

    async def _apply_update_memory(self, user_message: str) -> None:
        related = await self._memory_service.search_memory(query=user_message, top_k=1)
        if not related:
            return
        mem = related[0]
        meta = dict(mem.get("metadata", {}))
        meta["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
        meta["reflection_updated"] = True
        new_document = f"{mem['document']}\n{user_message}"
        try:
            new_embedding = await self._embedding_service.embed_text(new_document)
            self._chroma_service.update_memory(
                memory_id=mem["id"],
                embedding=new_embedding,
                document=new_document,
                metadata=meta,
            )
            logger.info("Memory updated via reflection | id=%s", mem["id"])
        except Exception as exc:
            logger.error(
                "Failed to update memory document | id=%s | error=%s",
                mem["id"], exc,
            )

    async def _apply_merge_memory(self, reflection: ReflectionOutput) -> None:
        related = await self._memory_service.search_memory(
            query=reflection.reflection_summary, top_k=2,
        )
        if len(related) < 2:
            return
        target = related[0]
        source = related[1]
        source_meta = dict(source.get("metadata", {}))
        source_meta["status"] = "archived"
        source_meta["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
        merged_text = f"{source['document']}\n{target['document']}"
        meta = dict(target.get("metadata", {}))
        meta["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
        meta["memory_strength"] = max(
            meta.get("memory_strength", INITIAL_STRENGTH),
            source.get("metadata", {}).get("memory_strength", INITIAL_STRENGTH),
        )
        try:
            merged_embedding = await self._embedding_service.embed_text(merged_text)
            source_embedding = await self._embedding_service.embed_text(source["document"])
            self._chroma_service.update_memory(
                memory_id=source["id"],
                embedding=source_embedding,
                document=source["document"],
                metadata=source_meta,
            )
            self._chroma_service.update_memory(
                memory_id=target["id"],
                embedding=merged_embedding,
                document=merged_text,
                metadata=meta,
            )
            logger.info(
                "Memory merged via reflection | target=%s | source=%s archived",
                target["id"], source["id"],
            )
        except Exception as exc:
            logger.error("Failed to merge memories | error=%s", exc)

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
    ) -> ReflectionRecord:
        record = ReflectionRecord(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            user_message=user_message,
            assistant_response=assistant_response,
            reflection=reflection,
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
