"""Immediate Memory Processor — synchronous memory mutations before LLM response.

Runs BEFORE the LLM generates a response to ensure all memory writes are
committed before retrieval occurs.  This eliminates the race condition where
the next user message might see stale data because reflection hadn't finished
writing yet.

Responsibilities (all execute synchronously before the LLM call):
  CREATE, UPDATE, MERGE, DELETE, ARCHIVE

This service never calls the LLM.  It delegates to:
  - MemoryEvaluatorService  (decides whether the message is worth storing)
  - MemoryEvolutionService   (decides CREATE / UPDATE / MERGE)
  - MemoryService             (direct delete / archive for explicit user requests)
"""

from __future__ import annotations

import logging
from typing import Any

from ..schemas.chat import ImmediateMemoryResult, MemoryActionType
from .chroma_service import ChromaService
from .embedding_service import EmbeddingService
from .memory_evaluator import MemoryEvaluation, MemoryEvaluatorService, MemoryEvaluatorServiceError
from .memory_evolution_service import MemoryEvolutionService, MemoryEvolutionServiceError
from .memory_normalizer import normalize_to_third_person
from .memory_service import MemoryService

logger = logging.getLogger(__name__)

_DELETE_TRIGGERS: tuple[str, ...] = (
    "forget", "delete", "remove", "erase",
)
_ARCHIVE_TRIGGERS: tuple[str, ...] = (
    "archive", "stash",
)


class ImmediateMemoryProcessorError(RuntimeError):
    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class ImmediateMemoryProcessor:
    """Process a user message and apply memory mutations immediately.

    Pipeline for each message:
        1. Check for explicit delete/archive user intent.
        2. Evaluate memory-worthiness via MemoryEvaluatorService.
        3. Decide evolution action via MemoryEvolutionService.
        4. Execute the chosen action (CREATE / UPDATE / MERGE / DELETE / ARCHIVE).
        5. Return a result summarising what happened.
    """

    def __init__(
        self,
        memory_service: MemoryService,
        memory_evaluator: MemoryEvaluatorService,
        memory_evolution: MemoryEvolutionService,
        chroma_service: ChromaService,
        embedding_service: EmbeddingService,
    ) -> None:
        self._memory_service = memory_service
        self._memory_evaluator = memory_evaluator
        self._memory_evolution = memory_evolution
        self._chroma_service = chroma_service
        self._embedding_service = embedding_service

    async def process_message(
        self,
        message_text: str,
    ) -> ImmediateMemoryResult:
        """Run the full immediate memory pipeline.

        Args:
            message_text: The raw user message.

        Returns:
            ImmediateMemoryResult with the action taken and any error details.
        """
        if not message_text or not message_text.strip():
            return ImmediateMemoryResult(
                action=MemoryActionType.SKIP,
                success=True,
                memory_id=None,
                error=None,
            )

        try:
            result = await self._process(message_text)
            return result
        except ImmediateMemoryProcessorError as exc:
            logger.error("Immediate memory processor failed | error=%s", exc)
            return ImmediateMemoryResult(
                action=MemoryActionType.ERROR,
                success=False,
                memory_id=None,
                error=str(exc),
            )
        except Exception as exc:
            logger.exception("Immediate memory processor unexpected error | error=%s", exc)
            return ImmediateMemoryResult(
                action=MemoryActionType.ERROR,
                success=False,
                memory_id=None,
                error=f"Unexpected error: {exc}",
            )

    async def _process(self, message_text: str) -> ImmediateMemoryResult:
        # --------------------------------------------------------------
        # 1. Check for explicit delete / forget commands
        # --------------------------------------------------------------
        delete_result = await self._handle_explicit_delete(message_text)
        if delete_result is not None:
            return delete_result

        # --------------------------------------------------------------
        # 2. Check for explicit archive commands
        # --------------------------------------------------------------
        archive_result = await self._handle_explicit_archive(message_text)
        if archive_result is not None:
            return archive_result

        # --------------------------------------------------------------
        # 3. Evaluate memory-worthiness
        # --------------------------------------------------------------
        try:
            evaluation: MemoryEvaluation = await self._memory_evaluator.evaluate_memory(
                message_text,
            )
        except MemoryEvaluatorServiceError as exc:
            logger.warning("Memory evaluation failed | error=%s", exc)
            return ImmediateMemoryResult(
                action=MemoryActionType.SKIP,
                success=True,
                memory_id=None,
                error=f"Evaluation failed: {exc}",
            )

        if not evaluation.store:
            return ImmediateMemoryResult(
                action=MemoryActionType.SKIP,
                success=True,
                memory_id=None,
                error=None,
            )

        # --------------------------------------------------------------
        # 4. Build third-person storage text and metadata
        # --------------------------------------------------------------
        storage_text = evaluation.memory_fact.strip()
        if not storage_text:
            storage_text = normalize_to_third_person(
                message_text, category=evaluation.category,
            )

        base_metadata: dict[str, Any] = {
            "source": "chat",
            "category": evaluation.category,
            "importance": evaluation.importance,
            "reason": evaluation.reason,
            "original_text": message_text.strip(),
            "memory_fact": storage_text,
        }

        if evaluation.attribute and evaluation.value:
            base_metadata["attribute"] = evaluation.attribute
            base_metadata["value"] = evaluation.value
            base_metadata["fact"] = {
                "category": evaluation.category.lower() if evaluation.category else "",
                "attribute": evaluation.attribute,
                "value": evaluation.value,
                "memory_fact": storage_text,
            }

        # --------------------------------------------------------------
        # 5. Decide evolution action (uses original message for search)
        # --------------------------------------------------------------
        try:
            decision = await self._memory_evolution.decide_evolution(
                memory_text=message_text,
                existing_evaluation={
                    "store": evaluation.store,
                    "category": evaluation.category,
                    "importance": evaluation.importance,
                    "reason": evaluation.reason,
                },
            )
        except MemoryEvolutionServiceError as exc:
            logger.warning("Memory evolution decision failed | error=%s", exc)
            return ImmediateMemoryResult(
                action=MemoryActionType.ERROR,
                success=False,
                memory_id=None,
                error=f"Evolution decision failed: {exc}",
            )

        # --------------------------------------------------------------
        # 6. Execute the chosen action (uses normalized text for storage)
        # --------------------------------------------------------------
        action = decision.get("action", "SKIP")

        try:
            if action == "CREATE":
                result = await self._memory_evolution.create_memory(
                    memory_text=storage_text,
                    metadata=base_metadata,
                )
                logger.info(
                    "Memory CREATED via immediate processor | id=%s | "
                    "attribute=%s | value=%s | fact=%.60r",
                    result["memory_id"],
                    evaluation.attribute or "unknown",
                    evaluation.value or "unknown",
                    storage_text,
                )
                return ImmediateMemoryResult(
                    action=MemoryActionType.CREATE,
                    success=True,
                    memory_id=result.get("memory_id"),
                    error=None,
                )

            elif action == "UPDATE":
                target_id = decision.get("target_id")
                if not target_id:
                    return ImmediateMemoryResult(
                        action=MemoryActionType.ERROR,
                        success=False,
                        memory_id=None,
                        error="UPDATE decision missing target_id",
                    )
                result = await self._memory_evolution.update_memory(
                    memory_id=target_id,
                    new_text=storage_text,
                    new_metadata=base_metadata,
                )
                logger.info(
                    "Memory UPDATED via immediate processor | id=%s | "
                    "attribute=%s | value=%s | fact=%.60r",
                    result["memory_id"],
                    evaluation.attribute or "unknown",
                    evaluation.value or "unknown",
                    storage_text,
                )
                return ImmediateMemoryResult(
                    action=MemoryActionType.UPDATE,
                    success=True,
                    memory_id=result.get("memory_id"),
                    error=None,
                )

            elif action == "MERGE":
                target_id = decision.get("target_id")
                if not target_id:
                    return ImmediateMemoryResult(
                        action=MemoryActionType.ERROR,
                        success=False,
                        memory_id=None,
                        error="MERGE decision missing target_id",
                    )
                existing_text = await self._memory_evolution.get_memory_document(target_id) or ""
                merged_text = f"{existing_text}\n{storage_text}"
                result = await self._memory_evolution.update_memory(
                    memory_id=target_id,
                    new_text=merged_text,
                    new_metadata=base_metadata,
                )
                logger.info("Memory MERGED via immediate processor | target=%s", result["memory_id"])
                return ImmediateMemoryResult(
                    action=MemoryActionType.MERGE,
                    success=True,
                    memory_id=result.get("memory_id"),
                    error=None,
                )

            else:
                logger.info("Memory evolution returned action=%s — skipping", action)
                return ImmediateMemoryResult(
                    action=MemoryActionType.SKIP,
                    success=True,
                    memory_id=None,
                    error=None,
                )

        except (MemoryEvolutionServiceError, Exception) as exc:
            logger.error("Failed to execute evolution action=%s | error=%s", action, exc)
            return ImmediateMemoryResult(
                action=MemoryActionType.ERROR,
                success=False,
                memory_id=None,
                error=f"Failed to execute {action}: {exc}",
            )

    # ------------------------------------------------------------------
    # Explicit delete / forget handling
    # ------------------------------------------------------------------
    async def _handle_explicit_delete(
        self,
        message_text: str,
    ) -> ImmediateMemoryResult | None:
        """Check if the user is asking to delete/forget a memory.

        Returns an ImmediateMemoryResult if a delete was performed,
        or None if no delete intent was detected.
        """
        lower = message_text.lower()
        if not any(trigger in lower for trigger in _DELETE_TRIGGERS):
            return None

        related = await self._memory_service.search_memory(query=message_text, top_k=1)
        if not related:
            return ImmediateMemoryResult(
                action=MemoryActionType.SKIP,
                success=True,
                memory_id=None,
                error=None,
            )

        mem = related[0]
        score = mem.get("score", 0.0)
        if score < 0.5:
            return ImmediateMemoryResult(
                action=MemoryActionType.SKIP,
                success=True,
                memory_id=None,
                error=None,
            )

        try:
            self._memory_service.delete_memory(mem["id"])
            logger.info("Memory DELETED via immediate processor | id=%s", mem["id"])
            return ImmediateMemoryResult(
                action=MemoryActionType.DELETE,
                success=True,
                memory_id=mem["id"],
                error=None,
            )
        except Exception as exc:
            logger.error("Failed to delete memory | id=%s | error=%s", mem["id"], exc)
            return ImmediateMemoryResult(
                action=MemoryActionType.ERROR,
                success=False,
                memory_id=mem["id"],
                error=f"Delete failed: {exc}",
            )

    # ------------------------------------------------------------------
    # Explicit archive handling
    # ------------------------------------------------------------------
    async def _handle_explicit_archive(
        self,
        message_text: str,
    ) -> ImmediateMemoryResult | None:
        """Check if the user is asking to archive a memory."""
        lower = message_text.lower()
        if not any(trigger in lower for trigger in _ARCHIVE_TRIGGERS):
            return None

        related = await self._memory_service.search_memory(query=message_text, top_k=1)
        if not related:
            return ImmediateMemoryResult(
                action=MemoryActionType.SKIP,
                success=True,
                memory_id=None,
                error=None,
            )

        mem = related[0]
        score = mem.get("score", 0.0)
        if score < 0.5:
            return ImmediateMemoryResult(
                action=MemoryActionType.SKIP,
                success=True,
                memory_id=None,
                error=None,
            )

        try:
            await self._memory_evolution.archive_memory(mem["id"])
            logger.info("Memory ARCHIVED via immediate processor | id=%s", mem["id"])
            return ImmediateMemoryResult(
                action=MemoryActionType.ARCHIVE,
                success=True,
                memory_id=mem["id"],
                error=None,
            )
        except Exception as exc:
            logger.error("Failed to archive memory | id=%s | error=%s", mem["id"], exc)
            return ImmediateMemoryResult(
                action=MemoryActionType.ERROR,
                success=False,
                memory_id=mem["id"],
                error=f"Archive failed: {exc}",
            )
