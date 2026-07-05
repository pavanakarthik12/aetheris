"""Chat router — Phase 4 Memory Retrieval & Context Injection pipeline.

Request flow:
    POST /api/chat
        │
        ├─ 1. Search ChromaDB for memories from PREVIOUS turns
        │
        ├─ 2. Save the current message to ChromaDB (available next turn)
        │
        ├─ 3. Build memory context block via ContextBuilderService
        │
        ├─ 4. Assemble system prompt + memory context + user message
        │
        └─ 5. Call Qwen via LLMService.generate_with_context()
                │
                └─ Return ChatResponse(response=..., memory_count=...)
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from ..dependencies import (
    get_context_builder_service,
    get_llm_service,
    get_memory_evaluator_service,
    get_memory_evolution_service,
    get_memory_service,
    get_reflection_service,
)
from ..schemas.chat import ChatRequest, ChatResponse
from ..services.context_builder import ContextBuilderService
from ..services.llm_service import LLMService, LLMServiceError
from ..services.memory_evaluator import MemoryEvaluation, MemoryEvaluatorService, MemoryEvaluatorServiceError
from ..services.memory_evolution_service import MemoryEvolutionService, MemoryEvolutionServiceError
from ..services.memory_service import MemoryService
from ..services.reflection_service import ReflectionService

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)

# Top-k memories to retrieve per request.
_MEMORY_TOP_K: int = 5


@router.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    llm_service: LLMService = Depends(get_llm_service),
    memory_service: MemoryService = Depends(get_memory_service),
    memory_evaluator_service: MemoryEvaluatorService = Depends(get_memory_evaluator_service),
    memory_evolution_service: MemoryEvolutionService = Depends(get_memory_evolution_service),
    context_builder: ContextBuilderService = Depends(get_context_builder_service),
    reflection_service: ReflectionService = Depends(get_reflection_service),
) -> ChatResponse:
    """Retrieve relevant memories, inject them into the prompt, call Qwen.

    Memory retrieval failures are non-fatal: if ChromaDB or the embedding
    service is unavailable the endpoint continues without context and the
    LLM answers from its general knowledge.
    """
    logger.info(
        "Chat request received | message_length=%d", len(request.message)
    )
    total_started = perf_counter()

    # ------------------------------------------------------------------
    # Step 1 — Search existing memories BEFORE saving the current message.
    # This ensures only previously stored memories surface as context —
    # not the message we're about to process.
    # ------------------------------------------------------------------
    memories: list[dict] = []
    retrieval_ms: float = 0.0

    retrieval_started = perf_counter()
    try:
        memories = await memory_service.search_memory(
            query=request.message,
            top_k=_MEMORY_TOP_K,
        )
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
        logger.info(
            "Memory retrieval complete | count=%d | duration_ms=%.2f",
            len(memories),
            retrieval_ms,
        )
    except Exception as mem_exc:
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
        logger.warning(
            "Memory retrieval failed — continuing without context | reason=%s | duration_ms=%.2f",
            mem_exc,
            retrieval_ms,
            exc_info=True,
        )

    # ------------------------------------------------------------------
    # Step 2 — Evaluate the current message before attempting to store it.
    # If the evaluator rejects the message, we skip persistence entirely.
    # ------------------------------------------------------------------
    evaluation: MemoryEvaluation | None = None
    try:
        evaluation = await memory_evaluator_service.evaluate_memory(request.message)
        logger.info(
            "Memory evaluation complete | store=%s | category=%s | importance=%.2f | reason=%s",
            evaluation.store,
            evaluation.category,
            evaluation.importance,
            evaluation.reason,
        )
    except MemoryEvaluatorServiceError as eval_exc:
        logger.warning(
            "Memory evaluation failed — continuing without storage | reason=%s",
            eval_exc,
            exc_info=True,
        )

    if evaluation is not None and evaluation.store:
        base_metadata: dict[str, Any] = {
            "source": "chat",
            "category": evaluation.category,
            "importance": evaluation.importance,
            "reason": evaluation.reason,
        }

        try:
            decision = await memory_evolution_service.decide_evolution(
                memory_text=request.message,
                existing_evaluation={
                    "store": evaluation.store,
                    "category": evaluation.category,
                    "importance": evaluation.importance,
                    "reason": evaluation.reason,
                },
            )

            action = decision["action"]

            if action == "CREATE":
                result = await memory_evolution_service.create_memory(
                    memory_text=request.message,
                    metadata=base_metadata,
                )
                logger.info(
                    "Memory CREATED | id=%s | category=%s | importance=%.2f",
                    result["memory_id"], evaluation.category, evaluation.importance,
                )

            elif action == "UPDATE":
                target_id = decision.get("target_id")
                if target_id:
                    result = await memory_evolution_service.update_memory(
                        memory_id=target_id,
                        new_text=request.message,
                        new_metadata=base_metadata,
                    )
                    logger.info(
                        "Memory UPDATED | id=%s | version=%d | category=%s",
                        result["memory_id"], result["version"], evaluation.category,
                    )

            elif action == "MERGE":
                target_id = decision.get("target_id")
                if target_id:
                    existing_text = await memory_evolution_service.get_memory_document(target_id) or ""
                    merged_text = f"{existing_text}\n{request.message}"
                    result = await memory_evolution_service.update_memory(
                        memory_id=target_id,
                        new_text=merged_text,
                        new_metadata=base_metadata,
                    )
                    logger.info(
                        "Memory MERGED | target=%s | version=%d | category=%s",
                        result["memory_id"], result["version"], evaluation.category,
                    )

            else:
                logger.info(
                    "Memory evolution skipped | action=%s | reason=%s",
                    action, decision.get("explanation", ""),
                )

        except MemoryEvolutionServiceError as evo_exc:
            logger.warning(
                "Memory evolution failed — continuing | reason=%s",
                evo_exc,
                exc_info=True,
            )
        except Exception as save_exc:
            logger.warning(
                "Failed to save user message as memory — continuing | reason=%s",
                save_exc,
                exc_info=True,
            )

    elif evaluation is not None:
        logger.info(
            "Memory storage skipped | store=%s | category=%s | importance=%.2f | reason=%s",
            evaluation.store,
            evaluation.category,
            evaluation.importance,
            evaluation.reason,
        )

    # ------------------------------------------------------------------
    # Step 3 — Build context block from retrieved memories
    # ------------------------------------------------------------------
    memory_context: str = context_builder.build_memory_context(memories)

    # Each surviving memory renders as exactly one "- " bullet line after the header.
    # Counting lines that start with "- " gives an exact injected_count with no
    # string-scanning fragility.
    injected_count: int = sum(
        1 for line in memory_context.splitlines() if line.startswith("- ")
    )

    if memory_context:
        logger.info(
            "Memory context built | injected_memories=%d", injected_count
        )
    else:
        logger.info("No memory context — responding from general knowledge")

    # ------------------------------------------------------------------
    # Step 4 — Assemble system prompt
    # ------------------------------------------------------------------
    system_prompt: str = context_builder.build_system_prompt()

    # ------------------------------------------------------------------
    # Step 5 — Call LLM with full context
    # ------------------------------------------------------------------
    llm_started = perf_counter()

    # Log the exact user-turn content being sent so memory injection is
    # visible in the server logs without needing a debugger.
    if memory_context:
        logger.info(
            "Prompt user-turn preview:\n%s\n\nCurrent User Message:\n%s",
            memory_context,
            request.message,
        )
    else:
        logger.info("Prompt user-turn (no memory): %s", request.message)

    try:
        response_text = await llm_service.generate_with_context(
            user_message=request.message,
            system_prompt=system_prompt,
            memory_context=memory_context,
        )
    except LLMServiceError as exc:
        logger.exception("LLM call failed")
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    llm_ms = (perf_counter() - llm_started) * 1000
    total_ms = (perf_counter() - total_started) * 1000

    logger.info(
        "Chat response ready | memory_retrieval_ms=%.2f | llm_ms=%.2f | total_ms=%.2f | injected=%d",
        retrieval_ms,
        llm_ms,
        total_ms,
        injected_count,
    )

    background_tasks.add_task(
        reflection_service.reflect_with_context,
        request.message,
        response_text,
    )

    return ChatResponse(response=response_text, memory_count=injected_count)
