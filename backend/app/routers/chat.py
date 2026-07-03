"""Chat router — Phase 4 Memory Retrieval & Context Injection pipeline.

Request flow:
    POST /api/chat
        │
        ├─ 1. Save user message to ChromaDB (auto-memory accumulation)
        │
        ├─ 2. Search ChromaDB for relevant memories
        │       (failures are caught and swallowed — memory is optional)
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

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import (
    get_context_builder_service,
    get_llm_service,
    get_memory_service,
)
from ..schemas.chat import ChatRequest, ChatResponse
from ..services.context_builder import ContextBuilderService
from ..services.llm_service import LLMService, LLMServiceError
from ..services.memory_service import MemoryService

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)

# Top-k memories to retrieve per request.
_MEMORY_TOP_K: int = 5


@router.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    llm_service: LLMService = Depends(get_llm_service),
    memory_service: MemoryService = Depends(get_memory_service),
    context_builder: ContextBuilderService = Depends(get_context_builder_service),
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
    # Step 1 — Save the incoming message as a memory BEFORE searching.
    # This ensures every user message accumulates in ChromaDB so future
    # turns can retrieve it.  Failures are non-fatal.
    # ------------------------------------------------------------------
    try:
        save_result = memory_service.save_memory(
            memory_text=request.message,
            metadata={"source": "chat"},
        )
        logger.info("User message saved as memory | id=%s", save_result["memory_id"])
    except Exception as save_exc:
        logger.warning(
            "Failed to save user message as memory — continuing | reason=%s",
            save_exc,
            exc_info=True,
        )

    # ------------------------------------------------------------------
    # Step 2 — Retrieve relevant memories for this message
    # ------------------------------------------------------------------
    memories: list[dict] = []
    retrieval_ms: float = 0.0

    retrieval_started = perf_counter()
    try:
        memories = memory_service.search_memory(
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

    return ChatResponse(response=response_text, memory_count=injected_count)
