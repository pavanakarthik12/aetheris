"""Chat router — Phase 9 Cognitive Request Router entry point.

Request flow:
    POST /api/chat
        │
        ├─ 1. CognitiveRequestRouter.classify()     — detect intent
        │
        ├─ 2. Route to correct subsystem(s):
        │      NORMAL_CHAT   → IMM → retrieve → context → LLM
        │      CREATE_MEMORY → evaluate → create → LLM
        │      UPDATE_MEMORY → evolve → update → retrieve → LLM
        │      DELETE_MEMORY → search → delete → verify → respond
        │      MERGE_MEMORY  → evolve → merge → retrieve → LLM
        │      SEARCH_MEMORY → retrieve → context → LLM
        │      WEB_SEARCH    → web → context → LLM
        │      SYSTEM_QUERY  → system info → LLM
        │      MULTI_ACTION  → split → execute each → combine → LLM
        │
        ├─ 3. Backend actions ALWAYS complete before LLM call
        │
        ├─ 4. LLM generates response (only if needed)
        │
        ├─ 5. Background Reflection (analysis + strengthen only)
        │
        └─ 6. Return response with routing debug info
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends

from ..dependencies import (
    get_request_router,
    get_reflection_service,
)
from ..schemas.chat import ChatRequest, ChatResponse, MemoryActionType
from ..services.reflection_service import ReflectionService
from ..services.request_router import CognitiveRequestRouter

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    router_service: CognitiveRequestRouter = Depends(get_request_router),
    reflection_service: ReflectionService = Depends(get_reflection_service),
) -> ChatResponse:
    logger.info(
        "Chat request received | message_length=%d", len(request.message),
    )

    # ------------------------------------------------------------------
    # Step 1–4 — Route through the Cognitive Request Router
    # The CRR handles: intent classification, subsystem dispatch,
    # backend execution, context building, and LLM call.
    # ------------------------------------------------------------------
    router_result = await router_service.route(request.message)

    # ------------------------------------------------------------------
    # Step 5 — Background Reflection (analysis + strengthen only)
    # ------------------------------------------------------------------
    background_tasks.add_task(
        reflection_service.reflect_with_context,
        request.message,
        router_result.response,
    )

    # ------------------------------------------------------------------
    # Step 6 — Build response (preserving backward compatibility)
    # ------------------------------------------------------------------
    logger.info(
        "Chat response ready | intent=%s | memory_action=%s | memory_count=%d | "
        "response_length=%d | router_duration_ms=%.2f",
        router_result.debug.detected_intent.value,
        router_result.memory_action.value,
        router_result.memory_count,
        len(router_result.response),
        router_result.debug.total_duration_ms,
    )

    return ChatResponse(
        response=router_result.response,
        memory_count=router_result.memory_count,
        memory_action=router_result.memory_action,
        memory_success=router_result.memory_success,
        memory_error=router_result.memory_error,
    )
