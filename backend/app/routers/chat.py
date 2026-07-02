"""Chat router for the Phase 1 LLM communication layer."""

from __future__ import annotations

import logging
from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_llm_service
from ..schemas.chat import ChatRequest, ChatResponse
from ..services.llm_service import LLMService, LLMServiceError

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, llm_service: LLMService = Depends(get_llm_service)) -> ChatResponse:
    """Forward a single user message to Qwen and return the reply."""

    logger.info("Incoming chat request | message_length=%s", len(request.message))
    started_at = perf_counter()

    try:
        response_text = await llm_service.generate_text(request.message)
    except LLMServiceError as exc:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    logger.info("Chat request completed | response_time_ms=%.2f", (perf_counter() - started_at) * 1000)
    return ChatResponse(response=response_text)