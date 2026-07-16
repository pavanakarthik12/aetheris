from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ..dependencies import get_conversation_memory, get_memory_hierarchy_service
from ..services.conversation_memory import ConversationMemory
from ..services.memory_hierarchy_service import MemoryHierarchyService

router = APIRouter(prefix="/api/debug", tags=["debug"])
logger = logging.getLogger(__name__)


@router.get("/memory-hierarchy")
async def memory_hierarchy_debug(
    conversation_memory: ConversationMemory = Depends(get_conversation_memory),
    memory_hierarchy: MemoryHierarchyService = Depends(get_memory_hierarchy_service),
) -> dict:
    hierarchy = memory_hierarchy.get_hierarchy_debug()
    hierarchy["current_session"] = conversation_memory.session_info
    hierarchy["conversation_memory"]["recent_messages"] = [
        {"role": m.get("role"), "content": m.get("content", "")[:200]}
        for m in conversation_memory.get_recent(turns=5)
    ]
    return hierarchy
