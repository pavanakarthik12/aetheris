from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from ..schemas.routing import IntentType
from .conversation_memory import ConversationMemory
from .memory_service import MemoryService
from .system_memory import SystemMemory

logger = logging.getLogger(__name__)


@dataclass
class HierarchyResult:
    memory_layer: str = "none"
    conversation_messages: list[dict[str, Any]] = field(default_factory=list)
    long_term_memories: list[dict[str, Any]] = field(default_factory=list)
    system_memories: list[dict[str, Any]] = field(default_factory=list)
    context_text: str = ""
    execution_time_ms: float = 0.0


class MemoryHierarchyService:
    def __init__(
        self,
        conversation_memory: ConversationMemory,
        long_term_memory: MemoryService,
        system_memory: SystemMemory,
    ) -> None:
        self._conversation = conversation_memory
        self._long_term = long_term_memory
        self._system = system_memory

    async def resolve(
        self,
        message: str,
        intent: IntentType,
    ) -> HierarchyResult:
        started = perf_counter()

        if intent == IntentType.CONVERSATION_QUERY:
            result = self._use_conversation_only(message)

        elif intent == IntentType.SEARCH_MEMORY:
            result = await self._use_long_term_only(message)

        elif intent == IntentType.SYSTEM_QUERY:
            result = self._use_system_only(message)

        elif intent in (IntentType.MATH, IntentType.GREETING):
            result = HierarchyResult(memory_layer="none")

        elif intent == IntentType.PROGRAMMING:
            conv = self._conversation.get_recent(turns=5)
            result = HierarchyResult(
                memory_layer="conversation",
                conversation_messages=conv,
                context_text=self._build_conversation_text(conv),
            )

        elif intent == IntentType.NORMAL_CHAT:
            conv = self._conversation.get_recent(turns=10)
            lt = await self._long_term.search_memory(query=message, top_k=5)
            result = HierarchyResult(
                memory_layer="conversation+long_term",
                conversation_messages=conv,
                long_term_memories=lt,
                context_text=self._build_combined_text(conv, lt),
            )

        else:
            conv = self._conversation.get_recent(turns=5)
            lt = await self._long_term.search_memory(query=message, top_k=5)
            result = HierarchyResult(
                memory_layer="conversation+long_term",
                conversation_messages=conv,
                long_term_memories=lt,
                context_text=self._build_combined_text(conv, lt),
            )

        result.execution_time_ms = (perf_counter() - started) * 1000

        logger.info(
            "Memory hierarchy resolved | layer=%s | conv=%d | lt=%d | system=%d | %.2fms",
            result.memory_layer,
            len(result.conversation_messages),
            len(result.long_term_memories),
            len(result.system_memories),
            result.execution_time_ms,
        )
        return result

    def _use_conversation_only(self, message: str) -> HierarchyResult:
        messages = self._conversation.search(message)
        if not messages:
            messages = self._conversation.get_recent(turns=10)
        return HierarchyResult(
            memory_layer="conversation",
            conversation_messages=messages,
            context_text=self._build_conversation_text(messages),
        )

    async def _use_long_term_only(self, message: str) -> HierarchyResult:
        lt = await self._long_term.search_memory(query=message, top_k=10)
        return HierarchyResult(
            memory_layer="long_term",
            long_term_memories=lt,
            context_text=self._build_long_term_text(lt),
        )

    def _use_system_only(self, message: str) -> HierarchyResult:
        facts = self._system.search(message)
        return HierarchyResult(
            memory_layer="system",
            system_memories=facts,
            context_text=self._system.to_context_block(),
        )

    @staticmethod
    def _build_conversation_text(messages: list[dict[str, Any]]) -> str:
        if not messages:
            return ""
        lines: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if content:
                label = "User" if role == "user" else "Assistant"
                lines.append(f"{label}: {content}")
        return "Previous Conversation:\n" + "\n".join(lines)

    @staticmethod
    def _build_long_term_text(memories: list[dict[str, Any]]) -> str:
        if not memories:
            return ""
        from .prompt_builder import PromptBuilder
        return PromptBuilder.user_facts_block(memories)

    @staticmethod
    def _build_combined_text(
        messages: list[dict[str, Any]],
        memories: list[dict[str, Any]],
    ) -> str:
        parts: list[str] = []
        conv_text = MemoryHierarchyService._build_conversation_text(messages)
        if conv_text:
            parts.append(conv_text)
        lt_text = MemoryHierarchyService._build_long_term_text(memories)
        if lt_text:
            parts.append(lt_text)
        return "\n\n".join(parts)

    def get_hierarchy_debug(self) -> dict[str, Any]:
        sys_facts = self._system.get_all()
        return {
            "conversation_memory": self._conversation.session_info,
            "long_term_memory": {
                "storage": "ChromaDB",
                "description": "Persistent user facts stored across sessions.",
            },
            "system_memory": {
                "facts": sys_facts,
                "description": "Static information about Aetheris itself.",
            },
            "memory_isolation": {
                "conversation": "Never embedded into ChromaDB. Temporary per-session.",
                "long_term": "Persists across sessions. Only queried for relevant intents.",
                "system": "Read-only. Never editable by user commands.",
            },
        }
