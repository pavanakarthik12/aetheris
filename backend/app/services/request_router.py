"""Cognitive Request Router (CRR) — the central execution engine for Aetheris."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from ..schemas.chat import MemoryActionType
from ..schemas.routing import (
    IntentClassification,
    IntentType,
    RouteStep,
    RouterDebugInfo,
    RouterResult,
)
from .chroma_service import ChromaService
from .context_builder import ContextBuilderService
from .embedding_service import EmbeddingService
from .immediate_memory_processor import ImmediateMemoryProcessor
from .intent_classifier import IntentClassifier
from .greeting_handler import detect_greeting
from .llm_service import (
    LLMService,
    LLMServiceError,
    LLMQuotaExceeded,
    LLMRateLimited,
    ProviderConnectionError,
    ProviderServerError,
    ProviderTimeout,
    ProviderUnauthorized,
    ProviderUnavailable,
)
from .memory_cache import MemorySearchCache
from .memory_evaluator import MemoryEvaluatorService
from .memory_evolution_service import MemoryEvolutionService
from .memory_hierarchy_service import MemoryHierarchyService
from .memory_service import MemoryService
from .metrics_collector import MetricsCollector
from .prompt_builder import PromptBuilder
from .reflection_service import ReflectionService
from .token_budget import select_budget

logger = logging.getLogger(__name__)

_MEMORY_TOP_K: int = 5


class CognitiveRequestRouter:
    """Single entry point for all user requests."""

    def __init__(
        self,
        llm_service: LLMService,
        memory_service: MemoryService,
        memory_evaluator: MemoryEvaluatorService,
        memory_evolution: MemoryEvolutionService,
        chroma_service: ChromaService,
        embedding_service: EmbeddingService,
        context_builder: ContextBuilderService,
        reflection_service: ReflectionService,
        intent_classifier: IntentClassifier,
        immediate_memory_processor: ImmediateMemoryProcessor,
        memory_hierarchy: MemoryHierarchyService | None = None,
    ) -> None:
        self._llm = llm_service
        self._memory_service = memory_service
        self._memory_evaluator = memory_evaluator
        self._memory_evolution = memory_evolution
        self._chroma_service = chroma_service
        self._embedding_service = embedding_service
        self._context_builder = context_builder
        self._reflection_service = reflection_service
        self._intent_classifier = intent_classifier
        self._imm = immediate_memory_processor
        self._prompt_builder = PromptBuilder()
        self._cache = MemorySearchCache()
        self._metrics = MetricsCollector()
        self._memory_hierarchy = memory_hierarchy

    async def route(
        self,
        message: str,
    ) -> RouterResult:
        started_at = perf_counter()
        steps: list[RouteStep] = []
        debug = RouterDebugifier()

        greeting_response = detect_greeting(message)
        if greeting_response is not None:
            elapsed = (perf_counter() - started_at) * 1000
            steps.append(RouteStep(
                subsystem="GreetingHandler",
                action="local_response",
                duration_ms=elapsed,
                success=True,
                detail="Greeting detected — LLM bypassed.",
            ))
            debug.set_duration(elapsed)
            debug.set_steps(steps)
            return RouterResult(
                response=greeting_response,
                memory_count=0,
                memory_action=MemoryActionType.SKIP,
                memory_success=True,
                debug=debug.build(),
            )

        classification = await self._classify(message, steps)
        debug.set_intent(classification)

        subsystems: list[str] = []

        try:
            if classification.primary_intent == IntentType.NORMAL_CHAT:
                result = await self._handle_with_hierarchy(message, steps, debug, classification)
                subsystems = ["MemoryHierarchy", "ImmediateMemoryProcessor", "LLM"]

            elif classification.primary_intent == IntentType.CONVERSATION_QUERY:
                result = await self._handle_conversation_query(message, steps, debug)
                subsystems = ["MemoryHierarchy", "LLM"]

            elif classification.primary_intent == IntentType.CREATE_MEMORY:
                result = await self._handle_create_memory(message, steps, debug)
                subsystems = ["MemoryEvaluator", "MemoryEvolution", "LLM"]

            elif classification.primary_intent == IntentType.UPDATE_MEMORY:
                result = await self._handle_update_memory(message, steps, debug)
                subsystems = ["MemoryEvolution", "MemoryService", "LLM"]

            elif classification.primary_intent == IntentType.DELETE_MEMORY:
                result = await self._handle_delete_memory(message, steps, debug)
                subsystems = ["MemoryService"]

            elif classification.primary_intent == IntentType.MERGE_MEMORY:
                result = await self._handle_merge_memory(message, steps, debug)
                subsystems = ["MemoryEvolution", "MemoryService", "LLM"]

            elif classification.primary_intent == IntentType.SEARCH_MEMORY:
                result = await self._handle_search_memory(message, steps, debug)
                subsystems = ["MemoryHierarchy", "LLM"]

            elif classification.primary_intent == IntentType.WEB_SEARCH:
                result = await self._handle_web_search(message, steps, debug)
                subsystems = ["WebSearch", "ContextBuilder", "LLM"]

            elif classification.primary_intent == IntentType.SYSTEM_QUERY:
                result = await self._handle_system_query(message, steps, debug)
                subsystems = ["MemoryHierarchy", "LLM"]

            elif classification.primary_intent == IntentType.MULTI_ACTION:
                result = await self._handle_multi_action(message, classification, steps, debug)
                subsystems = classification.metadata.get("subsystems", ["LLM"])

            else:
                result = await self._handle_with_hierarchy(message, steps, debug, classification)
                subsystems = ["MemoryHierarchy", "ImmediateMemoryProcessor", "LLM"]

        except Exception as exc:
            logger.exception("Router handler failed | intent=%s", classification.primary_intent)
            steps.append(RouteStep(
                subsystem="Router",
                action="error_handler",
                success=False,
                detail=f"Handler error: {exc}",
            ))
            result = await self._build_llm_response_with_context(
                message=message, steps=steps,
            )

        total_ms = (perf_counter() - started_at) * 1000
        debug.set_duration(total_ms)
        debug.set_subsystems(subsystems)
        debug.set_steps(steps)

        result.debug = debug.build()
        return result

    async def _classify(
        self,
        message: str,
        steps: list[RouteStep],
    ) -> IntentClassification:
        started = perf_counter()
        classification = await self._intent_classifier.classify(message)
        elapsed = (perf_counter() - started) * 1000

        logger.info(
            "Intent classified | intent=%s | confidence=%.2f | source=%s | duration_ms=%.2f",
            classification.primary_intent.value,
            classification.confidence,
            classification.classifier_source,
            elapsed,
        )

        steps.append(RouteStep(
            subsystem="IntentClassifier",
            action="classify",
            duration_ms=elapsed,
            success=True,
            detail=f"{classification.primary_intent.value} (conf={classification.confidence:.2f})",
        ))
        return classification

    async def _handle_normal_chat(
        self,
        message: str,
        steps: list[RouteStep],
        debug: RouterDebugifier,
        classification: IntentClassification | None = None,
    ) -> RouterResult:
        imm_result = await self._run_imm(message, steps)
        if imm_result.action in (MemoryActionType.CREATE, MemoryActionType.UPDATE, MemoryActionType.MERGE):
            self._cache.invalidate()

        memories = await self._retrieve_memories(message, steps)
        memory_context = self._build_context(memories, steps, query=message)
        injected_count = self._count_injected(memory_context)

        budget = select_budget(
            message,
            intent=classification.primary_intent if classification else None,
            memory_count=injected_count,
        )

        response = await self._call_llm(
            message=message,
            memory_context=memory_context,
            steps=steps,
            max_tokens=budget.max_tokens,
            temperature=budget.temperature,
        )

        debug.set_memory_action(imm_result.action)
        debug.set_memory_operation_count(
            1 if imm_result.action not in (MemoryActionType.SKIP, MemoryActionType.ERROR) else 0,
        )
        debug.set_reflection(True)

        return RouterResult(
            response=response,
            memory_count=injected_count,
            memory_action=imm_result.action,
            memory_success=imm_result.success,
            memory_error=imm_result.error,
        )

    async def _handle_create_memory(
        self,
        message: str,
        steps: list[RouteStep],
        debug: RouterDebugifier,
    ) -> RouterResult:
        started = perf_counter()

        try:
            evaluation = await self._memory_evaluator.evaluate_memory(message)
            steps.append(RouteStep(
                subsystem="MemoryEvaluator",
                action="evaluate",
                duration_ms=(perf_counter() - started) * 1000,
                success=True,
                detail=f"store={evaluation.store} category={evaluation.category}",
            ))

            if not evaluation.store:
                steps.append(RouteStep(
                    subsystem="MemoryEvaluator",
                    action="skip",
                    success=True,
                    detail="Evaluator decided not to store this message.",
                ))
                return await self._build_llm_response_with_context(
                    message, steps, hint="The information was not stored because it was not considered significant.",
                )

            result = await self._memory_evolution.create_memory(
                memory_text=message,
                metadata={
                    "source": "chat",
                    "category": evaluation.category,
                    "importance": evaluation.importance,
                    "reason": evaluation.reason,
                },
            )
            self._cache.invalidate()
            steps.append(RouteStep(
                subsystem="MemoryEvolution",
                action="create_memory",
                duration_ms=(perf_counter() - started) * 1000,
                success=True,
                detail=f"memory_id={result.get('memory_id')} dedup={result.get('duplicate', False)}",
            ))

            debug.set_memory_action(MemoryActionType.CREATE)
            debug.set_memory_operation_count(1)
            debug.set_reflection(True)

            budget = select_budget(message, intent=IntentType.CREATE_MEMORY)
            response = await self._call_llm(
                message=message,
                memory_context="Note: The user's message was just saved as a new memory. I've saved that information.",
                steps=steps,
                max_tokens=budget.max_tokens,
                temperature=budget.temperature,
            )
            return RouterResult(
                response=response,
                memory_count=0,
                memory_action=MemoryActionType.CREATE,
                memory_success=True,
            )

        except Exception as exc:
            steps.append(RouteStep(
                subsystem="MemoryEvolution",
                action="create_memory",
                success=False,
                detail=f"Failed to create memory: {exc}",
            ))
            return await self._build_llm_response_with_context(
                message, steps, hint="I wasn't able to save that information due to an error.",
            )

    async def _handle_update_memory(
        self,
        message: str,
        steps: list[RouteStep],
        debug: RouterDebugifier,
    ) -> RouterResult:
        try:
            related = await self._memory_service.search_memory(query=message, top_k=_MEMORY_TOP_K)
            evaluation = await self._memory_evaluator.evaluate_memory(message)

            decision = await self._memory_evolution.decide_evolution(
                memory_text=message,
                existing_evaluation={
                    "store": True,
                    "category": evaluation.category,
                    "importance": evaluation.importance,
                    "reason": evaluation.reason,
                },
            )

            action = decision.get("action", "SKIP")

            if action in ("UPDATE", "MERGE"):
                target_id = decision.get("target_id")
                if target_id:
                    result = await self._memory_evolution.update_memory(
                        memory_id=target_id,
                        new_text=message,
                        new_metadata={
                            "source": "chat",
                            "category": evaluation.category,
                            "importance": evaluation.importance,
                            "reason": evaluation.reason,
                        },
                    )
                    self._cache.invalidate()
                    steps.append(RouteStep(
                        subsystem="MemoryEvolution",
                        action="update_memory",
                        success=True,
                        detail=f"memory_id={target_id} version={result.get('version')}",
                    ))
                    debug.set_memory_action(MemoryActionType.UPDATE)
                    debug.set_memory_operation_count(1)
                    debug.set_reflection(True)

            elif action == "CREATE":
                result = await self._memory_evolution.create_memory(
                    memory_text=message,
                    metadata={
                        "source": "chat",
                        "category": evaluation.category,
                        "importance": evaluation.importance,
                        "reason": evaluation.reason,
                    },
                )
                self._cache.invalidate()
                steps.append(RouteStep(
                    subsystem="MemoryEvolution",
                    action="create_memory",
                    success=True,
                    detail=f"memory_id={result.get('memory_id')}",
                ))
                debug.set_memory_action(MemoryActionType.CREATE)
                debug.set_memory_operation_count(1)
                debug.set_reflection(True)

            memories = await self._retrieve_memories(message, steps)
            memory_context = self._build_context(memories, steps, query=message)

            budget = select_budget(message, intent=IntentType.UPDATE_MEMORY, memory_count=self._count_injected(memory_context))
            response = await self._call_llm(
                message=message,
                memory_context=memory_context,
                steps=steps,
                max_tokens=budget.max_tokens,
                temperature=budget.temperature,
            )

            return RouterResult(
                response=response,
                memory_count=self._count_injected(memory_context),
                memory_action=debug.memory_action,
                memory_success=True,
            )

        except Exception as exc:
            steps.append(RouteStep(
                subsystem="MemoryEvolution",
                action="update_memory",
                success=False,
                detail=f"Update failed: {exc}",
            ))
            return await self._build_llm_response_with_context(
                message, steps, hint="I wasn't able to update that memory due to an error.",
            )

    async def _handle_delete_memory(
        self,
        message: str,
        steps: list[RouteStep],
        debug: RouterDebugifier,
    ) -> RouterResult:
        started = perf_counter()

        try:
            related = await self._memory_service.search_memory(query=message, top_k=1)
            if not related:
                steps.append(RouteStep(
                    subsystem="MemoryService",
                    action="search",
                    success=True,
                    detail="No matching memory found to delete.",
                    duration_ms=(perf_counter() - started) * 1000,
                ))
                debug.set_memory_action(MemoryActionType.SKIP)
                return RouterResult(
                    response="I couldn't delete the requested memory because it wasn't found. "
                             "No matching memory exists in my records.",
                    memory_count=0,
                    memory_action=MemoryActionType.SKIP,
                    memory_success=True,
                    memory_error="No matching memory found.",
                )

            mem = related[0]
            score = mem.get("score", 0.0)
            if score < 0.5:
                steps.append(RouteStep(
                    subsystem="MemoryService",
                    action="search",
                    success=True,
                    detail=f"Best match score {score:.2f} below threshold 0.5",
                    duration_ms=(perf_counter() - started) * 1000,
                ))
                return RouterResult(
                    response="I couldn't find a clearly matching memory to delete. "
                             f"The closest match had a similarity of {score:.2f}, which is too low to proceed.",
                    memory_count=0,
                    memory_action=MemoryActionType.SKIP,
                    memory_success=True,
                    memory_error=f"Best score {score:.2f} below threshold.",
                )

            self._memory_service.delete_memory(mem["id"])
            self._cache.invalidate()
            steps.append(RouteStep(
                subsystem="MemoryService",
                action="delete_memory",
                success=True,
                detail=f"Deleted memory_id={mem['id']}",
                duration_ms=(perf_counter() - started) * 1000,
            ))
            debug.set_memory_action(MemoryActionType.DELETE)
            debug.set_memory_operation_count(1)

            return RouterResult(
                response="I've successfully deleted that memory.",
                memory_count=0,
                memory_action=MemoryActionType.DELETE,
                memory_success=True,
            )

        except Exception as exc:
            steps.append(RouteStep(
                subsystem="MemoryService",
                action="delete_memory",
                success=False,
                detail=f"Delete failed: {exc}",
            ))
            return RouterResult(
                response="I couldn't delete the requested memory because an error occurred.",
                memory_count=0,
                memory_action=MemoryActionType.ERROR,
                memory_success=False,
                memory_error=str(exc),
            )

    async def _handle_merge_memory(
        self,
        message: str,
        steps: list[RouteStep],
        debug: RouterDebugifier,
    ) -> RouterResult:
        try:
            evaluation = await self._memory_evaluator.evaluate_memory(message)
            decision = await self._memory_evolution.decide_evolution(
                memory_text=message,
                existing_evaluation={
                    "store": True,
                    "category": evaluation.category,
                    "importance": evaluation.importance,
                    "reason": evaluation.reason,
                },
            )

            target_id = decision.get("target_id")
            if target_id:
                existing_text = await self._memory_evolution.get_memory_document(target_id) or ""
                merged_text = f"{existing_text}\n{message}"
                result = await self._memory_evolution.update_memory(
                    memory_id=target_id,
                    new_text=merged_text,
                    new_metadata={
                        "source": "chat",
                        "category": evaluation.category,
                        "importance": evaluation.importance,
                        "reason": evaluation.reason,
                    },
                )
                self._cache.invalidate()
                steps.append(RouteStep(
                    subsystem="MemoryEvolution",
                    action="merge_memory",
                    success=True,
                    detail=f"target={target_id} version={result.get('version')}",
                ))
                debug.set_memory_action(MemoryActionType.MERGE)
                debug.set_memory_operation_count(1)
                debug.set_reflection(True)

            memories = await self._retrieve_memories(message, steps)
            memory_context = self._build_context(memories, steps, query=message)
            budget = select_budget(message, memory_count=self._count_injected(memory_context))
            response = await self._call_llm(
                message=message,
                memory_context=memory_context,
                steps=steps,
                max_tokens=budget.max_tokens,
                temperature=budget.temperature,
            )

            return RouterResult(
                response=response,
                memory_count=self._count_injected(memory_context),
                memory_action=debug.memory_action,
                memory_success=True,
            )

        except Exception as exc:
            steps.append(RouteStep(
                subsystem="MemoryEvolution",
                action="merge_memory",
                success=False,
                detail=f"Merge failed: {exc}",
            ))
            return await self._build_llm_response_with_context(
                message, steps, hint="I wasn't able to merge that information due to an error.",
            )

    async def _handle_search_memory(
        self,
        message: str,
        steps: list[RouteStep],
        debug: RouterDebugifier,
    ) -> RouterResult:
        memories = await self._retrieve_memories(message, steps)
        memory_context = self._build_context(memories, steps, query=message)
        injected_count = self._count_injected(memory_context)

        if not memory_context:
            steps.append(RouteStep(
                subsystem="MemoryService",
                action="search",
                success=True,
                detail="No memories found matching the query.",
            ))

        budget = select_budget(message, intent=IntentType.SEARCH_MEMORY)
        response = await self._call_llm(
            message=message,
            memory_context=memory_context,
            steps=steps,
            max_tokens=budget.max_tokens,
            temperature=budget.temperature,
        )

        return RouterResult(
            response=response,
            memory_count=injected_count,
            memory_action=MemoryActionType.SKIP,
            memory_success=True,
        )

    async def _handle_web_search(
        self,
        message: str,
        steps: list[RouteStep],
        debug: RouterDebugifier,
    ) -> RouterResult:
        debug.set_internet_used(True)

        steps.append(RouteStep(
            subsystem="WebSearch",
            action="search",
            success=True,
            detail="Web search dispatched.",
        ))

        memories = await self._retrieve_memories(message, steps)
        memory_context = self._build_context(memories, steps, query=message)
        combined_context = (
            "[Web search requested. Respond from your existing knowledge or note if a live connection is needed.]"
        )
        if memory_context:
            combined_context = f"{combined_context}\n\n{memory_context}"

        budget = select_budget(message)
        response = await self._call_llm(
            message=message,
            memory_context=combined_context,
            steps=steps,
            max_tokens=budget.max_tokens,
            temperature=budget.temperature,
        )

        return RouterResult(
            response=response,
            memory_count=self._count_injected(memory_context),
            memory_action=MemoryActionType.SKIP,
            memory_success=True,
        )

    async def _handle_system_query(
        self,
        message: str,
        steps: list[RouteStep],
        debug: RouterDebugifier,
    ) -> RouterResult:
        memories = await self._retrieve_memories(message, steps)
        memory_context = self._build_context(memories, steps, query=message)

        budget = select_budget(message, intent=IntentType.SYSTEM_QUERY)
        response = await self._call_llm(
            message=message,
            memory_context=memory_context,
            steps=steps,
            max_tokens=budget.max_tokens,
            temperature=budget.temperature,
        )

        return RouterResult(
            response=response,
            memory_count=self._count_injected(memory_context),
            memory_action=MemoryActionType.SKIP,
            memory_success=True,
        )

    async def _handle_multi_action(
        self,
        message: str,
        classification: IntentClassification,
        steps: list[RouteStep],
        debug: RouterDebugifier,
    ) -> RouterResult:
        sub_intents = classification.sub_intents
        if not sub_intents:
            return await self._handle_normal_chat(message, steps, debug, classification)

        sub_steps: list[RouteStep] = []
        context_parts: list[str] = []
        memory_action = MemoryActionType.SKIP
        subsystems: set[str] = set()

        for i, sub_intent in enumerate(sub_intents):
            sub_start = perf_counter()
            try:
                if sub_intent == IntentType.CREATE_MEMORY:
                    sub_result = await self._memory_evaluator.evaluate_memory(message)
                    if sub_result.store:
                        result = await self._memory_evolution.create_memory(
                            memory_text=message,
                            metadata={
                                "source": "chat",
                                "category": sub_result.category,
                                "importance": sub_result.importance,
                                "reason": sub_result.reason,
                            },
                        )
                        self._cache.invalidate()
                        context_parts.append(f"[Action {i + 1}: Memory saved ({result.get('memory_id', '')[:8]}…)]")
                        memory_action = MemoryActionType.CREATE
                        subsystems.add("MemoryEvaluator")
                        subsystems.add("MemoryEvolution")
                        debug.set_reflection(True)

                elif sub_intent == IntentType.WEB_SEARCH:
                    context_parts.append(f"[Action {i + 1}: Web search requested]")
                    subsystems.add("WebSearch")
                    debug.set_internet_used(True)

                elif sub_intent == IntentType.SEARCH_MEMORY:
                    sub_memories = await self._retrieve_memories(message, steps)
                    sub_context = self._build_context(sub_memories, steps, query=message)
                    if sub_context:
                        context_parts.append(f"[Action {i + 1}: Found memories]\n{sub_context}")
                    subsystems.add("MemoryService")

                else:
                    context_parts.append(f"[Action {i + 1}: Processed]")

                sub_steps.append(RouteStep(
                    subsystem=f"MultiAction[{sub_intent.value}]",
                    action="execute",
                    duration_ms=(perf_counter() - sub_start) * 1000,
                    success=True,
                ))
            except Exception as exc:
                sub_steps.append(RouteStep(
                    subsystem=f"MultiAction[{sub_intent.value}]",
                    action="execute",
                    duration_ms=(perf_counter() - sub_start) * 1000,
                    success=False,
                    detail=str(exc),
                ))

        steps.extend(sub_steps)

        combined_context = "\n".join(context_parts) if context_parts else ""
        memories = await self._retrieve_memories(message, steps)
        memory_context = self._build_context(memories, steps, query=message)

        if memory_context:
            combined_context = f"{combined_context}\n\n{memory_context}" if combined_context else memory_context

        budget = select_budget(message)
        response = await self._call_llm(
            message=message,
            memory_context=combined_context,
            steps=steps,
            max_tokens=budget.max_tokens,
            temperature=budget.temperature,
        )

        debug.set_memory_action(memory_action)
        debug.set_memory_operation_count(1 if memory_action != MemoryActionType.SKIP else 0)

        classification.metadata["subsystems"] = list(subsystems)

        return RouterResult(
            response=response,
            memory_count=self._count_injected(memory_context),
            memory_action=memory_action,
            memory_success=True,
        )

    async def _handle_with_hierarchy(
        self,
        message: str,
        steps: list[RouteStep],
        debug: RouterDebugifier,
        classification: IntentClassification | None = None,
    ) -> RouterResult:
        imm_result = await self._run_imm(message, steps)
        if imm_result.action in (MemoryActionType.CREATE, MemoryActionType.UPDATE, MemoryActionType.MERGE):
            self._cache.invalidate()

        intent = classification.primary_intent if classification else IntentType.NORMAL_CHAT

        use_hierarchy = self._memory_hierarchy is not None and intent not in (
            IntentType.CREATE_MEMORY, IntentType.UPDATE_MEMORY,
            IntentType.DELETE_MEMORY, IntentType.MERGE_MEMORY,
        )

        if use_hierarchy:
            hierarchy = await self._resolve_hierarchy(message, intent, steps)
            memory_context = hierarchy.context_text
            debug.set_memory_layer(hierarchy.memory_layer)
            debug.set_conversation_count(len(hierarchy.conversation_messages))
            debug.set_long_term_count(len(hierarchy.long_term_memories))
            debug.set_system_count(len(hierarchy.system_memories))
            debug.set_context_size(len(memory_context))
        else:
            memories = await self._retrieve_memories(message, steps)
            memory_context = self._build_context(memories, steps, query=message)

        injected_count = self._count_injected(memory_context)

        budget = select_budget(
            message,
            intent=intent,
            memory_count=injected_count,
        )

        if use_hierarchy:
            response = await self._call_llm_direct(
                message=message,
                memory_context=memory_context,
                steps=steps,
                max_tokens=budget.max_tokens,
                temperature=budget.temperature,
            )
        else:
            response = await self._call_llm(
                message=message,
                memory_context=memory_context,
                steps=steps,
                max_tokens=budget.max_tokens,
                temperature=budget.temperature,
            )

        debug.set_memory_action(imm_result.action)
        debug.set_memory_operation_count(
            1 if imm_result.action not in (MemoryActionType.SKIP, MemoryActionType.ERROR) else 0,
        )
        debug.set_reflection(True)

        return RouterResult(
            response=response,
            memory_count=injected_count,
            memory_action=imm_result.action,
            memory_success=imm_result.success,
            memory_error=imm_result.error,
        )

    async def _handle_conversation_query(
        self,
        message: str,
        steps: list[RouteStep],
        debug: RouterDebugifier,
    ) -> RouterResult:
        hierarchy = await self._resolve_hierarchy(message, IntentType.CONVERSATION_QUERY, steps)
        memory_context = hierarchy.context_text
        injected_count = 0

        debug.set_memory_layer(hierarchy.memory_layer)
        debug.set_conversation_count(len(hierarchy.conversation_messages))
        debug.set_context_size(len(memory_context))

        budget = select_budget(message, intent=IntentType.CONVERSATION_QUERY)
        response = await self._call_llm(
            message=message,
            memory_context=memory_context,
            steps=steps,
            max_tokens=budget.max_tokens,
            temperature=budget.temperature,
        )

        return RouterResult(
            response=response,
            memory_count=injected_count,
            memory_action=MemoryActionType.SKIP,
            memory_success=True,
        )

    async def _resolve_hierarchy(
        self,
        message: str,
        intent: IntentType,
        steps: list[RouteStep],
    ) -> Any:
        started = perf_counter()
        hierarchy = await self._memory_hierarchy.resolve(message, intent)
        elapsed = (perf_counter() - started) * 1000
        steps.append(RouteStep(
            subsystem="MemoryHierarchy",
            action="resolve",
            duration_ms=elapsed,
            success=True,
            detail=f"layer={hierarchy.memory_layer} conv={len(hierarchy.conversation_messages)} lt={len(hierarchy.long_term_memories)} sys={len(hierarchy.system_memories)}",
        ))
        return hierarchy

    async def _run_imm(
        self,
        message: str,
        steps: list[RouteStep],
    ):
        started = perf_counter()
        result = await self._imm.process_message(message)
        elapsed = (perf_counter() - started) * 1000
        steps.append(RouteStep(
            subsystem="ImmediateMemoryProcessor",
            action=result.action.value,
            duration_ms=elapsed,
            success=result.success,
            detail=f"action={result.action.value}" + (f" error={result.error}" if result.error else ""),
        ))
        return result

    async def _retrieve_memories(
        self,
        message: str,
        steps: list[RouteStep],
    ) -> list[dict[str, Any]]:
        started = perf_counter()

        cached = self._cache.get(message)
        if cached is not None:
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="MemoryCache",
                action="cache_hit",
                duration_ms=elapsed,
                success=True,
                detail=f"found={len(cached)}",
            ))
            return cached

        try:
            memories = await self._memory_service.search_memory(
                query=message,
                top_k=_MEMORY_TOP_K,
            )
            self._cache.set(message, memories)
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="MemoryService",
                action="search_memory",
                duration_ms=elapsed,
                success=True,
                detail=f"found={len(memories)}",
            ))
            return memories
        except Exception as exc:
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="MemoryService",
                action="search_memory",
                duration_ms=elapsed,
                success=False,
                detail=str(exc),
            ))
            return []

    def _build_context(
        self,
        memories: list[dict[str, Any]],
        steps: list[RouteStep],
        query: str = "",
    ) -> str:
        memory_context = self._context_builder.build_memory_context(memories, query=query)
        steps.append(RouteStep(
            subsystem="ContextBuilder",
            action="build_memory_context",
            success=True,
            detail=f"lines={self._count_injected(memory_context)}",
        ))
        return memory_context

    async def _call_llm(
        self,
        message: str,
        memory_context: str,
        steps: list[RouteStep],
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        started = perf_counter()
        try:
            response = await self._llm.generate_with_context(
                user_message=message,
                system_prompt=system_prompt or self._prompt_builder.chat_system(),
                memory_context=memory_context,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="LLM",
                action="generate_with_context",
                duration_ms=elapsed,
                success=True,
                detail=f"response_length={len(response)} max_tokens={max_tokens}",
            ))
            return response
        except LLMQuotaExceeded:
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="LLM",
                action="generate_with_context",
                duration_ms=elapsed,
                success=False,
                detail="402 Quota Exceeded",
            ))
            return ("The configured AI model is currently unavailable because "
                    "the provider account has insufficient credits.")
        except LLMRateLimited:
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="LLM",
                action="generate_with_context",
                duration_ms=elapsed,
                success=False,
                detail="429 Rate Limited",
            ))
            return "The AI provider is temporarily busy. Please try again in a few seconds."
        except ProviderUnauthorized:
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="LLM",
                action="generate_with_context",
                duration_ms=elapsed,
                success=False,
                detail="401 Unauthorized",
            ))
            return ("The AI provider returned an authentication error. "
                    "The API key may be invalid or expired.")
        except ProviderUnavailable:
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="LLM",
                action="generate_with_context",
                duration_ms=elapsed,
                success=False,
                detail="503 Unavailable",
            ))
            return "The AI provider is temporarily unavailable."
        except ProviderConnectionError:
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="LLM",
                action="generate_with_context",
                duration_ms=elapsed,
                success=False,
                detail="Connection Error",
            ))
            return "Unable to connect to the AI provider. Please check your network connection."
        except ProviderServerError:
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="LLM",
                action="generate_with_context",
                duration_ms=elapsed,
                success=False,
                detail="500 Server Error",
            ))
            return "The AI provider returned a server error. Please try again."
        except ProviderTimeout:
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="LLM",
                action="generate_with_context",
                duration_ms=elapsed,
                success=False,
                detail="Timeout",
            ))
            return "The AI service took too long to respond. Please try again."
        except LLMServiceError as exc:
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="LLM",
                action="generate_with_context",
                duration_ms=elapsed,
                success=False,
                detail=str(exc),
            ))
            return (f"I encountered an issue while generating a response. "
                    f"Please try again or rephrase your message.")
        except Exception as exc:
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="LLM",
                action="generate_with_context",
                duration_ms=elapsed,
                success=False,
                detail=f"Unexpected error: {exc}",
            ))
            logger.exception("Unexpected LLM error")
            return "An unexpected error occurred while processing your request. Please try again."

    async def _call_llm_direct(
        self,
        message: str,
        memory_context: str,
        steps: list[RouteStep],
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        started = perf_counter()
        system = system_prompt or self._prompt_builder.chat_system()
        content_parts: list[str] = []
        if memory_context.strip():
            content_parts.append(memory_context.strip())
        content_parts.append(f"Current User Message:\n{message.strip()}")
        prompt = f"{system}\n\n" + "\n\n".join(content_parts)
        try:
            response = await self._llm.generate_text(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="LLM",
                action="generate_direct",
                duration_ms=elapsed,
                success=True,
                detail=f"response_length={len(response)} max_tokens={max_tokens}",
            ))
            return response
        except LLMQuotaExceeded:
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="LLM",
                action="generate_direct",
                duration_ms=elapsed,
                success=False,
                detail="402 Quota Exceeded",
            ))
            return "The configured AI model is currently unavailable because the provider account has insufficient credits."
        except LLMRateLimited:
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="LLM",
                action="generate_direct",
                duration_ms=elapsed,
                success=False,
                detail="429 Rate Limited",
            ))
            return "The AI provider is temporarily busy. Please try again in a few seconds."
        except ProviderUnauthorized:
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="LLM",
                action="generate_direct",
                duration_ms=elapsed,
                success=False,
                detail="401 Unauthorized",
            ))
            return "The AI provider returned an authentication error. The API key may be invalid or expired."
        except ProviderUnavailable:
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="LLM",
                action="generate_direct",
                duration_ms=elapsed,
                success=False,
                detail="503 Unavailable",
            ))
            return "The AI provider is temporarily unavailable."
        except ProviderConnectionError:
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="LLM",
                action="generate_direct",
                duration_ms=elapsed,
                success=False,
                detail="Connection Error",
            ))
            return "Unable to connect to the AI provider. Please check your network connection."
        except ProviderServerError:
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="LLM",
                action="generate_direct",
                duration_ms=elapsed,
                success=False,
                detail="500 Server Error",
            ))
            return "The AI provider returned a server error. Please try again."
        except ProviderTimeout:
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="LLM",
                action="generate_direct",
                duration_ms=elapsed,
                success=False,
                detail="Timeout",
            ))
            return "The AI service took too long to respond. Please try again."
        except LLMServiceError as exc:
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="LLM",
                action="generate_direct",
                duration_ms=elapsed,
                success=False,
                detail=str(exc),
            ))
            return f"I encountered an issue while generating a response. Please try again or rephrase your message."
        except Exception as exc:
            elapsed = (perf_counter() - started) * 1000
            steps.append(RouteStep(
                subsystem="LLM",
                action="generate_direct",
                duration_ms=elapsed,
                success=False,
                detail=f"Unexpected error: {exc}",
            ))
            logger.exception("Unexpected LLM error")
            return "An unexpected error occurred while processing your request. Please try again."

    async def _build_llm_response_with_context(
        self,
        message: str,
        steps: list[RouteStep],
        hint: str = "",
    ) -> RouterResult:
        memories = await self._retrieve_memories(message, steps)
        memory_context = self._build_context(memories, steps, query=message)
        if hint and memory_context:
            memory_context = f"{hint}\n\n{memory_context}"
        elif hint:
            memory_context = hint

        budget = select_budget(message)
        response = await self._call_llm(
            message=message,
            memory_context=memory_context,
            steps=steps,
            max_tokens=budget.max_tokens,
            temperature=budget.temperature,
        )
        return RouterResult(
            response=response,
            memory_count=self._count_injected(memory_context),
            memory_action=MemoryActionType.SKIP,
            memory_success=True,
        )

    @staticmethod
    def _count_injected(memory_context: str) -> int:
        if not memory_context:
            return 0
        return sum(1 for line in memory_context.splitlines() if line.startswith("- "))

    def get_metrics_snapshot(self) -> dict[str, Any]:
        return self._metrics.snapshot()


class RouterDebugifier:
    """Builds RouterDebugInfo incrementally as the router executes."""

    def __init__(self) -> None:
        self._intent: IntentType = IntentType.UNKNOWN
        self._confidence: float = 0.0
        self._route: str = ""
        self._steps: list[RouteStep] = []
        self._subsystems: list[str] = []
        self._memory_action: MemoryActionType = MemoryActionType.SKIP
        self._memory_op_count: int = 0
        self._reflection: bool = False
        self._internet: bool = False
        self._duration: float = 0.0
        self._memory_layer: str = "none"
        self._conv_count: int = 0
        self._lt_count: int = 0
        self._sys_count: int = 0
        self._context_size: int = 0

    def set_intent(self, classification: IntentClassification) -> None:
        self._intent = classification.primary_intent
        self._confidence = classification.confidence
        self._route = classification.primary_intent.value

    def set_steps(self, steps: list[RouteStep]) -> None:
        self._steps = steps

    def set_subsystems(self, subsystems: list[str]) -> None:
        self._subsystems = subsystems

    def set_duration(self, ms: float) -> None:
        self._duration = ms

    @property
    def memory_action(self) -> MemoryActionType:
        return self._memory_action

    def set_memory_action(self, action: MemoryActionType) -> None:
        self._memory_action = action

    def set_memory_operation_count(self, count: int) -> None:
        self._memory_op_count = count

    def set_reflection(self, value: bool) -> None:
        self._reflection = value

    def set_internet_used(self, value: bool) -> None:
        self._internet = value

    def set_memory_layer(self, layer: str) -> None:
        self._memory_layer = layer

    def set_conversation_count(self, count: int) -> None:
        self._conv_count = count

    def set_long_term_count(self, count: int) -> None:
        self._lt_count = count

    def set_system_count(self, count: int) -> None:
        self._sys_count = count

    def set_context_size(self, size: int) -> None:
        self._context_size = size

    def build(self) -> RouterDebugInfo:
        return RouterDebugInfo(
            detected_intent=self._intent,
            confidence=self._confidence,
            route_taken=self._route,
            steps=self._steps,
            total_duration_ms=self._duration,
            subsystems_used=self._subsystems,
            memory_action=self._memory_action,
            memory_operation_count=self._memory_op_count,
            reflection_triggered=self._reflection,
            internet_used=self._internet,
            memory_layer=self._memory_layer,
            conversation_messages_used=self._conv_count,
            long_term_memories_used=self._lt_count,
            system_memories_used=self._sys_count,
            context_size_chars=self._context_size,
        )
