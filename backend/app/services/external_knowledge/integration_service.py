"""External Knowledge Integration Service — orchestrates search and prompt injection.

Called by the Request Router when the Reasoning Engine determines
that external knowledge is needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from ...schemas.external_knowledge import NormalizedSearchResultItem
from .context_formatter import ExternalKnowledgeContextFormatter
from .decision_layer import ExternalKnowledgeDecisionLayer
from .search_pipeline import SearchPipeline

logger = logging.getLogger(__name__)


@dataclass
class ExternalKnowledgeResult:
    triggered: bool = False
    success: bool = False
    query: str = ""
    result_count: int = 0
    context_block: str = ""
    instruction: str = ""
    execution_time_ms: float = 0.0
    error: str | None = None


class ExternalKnowledgeIntegrationService:

    def __init__(
        self,
        decision_layer: ExternalKnowledgeDecisionLayer,
        search_pipeline: SearchPipeline,
        formatter: ExternalKnowledgeContextFormatter | None = None,
    ) -> None:
        self._decision = decision_layer
        self._pipeline = search_pipeline
        self._formatter = formatter or ExternalKnowledgeContextFormatter()

    async def execute(
        self,
        message: str,
        semantic_intent: Any,
    ) -> ExternalKnowledgeResult:
        started = perf_counter()

        if not self._decision.needs_external_knowledge(message, semantic_intent):
            return ExternalKnowledgeResult(triggered=False)

        logger.info(
            "ExternalKnowledgeIntegration | triggered | intent=%s | query=%.50s",
            semantic_intent.value if hasattr(semantic_intent, "value") else semantic_intent,
            message,
        )

        try:
            search_response = await self._pipeline.execute(message.strip(), max_results=5)

            elapsed = (perf_counter() - started) * 1000

            if search_response.success and search_response.results:
                context_block = self._formatter.format_results(search_response.results)
                instruction = self._formatter.format_instruction()

                logger.info(
                    "ExternalKnowledgeIntegration | success | provider=%s | results=%d | duration_ms=%.2f",
                    search_response.provider,
                    search_response.result_count,
                    elapsed,
                )

                return ExternalKnowledgeResult(
                    triggered=True,
                    success=True,
                    query=message.strip(),
                    result_count=search_response.result_count,
                    context_block=context_block,
                    instruction=instruction,
                    execution_time_ms=elapsed,
                    error=None,
                )

            logger.warning(
                "ExternalKnowledgeIntegration | search returned no useful results | duration_ms=%.2f | error=%s",
                elapsed,
                search_response.error,
            )

            return ExternalKnowledgeResult(
                triggered=True,
                success=False,
                query=message.strip(),
                result_count=0,
                context_block="",
                instruction="",
                execution_time_ms=elapsed,
                error=search_response.error or "Search returned no useful results",
            )

        except Exception as exc:
            elapsed = (perf_counter() - started) * 1000
            logger.exception(
                "ExternalKnowledgeIntegration | unexpected error | duration_ms=%.2f",
                elapsed,
            )
            return ExternalKnowledgeResult(
                triggered=True,
                success=False,
                query=message.strip(),
                result_count=0,
                context_block="",
                instruction="",
                execution_time_ms=elapsed,
                error=str(exc),
            )
