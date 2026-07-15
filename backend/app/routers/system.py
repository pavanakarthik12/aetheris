from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ..dependencies import get_llm_service, get_request_router
from ..services.llm_service import LLMService
from ..services.request_router import CognitiveRequestRouter

router = APIRouter(tags=["system"])
logger = logging.getLogger(__name__)


@router.get("/api/system/performance")
async def system_performance(
    router_service: CognitiveRequestRouter = Depends(get_request_router),
) -> dict:
    metrics = router_service.get_metrics_snapshot()
    logger.info("Performance metrics requested | %s", metrics)
    return metrics


@router.get("/api/system/provider-status")
async def provider_status(
    llm_service: LLMService = Depends(get_llm_service),
) -> dict:
    pm = llm_service.provider_manager
    providers = pm.all_snapshots()

    result = {
        "active_provider": pm.active_provider_name,
        "active_model": pm.active_model_name,
        "fallover_active": pm.fallback_used,
        "providers": providers,
    }

    logger.info(
        "Provider status requested | active=%s | fallback=%s | provider_count=%d",
        result["active_provider"], result["fallover_active"], len(providers),
    )
    return result


@router.get("/api/system/providers")
async def system_providers(
    llm_service: LLMService = Depends(get_llm_service),
) -> list[dict]:
    """Return per-provider health and circuit breaker status.

    Each entry includes provider name, model, health status, circuit state,
    average latency, last success/failure timestamps, and request counts.
    """
    pm = llm_service.provider_manager
    providers = pm.all_snapshots()

    logger.info(
        "Providers health requested | count=%d | active=%s",
        len(providers), pm.active_provider_name,
    )
    return providers
