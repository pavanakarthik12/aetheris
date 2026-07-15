"""Service package exposing application boundaries."""

from .exceptions import (
    LLMQuotaExceeded,
    LLMRateLimited,
    LLMServiceError,
    ProviderBadRequest,
    ProviderConflict,
    ProviderConnectionError,
    ProviderForbidden,
    ProviderMalformedResponse,
    ProviderNotFound,
    ProviderServerError,
    ProviderTimeout,
    ProviderUnauthorized,
    ProviderUnavailable,
)
from .llm_service import LLMService

__all__ = [
    "LLMService",
    "LLMServiceError",
    "LLMQuotaExceeded",
    "LLMRateLimited",
    "ProviderBadRequest",
    "ProviderUnauthorized",
    "ProviderForbidden",
    "ProviderNotFound",
    "ProviderConflict",
    "ProviderTimeout",
    "ProviderConnectionError",
    "ProviderServerError",
    "ProviderUnavailable",
    "ProviderMalformedResponse",
]
