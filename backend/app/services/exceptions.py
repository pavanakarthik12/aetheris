"""LLM provider exception hierarchy — shared by llm_service and all provider implementations."""

from __future__ import annotations


class LLMServiceError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMQuotaExceeded(LLMServiceError):
    def __init__(self, message: str = "LLM quota exceeded.") -> None:
        super().__init__(message, status_code=402)


class LLMRateLimited(LLMServiceError):
    def __init__(self, message: str = "LLM rate limit hit.") -> None:
        super().__init__(message, status_code=429)


class ProviderBadRequest(LLMServiceError):
    def __init__(self, message: str = "Bad request to LLM provider.") -> None:
        super().__init__(message, status_code=400)


class ProviderUnauthorized(LLMServiceError):
    def __init__(self, message: str = "Unauthorized. Check API key.") -> None:
        super().__init__(message, status_code=401)


class ProviderForbidden(LLMServiceError):
    def __init__(self, message: str = "Forbidden. Access denied.") -> None:
        super().__init__(message, status_code=403)


class ProviderNotFound(LLMServiceError):
    def __init__(self, message: str = "Endpoint not found.") -> None:
        super().__init__(message, status_code=404)


class ProviderConflict(LLMServiceError):
    def __init__(self, message: str = "Conflict error from provider.") -> None:
        super().__init__(message, status_code=409)


class ProviderTimeout(LLMServiceError):
    def __init__(self, message: str = "LLM request timed out.") -> None:
        super().__init__(message, status_code=504)


class ProviderConnectionError(LLMServiceError):
    def __init__(self, message: str = "Unable to reach the LLM API.") -> None:
        super().__init__(message, status_code=502)


class ProviderServerError(LLMServiceError):
    def __init__(self, message: str = "LLM provider returned a server error.") -> None:
        super().__init__(message, status_code=500)


class ProviderUnavailable(LLMServiceError):
    def __init__(self, message: str = "LLM provider is unavailable.") -> None:
        super().__init__(message, status_code=503)


class ProviderMalformedResponse(LLMServiceError):
    def __init__(self, message: str = "LLM returned a malformed response.") -> None:
        super().__init__(message, status_code=502)
