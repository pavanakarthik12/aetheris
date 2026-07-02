"""Pydantic schema package for request and response models."""

from .common import ErrorResponse, HealthResponse

__all__ = ["ErrorResponse", "HealthResponse"]