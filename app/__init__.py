"""Compatibility package for running the backend from the repository root."""

from backend.app.main import app, create_app

__all__ = ["app", "create_app"]