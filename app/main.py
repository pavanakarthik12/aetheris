"""Compatibility entrypoint for Uvicorn.

This keeps the existing backend implementation in backend.app.main while
allowing the repository-root command `uvicorn app.main:app` to work.
"""

from backend.app.main import app, create_app

__all__ = ["app", "create_app"]