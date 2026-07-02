"""Centralized logging configuration for the backend application."""

from __future__ import annotations

import logging
from logging.config import dictConfig


def configure_logging(log_level: str = "INFO") -> None:
    """Configure a shared logging format for the application."""

    level = getattr(logging, log_level.upper(), logging.INFO)

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                }
            },
            "root": {
                "level": level,
                "handlers": ["console"],
            },
        }
    )