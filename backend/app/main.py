"""FastAPI application entrypoint for Aetheris."""

import logging
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

from .config.settings import get_settings
from .dependencies import get_embedding_service, get_llm_service
from .middleware.error_handlers import register_exception_handlers
from .middleware.request_id import RequestIdMiddleware
from .routers.chat import router as chat_router
from .routers.context_debug import router as context_debug_router
from .routers.health import router as health_router
from .routers.memory import router as memory_router
from .routers.memory_evolution import router as memory_evolution_router
from .routers.reflection import router as reflection_router
from .routers.system import router as system_router
from .utils.logging import configure_logging

_LOCALHOST_ORIGIN_RE = re.compile(
    r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
)

_PRODUCTION_ALLOWED_ORIGINS: list[str] = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def _is_development() -> bool:
    return os.getenv("ENVIRONMENT", "development").lower() not in ("production", "prod")


def _configure_cors(
    application: FastAPI,
) -> None:
    if _is_development():
        logger.info(
            "CORS | environment=development | strategy=allow_origin_regex | "
            "pattern=%s",
            _LOCALHOST_ORIGIN_RE.pattern,
        )
        application.add_middleware(
            CORSMiddleware,
            allow_origin_regex=_LOCALHOST_ORIGIN_RE.pattern,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        logger.info(
            "CORS | environment=production | strategy=allow_origins | "
            "origins=%s",
            _PRODUCTION_ALLOWED_ORIGINS,
        )
        application.add_middleware(
            CORSMiddleware,
            allow_origins=_PRODUCTION_ALLOWED_ORIGINS,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    logger.info("CORS | initialized successfully")


@asynccontextmanager
async def lifespan(application: FastAPI):
    try:
        emb = get_embedding_service()
        await emb.embed_text("warmup")
        logger.info("Embedding model pre-loaded on startup")
    except Exception as exc:
        logger.warning("Embedding model pre-load failed | error=%s", exc)
    yield
    await get_llm_service().aclose()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""

    settings = get_settings()
    configure_logging(settings.log_level)

    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    application.add_middleware(RequestIdMiddleware)
    _configure_cors(application)
    register_exception_handlers(application)
    application.include_router(health_router, prefix=settings.api_v1_prefix)
    application.include_router(chat_router)
    application.include_router(context_debug_router)
    application.include_router(memory_router)
    application.include_router(memory_evolution_router)
    application.include_router(reflection_router)
    application.include_router(system_router)

    return application


app = create_app()