"""FastAPI application entrypoint for Aetheris."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config.settings import get_settings
from .dependencies import get_llm_service
from .middleware.error_handlers import register_exception_handlers
from .middleware.request_id import RequestIdMiddleware
from .routers.chat import router as chat_router
from .routers.health import router as health_router
from .routers.memory import router as memory_router
from .utils.logging import configure_logging


@asynccontextmanager
async def lifespan(application: FastAPI):
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
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(application)
    application.include_router(health_router, prefix=settings.api_v1_prefix)
    application.include_router(chat_router)
    application.include_router(memory_router)

    return application


app = create_app()