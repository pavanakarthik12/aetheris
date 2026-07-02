"""FastAPI application entrypoint for Aetheris."""

from fastapi import FastAPI

from app.config.settings import get_settings
from app.middleware.error_handlers import register_exception_handlers
from app.middleware.request_id import RequestIdMiddleware
from app.routers.health import router as health_router
from app.utils.logging import configure_logging


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""

    settings = get_settings()
    configure_logging(settings.log_level)

    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    application.add_middleware(RequestIdMiddleware)
    register_exception_handlers(application)
    application.include_router(health_router, prefix=settings.api_v1_prefix)

    return application


app = create_app()