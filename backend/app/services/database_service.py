"""SQLAlchemy database service for PostgreSQL connectivity."""

from collections.abc import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..config.settings import Settings, get_settings


class DatabaseService:
    """Create and expose SQLAlchemy engine and session factory objects."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None

    @property
    def engine(self) -> Engine:
        """Return the configured SQLAlchemy engine."""

        if self._engine is None:
            if not self._settings.database_url:
                raise RuntimeError("DATABASE_URL is not configured.")
            self._engine = create_engine(self._settings.database_url, pool_pre_ping=True)
        return self._engine

    @property
    def session_factory(self) -> sessionmaker[Session]:
        """Return the configured SQLAlchemy session factory."""

        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autoflush=False,
                autocommit=False,
                expire_on_commit=False,
            )
        return self._session_factory

    def session_scope(self) -> Iterator[Session]:
        """Yield a SQLAlchemy session for dependency injection."""

        session = self.session_factory()
        try:
            yield session
        finally:
            session.close()