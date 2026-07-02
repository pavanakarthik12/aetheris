"""ChromaDB service boundary for future vector storage operations."""

from pathlib import Path
from typing import Any

from app.config.settings import Settings, get_settings


class ChromaService:
    """Encapsulate ChromaDB initialization and persistence boundaries."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._storage_path = Path(self._settings.chroma_db_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)

    @property
    def storage_path(self) -> Path:
        """Return the filesystem location used for ChromaDB persistence."""

        return self._storage_path

    def add_memory(self, *args: Any, **kwargs: Any) -> None:
        """Placeholder for adding vectorized memory records."""

        raise NotImplementedError("ChromaDB memory writes are not implemented yet.")

    def search_memory(self, *args: Any, **kwargs: Any) -> list[Any]:
        """Placeholder for searching vectorized memory records."""

        raise NotImplementedError("ChromaDB search is not implemented yet.")

    def delete_memory(self, *args: Any, **kwargs: Any) -> None:
        """Placeholder for deleting vectorized memory records."""

        raise NotImplementedError("ChromaDB deletion is not implemented yet.")