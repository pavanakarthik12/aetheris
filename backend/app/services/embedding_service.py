"""Embedding service boundary for future vector generation work."""

from typing import Any

from app.config.settings import Settings, get_settings


class EmbeddingService:
    """Expose the embedding model boundary without implementing business logic."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model_name = self._settings.embedding_model

    @property
    def model_name(self) -> str:
        """Return the configured embedding model identifier."""

        return self._model_name

    def embed_text(self, text: str, *args: Any, **kwargs: Any) -> list[float]:
        """Placeholder for single-text embedding generation."""

        raise NotImplementedError("Embedding generation is not implemented yet.")

    def embed_documents(self, documents: list[str], *args: Any, **kwargs: Any) -> list[list[float]]:
        """Placeholder for batch embedding generation."""

        raise NotImplementedError("Embedding generation is not implemented yet.")