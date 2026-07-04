"""Embedding service using sentence-transformers (BAAI/bge-base-en-v1.5)."""

from __future__ import annotations

import asyncio
import logging
from functools import partial

from ..config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class EmbeddingServiceError(RuntimeError):
    """Raised when the embedding service cannot produce vectors."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class EmbeddingService:
    """Generate dense vector embeddings via sentence-transformers.

    The model is loaded lazily on first use so the service can be
    instantiated cheaply during app startup without blocking.

    Args:
        settings: Optional settings override; falls back to the shared
                  application settings when omitted.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model_name = self._settings.embedding_model
        self._model = None  # loaded lazily

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Return the configured embedding model identifier."""
        return self._model_name

    async def embed_text(self, text: str) -> list[float]:
        """Generate a single embedding vector for *text*.

        Args:
            text: The raw text string to embed.

        Returns:
            A list of floats representing the dense embedding vector.

        Raises:
            EmbeddingServiceError: If the model fails to produce an embedding.
        """
        if not text or not text.strip():
            raise EmbeddingServiceError("Cannot embed an empty string.", status_code=422)

        try:
            model = self._get_model()
            loop = asyncio.get_running_loop()
            fn = partial(model.encode, text, normalize_embeddings=True)
            vector = await loop.run_in_executor(None, fn)
            result = vector.tolist()
            logger.info(
                "Embedding generated | model=%s | dim=%d | text_preview=%.60r",
                self._model_name,
                len(result),
                text,
            )
            return result
        except EmbeddingServiceError:
            raise
        except Exception as exc:
            logger.exception("embed_text failed | model=%s", self._model_name)
            raise EmbeddingServiceError(
                f"Embedding generation failed: {exc}"
            ) from exc

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of documents.

        Args:
            documents: List of raw text strings to embed.

        Returns:
            A list of embedding vectors, one per document.

        Raises:
            EmbeddingServiceError: If any document fails to embed.
        """
        if not documents:
            return []

        try:
            model = self._get_model()
            loop = asyncio.get_running_loop()
            fn = partial(model.encode, documents, normalize_embeddings=True, batch_size=32)
            vectors = await loop.run_in_executor(None, fn)
            return [v.tolist() for v in vectors]
        except EmbeddingServiceError:
            raise
        except Exception as exc:
            logger.exception(
                "embed_documents failed | model=%s | batch_size=%d",
                self._model_name,
                len(documents),
            )
            raise EmbeddingServiceError(
                f"Batch embedding generation failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """Return the loaded SentenceTransformer model, loading it on first call."""
        if self._model is None:
            logger.info("Loading embedding model | model=%s", self._model_name)
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import]

                self._model = SentenceTransformer(self._model_name)
                logger.info("Embedding model loaded | model=%s", self._model_name)
            except ImportError as exc:
                raise EmbeddingServiceError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc
            except Exception as exc:
                raise EmbeddingServiceError(
                    f"Failed to load embedding model '{self._model_name}': {exc}"
                ) from exc
        return self._model
