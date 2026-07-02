"""Isolated LLM service boundary for all future Qwen communication."""

from typing import Any

from app.config.settings import Settings, get_settings


class LLMService:
    """Encapsulate provider selection, model identity, and future Qwen access."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def provider(self) -> str:
        """Return the configured LLM provider name."""

        return self._settings.llm_provider

    @property
    def model_name(self) -> str:
        """Return the configured LLM model name."""

        return self._settings.llm_model

    @property
    def api_key(self) -> str:
        """Return the configured provider API key."""

        return self._settings.qwen_api_key

    def generate_text(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        """Placeholder for text generation calls."""

        raise NotImplementedError("LLM generation is not implemented yet.")

    def stream_text(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
        """Placeholder for streaming text generation calls."""

        raise NotImplementedError("LLM streaming is not implemented yet.")