from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MAX_EXCHANGES: int = 20


@dataclass
class ConversationSession:
    id: str
    created_at: str
    last_activity: str
    history: list[dict[str, Any]] = field(default_factory=list)


class ConversationMemory:
    def __init__(self, max_exchanges: int = _DEFAULT_MAX_EXCHANGES) -> None:
        self._max_exchanges = max_exchanges
        self._session: ConversationSession = self._new_session()

    def _new_session(self) -> ConversationSession:
        now = datetime.now(tz=timezone.utc).isoformat()
        return ConversationSession(
            id=str(uuid.uuid4()),
            created_at=now,
            last_activity=now,
        )

    def reset_session(self) -> str:
        old_id = self._session.id
        self._session = self._new_session()
        logger.info("Conversation session reset | old=%s | new=%s", old_id, self._session.id)
        return self._session.id

    @property
    def session_id(self) -> str:
        return self._session.id

    @property
    def session_info(self) -> dict[str, Any]:
        return {
            "session_id": self._session.id,
            "created_at": self._session.created_at,
            "last_activity": self._session.last_activity,
            "exchange_count": len(self._session.history) // 2,
        }

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._session.history)

    def add_exchange(self, user_message: str, assistant_response: str) -> None:
        self._session.history.append({
            "role": "user",
            "content": user_message.strip(),
        })
        self._session.history.append({
            "role": "assistant",
            "content": assistant_response.strip(),
        })
        self._session.last_activity = datetime.now(tz=timezone.utc).isoformat()
        if len(self._session.history) > self._max_exchanges * 2:
            self._session.history = self._session.history[-(self._max_exchanges * 2):]

    def get_recent(self, turns: int = 10) -> list[dict[str, Any]]:
        count = min(turns * 2, len(self._session.history))
        return self._session.history[-count:] if count > 0 else []

    def search(self, query: str) -> list[dict[str, Any]]:
        query_lower = query.lower()
        results: list[dict[str, Any]] = []
        for msg in self._session.history:
            content = (msg.get("content") or "").lower()
            if any(word in content for word in query_lower.split() if len(word) > 2):
                results.append(msg)
        return results

    def clear(self) -> None:
        self._session.history.clear()
        self._session.last_activity = datetime.now(tz=timezone.utc).isoformat()
        logger.info("Conversation history cleared")
