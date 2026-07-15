from __future__ import annotations

import re


_GREETING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*hi\s*$", re.IGNORECASE),
    re.compile(r"^\s*hello\s*$", re.IGNORECASE),
    re.compile(r"^\s*hey\s*$", re.IGNORECASE),
    re.compile(r"^\s*bye\s*$", re.IGNORECASE),
    re.compile(r"^\s*good\s+morning\s*$", re.IGNORECASE),
    re.compile(r"^\s*good\s+afternoon\s*$", re.IGNORECASE),
    re.compile(r"^\s*good\s+evening\s*$", re.IGNORECASE),
    re.compile(r"^\s*good\s+night\s*$", re.IGNORECASE),
    re.compile(r"^\s*thanks?\s*$", re.IGNORECASE),
    re.compile(r"^\s*thank\s+you\s*$", re.IGNORECASE),
    re.compile(r"^\s*thank\s+you\s*!*\s*$", re.IGNORECASE),
]

_GREETING_RESPONSES: dict[str, str] = {
    "hi": "Hi there! How can I help you today?",
    "hello": "Hello! How can I assist you?",
    "hey": "Hey! What can I do for you?",
    "good morning": "Good morning! How can I help you today?",
    "good afternoon": "Good afternoon! How can I assist you?",
    "good evening": "Good evening! How can I help you?",
    "good night": "Good night! Take care.",
    "bye": "Goodbye! Have a great day!",
    "thanks": "You're welcome! Let me know if you need anything else.",
    "thank you": "You're welcome! Happy to help.",
}

_FALLBACK_RESPONSE = "Hello! How can I assist you today?"


def detect_greeting(message: str) -> str | None:
    stripped = message.strip()
    if not stripped:
        return None
    for pattern in _GREETING_PATTERNS:
        match = pattern.match(stripped)
        if match:
            key = match.group(0).strip().lower()
            return _GREETING_RESPONSES.get(key, _FALLBACK_RESPONSE)
    return None
