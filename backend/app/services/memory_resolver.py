"""Memory Resolution Engine — resolves conflicts for single-value attributes.

Every attribute listed in ``SINGLE_VALUE_ATTRIBUTES`` must have at most one
active memory at any time.  When a new value arrives, the previous active
memory is automatically archived and the new one becomes active.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Single-value attributes
# ---------------------------------------------------------------------------

SINGLE_VALUE_ATTRIBUTES: frozenset[str] = frozenset({
    "name",
    "age",
    "birthday",
    "favorite_programming_language",
    "favorite_language",
    "favorite_subject",
    "favorite_food",
    "city",
    "country",
    "occupation",
    "school",
    "university",
    "company",
    "relationship_status",
})

# ---------------------------------------------------------------------------
# Attribute extraction patterns
# ---------------------------------------------------------------------------

_ATTRIBUTE_PATTERNS: dict[str, re.Pattern[str]] = {
    "name": re.compile(
        r"\bmy name is\s+(.+?)(?:\.|,|$)", re.IGNORECASE,
    ),
    "age": re.compile(
        r"\bi\s*(?:am|'m)\s+(\d+)\s*(?:years?\s*old|yo)\b", re.IGNORECASE,
    ),
    "birthday": re.compile(
        r"\b(?:my birthday is|i was born on)\s+(.+?)(?:\.|,|$)", re.IGNORECASE,
    ),
    "favorite_programming_language": re.compile(
        r"\bfavorite\s+(?:programming\s+)?language is\s+(.+?)(?:\.|,|$)",
        re.IGNORECASE,
    ),
    "favorite_subject": re.compile(
        r"\bfavorite\s+subject is\s+(.+?)(?:\.|,|$)", re.IGNORECASE,
    ),
    "favorite_food": re.compile(
        r"\bfavorite\s+food is\s+(.+?)(?:\.|,|$)", re.IGNORECASE,
    ),
    "city": re.compile(
        r"\bmy city is\s+(.+?)(?:\.|,|$)", re.IGNORECASE,
    ),
    "country": re.compile(
        r"\bmy country is\s+(.+?)(?:\.|,|$)", re.IGNORECASE,
    ),
    "occupation": re.compile(
        r"\b(?:i work as|my occupation is)\s+(.+?)(?:\.|,|$)", re.IGNORECASE,
    ),
    "school": re.compile(
        r"\b(?:my school is|i go to)\s+(.+?)(?:\.|,|$)", re.IGNORECASE,
    ),
    "university": re.compile(
        r"\b(?:my university is|i study at)\s+(.+?)(?:\.|,|$)", re.IGNORECASE,
    ),
    "company": re.compile(
        r"\b(?:my company is|i work at)\s+(.+?)(?:\.|,|$)", re.IGNORECASE,
    ),
    "relationship_status": re.compile(
        r"\b(?:i am|my relationship status is)\s+"
        r"(single|married|engaged|divorced|dating|widowed)\b",
        re.IGNORECASE,
    ),
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_attribute(text: str) -> str | None:
    """Return the single-value attribute name detected in *text*, or *None*.

    Example::

        >>> extract_attribute("My name is John Paul")
        "name"
        >>> extract_attribute("What is the weather?")
        None
    """
    for attr, pattern in _ATTRIBUTE_PATTERNS.items():
        if pattern.search(text):
            return attr
    return None


async def resolve_conflict(
    memory_text: str,
    chroma_service: Any,
    archive_fn: Any,
) -> dict[str, Any]:
    """Detect and resolve single-value attribute conflicts.

    Args:
        memory_text:  The incoming memory text about to be stored.
        chroma_service:  A ``ChromaService`` instance for metadata lookups.
        archive_fn:   An async callable ``archive_fn(memory_id)`` that
                      archives a memory (e.g. ``MemoryEvolutionService.archive_memory``).

    Returns:
        A dict with keys:
        - ``attribute`` (str or None) — the detected attribute, if any.
        - ``conflict_detected`` (bool) — whether an active conflict was found.
        - ``archived_memory_id`` (str or None) — the ID of the archived memory.
        - ``verified`` (bool) — *True* if post-resolution verification passed.
        - ``error`` (str or None) — error message if resolution failed.
    """
    attribute = extract_attribute(memory_text)
    if attribute is None or attribute not in SINGLE_VALUE_ATTRIBUTES:
        return {
            "attribute": attribute,
            "conflict_detected": False,
            "archived_memory_id": None,
            "verified": True,
            "error": None,
        }

    try:
        existing = chroma_service.get_memories_by_metadata(
            where={"attribute": attribute, "status": "active"},
            limit=10,
        )
    except Exception as exc:
        logger.exception("Failed to search for existing attribute=%s", attribute)
        return {
            "attribute": attribute,
            "conflict_detected": False,
            "archived_memory_id": None,
            "verified": False,
            "error": f"Search failed: {exc}",
        }

    if not existing:
        return {
            "attribute": attribute,
            "conflict_detected": False,
            "archived_memory_id": None,
            "verified": True,
            "error": None,
        }

    # Conflict detected — archive the previous active memory (only the first)
    prev = existing[0]
    prev_id = prev["id"]
    prev_text = prev.get("document", "")

    logger.info(
        "Conflict Detected | attribute=%s | previous_id=%s | previous_text=%.80r",
        attribute, prev_id, prev_text,
    )
    logger.info("Previous Active Memory | id=%s | text=%.80r", prev_id, prev_text)

    try:
        await archive_fn(prev_id)
    except Exception as exc:
        logger.exception(
            "Failed to archive previous memory | id=%s | attribute=%s",
            prev_id, attribute,
        )
        return {
            "attribute": attribute,
            "conflict_detected": True,
            "archived_memory_id": prev_id,
            "verified": False,
            "error": f"Archive failed: {exc}",
        }

    logger.info("Archived Memory | id=%s | attribute=%s", prev_id, attribute)

    # Verification: confirm the old memory is archived
    try:
        recheck = chroma_service.get_memories_by_metadata(
            where={"attribute": attribute, "status": "active"},
            limit=10,
        )
    except Exception as exc:
        logger.warning("Verification lookup failed | attribute=%s | error=%s", attribute, exc)
        return {
            "attribute": attribute,
            "conflict_detected": True,
            "archived_memory_id": prev_id,
            "verified": False,
            "error": f"Verification search failed: {exc}",
        }

    if recheck:
        still_active = [m for m in recheck if m["id"] != prev_id]
        if still_active:
            logger.error(
                "Verification Failed | attribute=%s | remaining_active=%d",
                attribute, len(still_active),
            )
            return {
                "attribute": attribute,
                "conflict_detected": True,
                "archived_memory_id": prev_id,
                "verified": False,
                "error": "Stale active memories remain after archiving.",
            }

    logger.info(
        "Verification Passed | attribute=%s | archived=%s",
        attribute, prev_id,
    )

    return {
        "attribute": attribute,
        "conflict_detected": True,
        "archived_memory_id": prev_id,
        "verified": True,
        "error": None,
    }
