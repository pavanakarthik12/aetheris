"""Memory Migration Script — convert all existing first-person memories to third-person user facts.

Usage:
    python -m backend.scripts.migrate_memories [--dry-run] [--force]

This script:
  1. Reads all existing memories from ChromaDB.
  2. Detects first-person text (I, my, me, we, our).
  3. Converts to third-person "The user..." format.
  4. Updates metadata with structured fact data.
  5. Re-embeds the normalized text.
  6. Preserves all existing metadata.

Run with --dry-run first to see what would change.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from backend.app.config.settings import get_settings
from backend.app.services.chroma_service import ChromaService
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.memory_normalizer import (
    is_first_person,
    normalize_to_third_person,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def migrate_all(dry_run: bool = False, force: bool = False) -> None:
    """Migrate all first-person memories to third-person user facts."""
    settings = get_settings()
    chroma = ChromaService(settings)
    embedding_service = EmbeddingService(settings)

    # Read all existing memories
    all_memories = chroma.list_all_memories()
    logger.info("Found %d total memories", len(all_memories))

    migrated_count = 0
    skipped_count = 0
    error_count = 0

    for memory in all_memories:
        memory_id = memory.get("id", "")
        document = memory.get("document", "")
        metadata = memory.get("metadata", {})

        if not document or not document.strip():
            skipped_count += 1
            continue

        # Check if already in third-person
        if not is_first_person(document):
            # Already third-person; check if metadata has fact data
            if "fact" not in metadata and "memory_fact" not in metadata:
                # Add fact metadata even for already-correct memories
                if not dry_run and force:
                    _add_fact_metadata(metadata, document)
                    try:
                        chroma.update_memory(
                            memory_id=memory_id,
                            embedding=metadata.get("_embedding"),
                            document=document,
                            metadata=metadata,
                        )
                        logger.info(
                            "Added fact metadata | id=%s | document=%.60r",
                            memory_id, document,
                        )
                    except Exception as exc:
                        logger.error("Failed to update metadata | id=%s | error=%s", memory_id, exc)
                        error_count += 1
            skipped_count += 1
            continue

        # Convert to third-person
        category = metadata.get("category", "")
        new_document = normalize_to_third_person(document, category=category)
        if new_document == document:
            logger.info(
                "No change needed | id=%s | document=%.60r",
                memory_id, document,
            )
            skipped_count += 1
            continue

        logger.info(
            "Migrating memory | id=%s\n  OLD: %.80r\n  NEW: %.80r",
            memory_id, document, new_document,
        )

        if dry_run:
            migrated_count += 1
            continue

        # Generate embedding for new document
        try:
            new_embedding = embedding_service.embed_text(new_document)
        except Exception as exc:
            logger.error(
                "Embedding failed | id=%s | error=%s", memory_id, exc,
            )
            error_count += 1
            continue

        # Build updated metadata with fact data
        new_metadata = dict(metadata)
        new_metadata["original_text"] = document
        new_metadata["memory_fact"] = new_document
        _add_fact_metadata(new_metadata, new_document, category)

        # Persist
        try:
            chroma.update_memory(
                memory_id=memory_id,
                embedding=new_embedding,
                document=new_document,
                metadata=new_metadata,
            )
            migrated_count += 1
            logger.info(
                "Migrated successfully | id=%s | fact=%.60r",
                memory_id, new_document,
            )
        except Exception as exc:
            logger.error(
                "Failed to persist migration | id=%s | error=%s",
                memory_id, exc,
            )
            error_count += 1

    # Summary
    logger.info("=" * 60)
    logger.info("Migration complete")
    logger.info("  Total memories:  %d", len(all_memories))
    logger.info("  Migrated:        %d", migrated_count)
    logger.info("  Skipped:         %d", skipped_count)
    logger.info("  Errors:          %d", error_count)
    if dry_run:
        logger.info("  (dry-run — no changes were made)")


def _add_fact_metadata(
    metadata: dict[str, Any],
    document: str,
    category: str = "",
) -> None:
    """Add structured fact data to metadata based on document text."""
    if "fact" not in metadata:
        # Try to extract attribute and value from the third-person text
        attribute = _extract_attribute_from_fact(document)
        value = _extract_value_from_fact(document)
        metadata["fact"] = {
            "category": category.lower() if category else "",
            "attribute": attribute,
            "value": value,
            "memory_fact": document,
        }


def _extract_attribute_from_fact(text: str) -> str:
    """Extract a machine-readable attribute key from a user fact sentence."""
    lower = text.lower()
    patterns: list[tuple[str, str]] = [
        (r"preferred programming language", "favorite_programming_language"),
        (r"favorite (?:programming )?language", "favorite_programming_language"),
        (r"favorite food", "favorite_food"),
        (r"favorite subject", "favorite_subject"),
        (r"current name", "name"),
        (r"'s name is", "name"),
        (r"building a project", "current_project"),
        (r"'s goal is", "goal"),
        (r"learning ", "skill"),
        (r"lives in", "city"),
        (r"occupation is", "occupation"),
        (r"age is", "age"),
        (r"prefers ", "preference"),
        (r"likes ", "preference"),
    ]
    for pattern, attr in patterns:
        if pattern in lower:
            return attr
    return ""


def _extract_value_from_fact(text: str) -> str:
    """Extract the value portion from a user fact sentence."""
    import re

    # "The user is building a project called X."
    m = re.search(r"called\s+(.+?)[\.!\?]?\s*$", text)
    if m:
        return m.group(1).strip()

    # "The user's X is Y."
    m = re.search(r"is\s+(.+?)[\.!\?]?\s*$", text)
    if m:
        return m.group(1).strip()

    # "The user [verb]s X."
    m = re.search(r"(?:likes|prefers|uses|has|knows|wants|needs)\s+(.+?)[\.!\?]?\s*$", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate first-person memories to third-person user facts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without making changes.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Also add fact metadata to already-third-person memories.",
    )
    args = parser.parse_args()

    logger.info("Starting memory migration (dry_run=%s, force=%s)", args.dry_run, args.force)
    migrate_all(dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
