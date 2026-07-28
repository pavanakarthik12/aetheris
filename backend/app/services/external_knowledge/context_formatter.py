"""External Knowledge Context Formatter.

Transforms normalized search results into a clean text block
for injection into the LLM prompt. Never exposes provider-specific JSON.
"""

from __future__ import annotations

from typing import Any

from ...schemas.external_knowledge import NormalizedSearchResultItem


class ExternalKnowledgeContextFormatter:

    @staticmethod
    def format_results(
        results: list[NormalizedSearchResultItem],
    ) -> str:
        if not results:
            return ""

        sections: list[str] = []
        for i, item in enumerate(results, start=1):
            title = item.title.strip() or "Untitled"
            snippet = item.snippet.strip() or ""
            url = item.url.strip() or ""
            section_parts = [f"Source {i}"]
            section_parts.append(f"Title: {title}")
            if snippet:
                section_parts.append(f"Snippet: {snippet}")
            if url:
                section_parts.append(f"URL: {url}")
            sections.append("\n".join(section_parts))

        return "External Knowledge:\n\n" + "\n\n---\n\n".join(sections)

    @staticmethod
    def format_instruction() -> str:
        return (
            "When external knowledge is provided above, prioritize the retrieved information. "
            "Combine it with your own reasoning. "
            "If the search results are insufficient to answer the question, admit uncertainty. "
            "Never fabricate facts that are absent from both memory and the search results."
        )
