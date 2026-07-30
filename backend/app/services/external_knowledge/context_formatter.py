"""External Knowledge Context Formatter.

Transforms normalized search results or structured context into a clean
text block for injection into the LLM prompt. Never exposes provider-specific JSON.
"""

from __future__ import annotations

from typing import Any

from ...schemas.context_intelligence import ScoredResultItem, StructuredContext
from ...schemas.external_knowledge import NormalizedSearchResultItem


class ExternalKnowledgeContextFormatter:

    @staticmethod
    def format_results(
        results: list[NormalizedSearchResultItem] | StructuredContext | None = None,
        structured_context: StructuredContext | None = None,
    ) -> str:
        if structured_context is not None:
            return ExternalKnowledgeContextFormatter._format_structured(structured_context)
        if results is None:
            return ""
        return ExternalKnowledgeContextFormatter._format_raw(results)

    @staticmethod
    def _format_raw(
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
    def _format_structured(ctx: StructuredContext) -> str:
        if not ctx.results:
            return ""

        sections: list[str] = []
        for i, item in enumerate(ctx.results, start=1):
            title = item.title.strip() or "Untitled"
            snippet = item.snippet.strip() or ""
            url = item.url.strip() or ""
            domain = item.domain.strip() or ""
            section_parts = [f"Source {i}"]
            section_parts.append(f"Title: {title}")
            if snippet:
                section_parts.append(f"Snippet: {snippet}")
            if url:
                section_parts.append(f"URL: {url}")
            if domain:
                section_parts.append(f"Domain: {domain}")
            sections.append("\n".join(section_parts))

        header = "External Knowledge:\n"
        if ctx.original_count > 0:
            header += (
                f"(Filtered from {ctx.original_count} sources — "
                f"{ctx.duplicates_removed} duplicates removed, "
                f"{ctx.filtered_count} low-quality results filtered)\n"
            )
        return header + "\n".join(sections)

    @staticmethod
    def format_instruction() -> str:
        return (
            "When external knowledge is provided above, prioritize the retrieved information. "
            "Combine it with your own reasoning. "
            "If the search results are insufficient to answer the question, admit uncertainty. "
            "Never fabricate facts that are absent from both memory and the search results."
        )
