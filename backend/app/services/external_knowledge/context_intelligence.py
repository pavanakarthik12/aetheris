"""Context Intelligence Engine — transforms raw search results into structured context.

Pipeline:
  1. Normalize → 2. Remove invalid → 3. Remove duplicates → 4. Trust score
  → 5. Relevance rank → 6. Compress content → 7. Enforce token budget
  → 8. Produce structured context

This module never calls the LLM, never modifies memories, and never performs reasoning.
"""

from __future__ import annotations

import logging
import re
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

from ...schemas.context_intelligence import ScoredResultItem, StructuredContext, TrustConfig
from ...schemas.external_knowledge import NormalizedSearchResultItem

logger = logging.getLogger(__name__)

_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(r"cookie[-\s]?(policy|notice|consent)", re.IGNORECASE),
    re.compile(r"accept (all|cookies)", re.IGNORECASE),
    re.compile(r"sign[-\s]?(up|in|out)", re.IGNORECASE),
    re.compile(r"subscribe to our", re.IGNORECASE),
    re.compile(r"advertisement", re.IGNORECASE),
    re.compile(r"click here", re.IGNORECASE),
    re.compile(r"privacy policy", re.IGNORECASE),
    re.compile(r"terms of (service|use)", re.IGNORECASE),
    re.compile(r"all rights reserved", re.IGNORECASE),
    re.compile(r"©", re.IGNORECASE),
    re.compile(r"brought to you by", re.IGNORECASE),
    re.compile(r"sponsored (by|content)", re.IGNORECASE),
]

_CLICKBAIT_PATTERNS: list[re.Pattern] = [
    re.compile(r"you won't believe", re.IGNORECASE),
    re.compile(r"mind[-\s]?blowing", re.IGNORECASE),
    re.compile(r"shocked|shocking", re.IGNORECASE),
    re.compile(r"this will blow your mind", re.IGNORECASE),
    re.compile(r"number \d+ will surprise you", re.IGNORECASE),
    re.compile(r"what happened next", re.IGNORECASE),
    re.compile(r"the truth about", re.IGNORECASE),
]


def _extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""


def _estimate_tokens(text: str) -> int:
    return max(0, len(text) // 4)


class ContextIntelligenceEngine:

    def __init__(self, config: TrustConfig | None = None) -> None:
        self._config = config or TrustConfig()

    def process(
        self,
        query: str,
        results: list[NormalizedSearchResultItem],
    ) -> StructuredContext:
        started = perf_counter()
        original_count = len(results)

        if not results:
            elapsed = (perf_counter() - started) * 1000
            logger.info("CII | no results to process | query=%.50s", query)
            return StructuredContext(
                query=query,
                original_count=0,
                processing_time_ms=elapsed,
            )

        valid, filtered_count = self._remove_invalid(results)
        deduped, duplicates_removed = self._remove_duplicates(valid)
        scored = self._assign_trust_scores(deduped)
        compressed = self._compress_all(scored)
        ranked = self._rank_by_relevance(compressed, query)
        capped = self._enforce_max_results(ranked)
        capped_and_budgeted, estimated_tokens = self._enforce_token_budget(capped)

        elapsed = (perf_counter() - started) * 1000

        logger.info(
            "CII | query=%.50s | original=%d | invalid=%d | dups=%d | final=%d | "
            "tokens=%d | duration_ms=%.2f",
            query,
            original_count,
            filtered_count,
            duplicates_removed,
            len(capped_and_budgeted),
            estimated_tokens,
            elapsed,
        )

        return StructuredContext(
            query=query,
            results=capped_and_budgeted,
            estimated_tokens=estimated_tokens,
            sources_used=len(capped_and_budgeted),
            original_count=original_count,
            duplicates_removed=duplicates_removed,
            filtered_count=filtered_count,
            processing_time_ms=round(elapsed, 2),
        )

    def _remove_invalid(
        self,
        results: list[NormalizedSearchResultItem],
    ) -> tuple[list[ScoredResultItem], int]:
        valid: list[ScoredResultItem] = []
        filtered = 0
        for item in results:
            title = (item.title or "").strip()
            snippet = (item.snippet or "").strip()
            url = (item.url or "").strip()
            if not title and not snippet:
                filtered += 1
                continue
            if not url:
                filtered += 1
                continue
            if item.score < self._config.min_confidence:
                filtered += 1
                continue
            if len(title) < 2 and len(snippet) < 10:
                filtered += 1
                continue
            valid.append(ScoredResultItem(
                title=title,
                snippet=snippet,
                url=url,
                domain=_extract_domain(url),
                confidence=item.score,
                source=item.source or "",
            ))
        return valid, filtered

    def _remove_duplicates(
        self,
        items: list[ScoredResultItem],
    ) -> tuple[list[ScoredResultItem], int]:
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()
        seen_snippets: set[str] = set()
        deduped: list[ScoredResultItem] = []
        removed = 0

        for item in items:
            url_key = item.url.strip().lower()
            title_key = item.title.strip().lower()
            snippet_key = item.snippet.strip().lower()[:80]

            if url_key and url_key in seen_urls:
                removed += 1
                continue
            if title_key and title_key in seen_titles:
                removed += 1
                item.is_duplicate = True
                continue
            if snippet_key and snippet_key in seen_snippets:
                removed += 1
                item.is_duplicate = True
                continue

            if url_key:
                seen_urls.add(url_key)
            if title_key:
                seen_titles.add(title_key)
            if snippet_key:
                seen_snippets.add(snippet_key)

            item.is_duplicate = False
            deduped.append(item)

        return deduped, removed

    def _assign_trust_scores(
        self,
        items: list[ScoredResultItem],
    ) -> list[ScoredResultItem]:
        for item in items:
            score = self._score_trust(item)
            item.trust_score = score
        return items

    def _score_trust(self, item: ScoredResultItem) -> float:
        domain = item.domain
        score = self._config.default_trust

        for pattern, trusted_score in self._config.trusted_domains.items():
            if pattern.startswith("."):
                if domain.endswith(pattern) or domain.endswith(pattern.lstrip(".") + "."):
                    score = max(score, trusted_score)
            elif pattern in domain:
                score = max(score, trusted_score)

        desc = (item.title + " " + item.snippet).lower()
        clickbait_penalty = 0.0
        for pat in _CLICKBAIT_PATTERNS:
            if pat.search(desc):
                clickbait_penalty = max(clickbait_penalty, 0.3)
        score -= clickbait_penalty

        if item.confidence < 0.3:
            score -= 0.15
        elif item.confidence > 0.8:
            score += 0.05

        return max(self._config.min_trust, min(self._config.max_trust, score))

    def _compress_all(
        self,
        items: list[ScoredResultItem],
    ) -> list[ScoredResultItem]:
        for item in items:
            item.snippet = self._compress_text(item.snippet)
        return items

    @staticmethod
    def _compress_text(text: str) -> str:
        if not text:
            return ""
        lines = text.split("\n")
        cleaned: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if any(p.search(stripped) for p in _BOILERPLATE_PATTERNS):
                continue
            cleaned.append(stripped)
        result = " ".join(cleaned)
        result = re.sub(r"\s+", " ", result).strip()
        return result

    def _rank_by_relevance(
        self,
        items: list[ScoredResultItem],
        query: str,
    ) -> list[ScoredResultItem]:
        query_lower = query.lower().strip()
        query_terms = set(query_lower.split())

        for item in items:
            score = item.trust_score * 0.4 + item.confidence * 0.3
            title_lower = item.title.lower()
            snippet_lower = item.snippet.lower()
            domain_lower = item.domain.lower()

            matches = sum(1 for t in query_terms if t in title_lower or t in snippet_lower)
            if matches > 0:
                score += min(0.3, matches * 0.05)

            if any(t in domain_lower for t in query_terms):
                score += 0.05

            title_len = len(item.title.strip().split())
            if 3 <= title_len <= 20:
                score += 0.05

            snippet_len = len(item.snippet.strip().split())
            if snippet_len >= 10:
                score += 0.05

            item.relevance_score = round(min(1.0, score), 4)

        items.sort(key=lambda x: x.relevance_score, reverse=True)
        return items

    def _enforce_max_results(
        self,
        items: list[ScoredResultItem],
    ) -> list[ScoredResultItem]:
        return items[:self._config.max_results]

    def _enforce_token_budget(
        self,
        items: list[ScoredResultItem],
    ) -> tuple[list[ScoredResultItem], int]:
        result: list[ScoredResultItem] = []
        total = 0
        for item in items:
            text = f"{item.title} {item.snippet} {item.url}"
            tokens = _estimate_tokens(text)
            if total + tokens > self._config.max_tokens:
                break
            total += tokens
            result.append(item)
        return result, total
