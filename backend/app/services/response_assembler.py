from __future__ import annotations

import logging
import re
from time import perf_counter

from ..schemas.context_intelligence import ScoredResultItem, StructuredContext
from ..schemas.response import AssembledResponse, QualityReport, SourceCitation

logger = logging.getLogger(__name__)

_DONT_KNOW_PHRASES: list[str] = [
    "i don't know",
    "i don't have",
    "i'm not sure",
    "i am not sure",
    "i cannot",
    "i can't",
    "no information",
    "is not available",
    "unable to",
    "cannot provide",
    "i have no",
    "i do not know",
    "i do not have",
    "i'm unable",
    "i am unable",
]

_CODE_KEYWORDS: list[str] = [
    "code", "function", "class ", "def ", "example",
    "implement", "write a", "syntax", "error",
    "bug", "script", "program", "api", "endpoint",
    "route", "component", "hook", "function ",
]

_SOURCES_DIVIDER: str = "---\n\n**Sources:**"


class ResponseAssembler:

    def assemble(
        self,
        raw_response: str,
        structured_context: StructuredContext | None = None,
        user_message: str = "",
    ) -> AssembledResponse:
        started = perf_counter()

        quality = self._evaluate_quality(raw_response, user_message, structured_context)
        sources = self._extract_sources(structured_context)
        final_response = self._append_sources(raw_response, sources)

        elapsed = (perf_counter() - started) * 1000

        logger.info(
            "ResponseAssembler | quality_score=%.2f | quality_passed=%s | sources=%d | issues=%d | %.2fms",
            quality.score,
            quality.passed,
            len(sources),
            len(quality.issues),
            elapsed,
        )
        if quality.issues:
            for issue in quality.issues:
                logger.warning("ResponseAssembler | quality_issue=%s", issue)

        return AssembledResponse(
            response=final_response,
            sources=sources,
            quality=quality,
            has_sources=bool(sources),
            assembly_time_ms=elapsed,
        )

    def _evaluate_quality(
        self,
        response: str,
        user_message: str,
        ctx: StructuredContext | None,
    ) -> QualityReport:
        checks: dict[str, bool] = {}
        issues: list[str] = []
        stripped = response.strip()

        non_empty = bool(stripped)
        checks["non_empty"] = non_empty
        if not non_empty:
            issues.append("Response is empty")

        sufficient_length = len(stripped) >= 10
        checks["sufficient_length"] = sufficient_length
        if not sufficient_length:
            issues.append("Response is too short (under 10 characters)")

        has_sources = ctx is not None and len(ctx.results) > 0
        sources_available = len(ctx.results) if ctx else 0

        if has_sources:
            lowered = response.lower()
            mentioned_dont_know = any(p in lowered for p in _DONT_KNOW_PHRASES)
            checks["no_false_ignorance"] = not mentioned_dont_know
            if mentioned_dont_know:
                issues.append(
                    "Response claims ignorance despite having sources available"
                )
        else:
            checks["no_false_ignorance"] = True

        is_code_query = any(kw in user_message.lower() for kw in _CODE_KEYWORDS)
        has_code = "```" in response
        if is_code_query:
            checks["code_blocks_present"] = has_code
            if not has_code:
                issues.append("Code-related query but response has no code blocks")
        else:
            checks["code_blocks_present"] = True

        has_urls = bool(re.search(r'https?://[^\s]+', response))
        checks["has_urls"] = has_urls

        blocker_checks = {"non_empty", "sufficient_length"}
        if has_sources:
            blocker_checks.add("no_false_ignorance")

        all_blockers_pass = all(checks.get(c, True) for c in blocker_checks)
        passed_checks = sum(1 for v in checks.values() if v)
        total_checks = len(checks)
        score = (passed_checks / total_checks) if total_checks > 0 else 1.0
        passed = all_blockers_pass

        return QualityReport(
            passed=passed,
            score=score,
            issues=issues,
            response_length=len(stripped),
            has_code_blocks=has_code,
            has_urls=has_urls,
            check_details=checks,
            sources_available=sources_available,
        )

    def _extract_sources(
        self,
        ctx: StructuredContext | None,
    ) -> list[SourceCitation]:
        if ctx is None:
            return []
        sources: list[SourceCitation] = []
        seen_urls: set[str] = set()
        for item in ctx.results:
            if item.url and item.url not in seen_urls:
                seen_urls.add(item.url)
                sources.append(SourceCitation(
                    title=item.title or item.url,
                    url=item.url,
                    domain=item.domain,
                    confidence=item.confidence,
                    trust_score=item.trust_score,
                    position=len(sources) + 1,
                ))
        return sources

    def _append_sources(
        self,
        response: str,
        sources: list[SourceCitation],
    ) -> str:
        if not sources:
            return response

        if _SOURCES_DIVIDER in response:
            return response

        parts = ["---", "", "**Sources:**"]
        for src in sources:
            parts.append(f"{src.position}. [{src.title}]({src.url})")

        return f"{response.strip()}\n\n" + "\n".join(parts)
