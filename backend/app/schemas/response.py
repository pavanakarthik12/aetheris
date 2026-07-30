from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SourceCitation:
    title: str = ""
    url: str = ""
    domain: str = ""
    confidence: float = 0.0
    trust_score: float = 0.0
    position: int = 0


@dataclass
class QualityReport:
    passed: bool = True
    score: float = 1.0
    issues: list[str] = field(default_factory=list)
    response_length: int = 0
    has_code_blocks: bool = False
    has_urls: bool = False
    check_details: dict[str, bool] = field(default_factory=dict)
    sources_available: int = 0


@dataclass
class AssembledResponse:
    response: str = ""
    sources: list[SourceCitation] = field(default_factory=list)
    quality: QualityReport = field(default_factory=QualityReport)
    has_sources: bool = False
    assembly_time_ms: float = 0.0
