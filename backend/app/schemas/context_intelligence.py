"""Schemas for the Context Intelligence Engine output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrustConfig:
    default_trust: float = 0.5
    min_trust: float = 0.0
    max_trust: float = 1.0
    max_tokens: int = 1500
    max_results: int = 10
    min_confidence: float = 0.05
    trusted_domains: dict[str, float] = field(default_factory=lambda: {
        "docs.python.org": 0.99,
        "python.org": 0.95,
        "github.com": 0.90,
        "wikipedia.org": 0.90,
        ".gov": 0.95,
        ".edu": 0.90,
        ".ac.": 0.85,
        "react.dev": 0.99,
        "fastapi.tiangolo.com": 0.99,
        "tiangolo.com": 0.90,
        "nodejs.org": 0.95,
        "developer.mozilla.org": 0.99,
        "mdn.dev": 0.95,
        "dev.to": 0.60,
        "medium.com": 0.40,
        "towardsdatascience.com": 0.50,
        "geeksforgeeks.org": 0.55,
        "stackoverflow.com": 0.85,
        "stackexchange.com": 0.80,
        "npmjs.com": 0.85,
        "pypi.org": 0.90,
        "crates.io": 0.85,
        "docker.com": 0.85,
        "kubernetes.io": 0.90,
        "aws.amazon.com": 0.85,
        "learn.microsoft.com": 0.90,
        "oracle.com": 0.80,
        "ibm.com": 0.80,
        "redhat.com": 0.80,
        "cncf.io": 0.85,
        "ieee.org": 0.90,
        "arxiv.org": 0.85,
        "scholar.google.com": 0.90,
        "spring.io": 0.85,
        "typescriptlang.org": 0.90,
        "rust-lang.org": 0.90,
        "golang.org": 0.90,
        "jQuery.com": 0.70,
        "w3schools.com": 0.65,
        "tutorialspoint.com": 0.45,
        "javapoint.com": 0.40,
        "baeldung.com": 0.70,
    })


@dataclass
class ScoredResultItem:
    title: str = ""
    snippet: str = ""
    url: str = ""
    domain: str = ""
    confidence: float = 0.0
    trust_score: float = 0.0
    relevance_score: float = 0.0
    source: str = ""
    is_duplicate: bool = False
    is_invalid: bool = False


@dataclass
class StructuredContext:
    query: str = ""
    results: list[ScoredResultItem] = field(default_factory=list)
    estimated_tokens: int = 0
    sources_used: int = 0
    original_count: int = 0
    duplicates_removed: int = 0
    filtered_count: int = 0
    processing_time_ms: float = 0.0
