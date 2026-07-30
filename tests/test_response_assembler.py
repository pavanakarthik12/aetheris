"""Comprehensive tests for the ResponseAssembler (Phase 13)."""

from __future__ import annotations

from backend.app.schemas.context_intelligence import ScoredResultItem, StructuredContext
from backend.app.schemas.response import AssembledResponse, QualityReport, SourceCitation
from backend.app.services.response_assembler import (
    ResponseAssembler,
    _CODE_KEYWORDS,
    _DONT_KNOW_PHRASES,
    _SOURCES_DIVIDER,
)

assembler = ResponseAssembler()


# ====================================================================
# Source Extraction
# ====================================================================


def test_extract_sources_no_context():
    sources = assembler._extract_sources(None)
    assert sources == []


def test_extract_sources_empty_results():
    ctx = StructuredContext(query="test")
    sources = assembler._extract_sources(ctx)
    assert sources == []


def test_extract_sources_with_results():
    ctx = StructuredContext(
        query="python",
        results=[
            ScoredResultItem(
                title="Python Docs",
                url="https://docs.python.org/3/",
                domain="docs.python.org",
                confidence=0.95,
                trust_score=0.99,
            ),
        ],
    )
    sources = assembler._extract_sources(ctx)
    assert len(sources) == 1
    assert sources[0].title == "Python Docs"
    assert sources[0].url == "https://docs.python.org/3/"
    assert sources[0].domain == "docs.python.org"
    assert sources[0].confidence == 0.95
    assert sources[0].trust_score == 0.99
    assert sources[0].position == 1


def test_extract_sources_deduplicates_by_url():
    ctx = StructuredContext(
        query="test",
        results=[
            ScoredResultItem(url="https://example.com/a", title="First"),
            ScoredResultItem(url="https://example.com/a", title="Duplicate"),
            ScoredResultItem(url="https://example.com/b", title="Second"),
        ],
    )
    sources = assembler._extract_sources(ctx)
    assert len(sources) == 2
    assert sources[0].title == "First"
    assert sources[1].title == "Second"


def test_extract_sources_skips_empty_url():
    ctx = StructuredContext(
        query="test",
        results=[
            ScoredResultItem(url="", title="No URL"),
            ScoredResultItem(url="https://example.com", title="Has URL"),
        ],
    )
    sources = assembler._extract_sources(ctx)
    assert len(sources) == 1
    assert sources[0].url == "https://example.com"


def test_extract_sources_fallback_title():
    ctx = StructuredContext(
        query="test",
        results=[
            ScoredResultItem(url="https://example.com", title=""),
        ],
    )
    sources = assembler._extract_sources(ctx)
    assert len(sources) == 1
    assert sources[0].title == "https://example.com"


# ====================================================================
# Source Appending
# ====================================================================


def test_append_sources_no_sources():
    result = assembler._append_sources("Hello world", [])
    assert result == "Hello world"


def test_append_sources_single():
    sources = [SourceCitation(title="PyDocs", url="https://docs.python.org/3/", position=1)]
    result = assembler._append_sources("Use `print()`.", sources)
    assert "---" in result
    assert "**Sources:**" in result
    assert "1. [PyDocs](https://docs.python.org/3/)" in result
    assert result.startswith("Use `print()`.")


def test_append_sources_multiple():
    sources = [
        SourceCitation(title="Source A", url="https://a.com", position=1),
        SourceCitation(title="Source B", url="https://b.com", position=2),
    ]
    result = assembler._append_sources("Response text.", sources)
    assert "1. [Source A](https://a.com)" in result
    assert "2. [Source B](https://b.com)" in result


def test_append_sources_already_present():
    sources = [SourceCitation(title="PyDocs", url="https://docs.python.org/3/", position=1)]
    result = assembler._append_sources(
        f"Response text.\n\n{_SOURCES_DIVIDER}\n1. [PyDocs](https://docs.python.org/3/)",
        sources,
    )
    assert result.count(_SOURCES_DIVIDER) == 1


def test_append_sources_preserves_whitespace():
    sources = [SourceCitation(title="Test", url="https://test.com", position=1)]
    result = assembler._append_sources("  Hello world  ", sources)
    assert result.startswith("Hello world")


# ====================================================================
# Quality Evaluation
# ====================================================================


def test_quality_empty_response():
    q = assembler._evaluate_quality("", "hello", None)
    assert not q.passed
    assert q.score < 0.6
    assert "Response is empty" in q.issues
    assert q.response_length == 0


def test_quality_whitespace_response():
    q = assembler._evaluate_quality("   ", "hello", None)
    assert not q.passed
    assert "Response is empty" in q.issues


def test_quality_too_short():
    q = assembler._evaluate_quality("Hi", "hello", None)
    assert not q.passed
    assert "Response is too short (under 10 characters)" in q.issues


def test_quality_valid_response():
    q = assembler._evaluate_quality(
        "This is a reasonably long response that should pass quality checks.",
        "hello",
        None,
    )
    assert q.passed
    assert q.score >= 0.6
    assert q.issues == []


def test_quality_false_ignorance_with_sources():
    ctx = StructuredContext(
        query="test",
        results=[ScoredResultItem(url="https://example.com", title="Test")],
    )
    q = assembler._evaluate_quality(
        "I don't know the answer to that question.",
        "what is python",
        ctx,
    )
    assert not q.passed
    assert "Response claims ignorance despite having sources available" in q.issues


def test_quality_no_false_ignorance_with_sources():
    ctx = StructuredContext(
        query="test",
        results=[ScoredResultItem(url="https://example.com", title="Test")],
    )
    q = assembler._evaluate_quality(
        "Here is the information you requested about Python.",
        "what is python",
        ctx,
    )
    assert q.passed
    assert "no_false_ignorance" not in q.issues


def test_quality_code_query_with_code_blocks():
    q = assembler._evaluate_quality(
        "Here is the code:\n```python\nprint('hello')\n```",
        "write a python function",
        None,
    )
    assert q.passed
    assert q.has_code_blocks


def test_quality_code_query_without_code_blocks():
    q = assembler._evaluate_quality(
        "You should use the print function to output text.",
        "write a python function",
        None,
    )
    assert "Code-related query but response has no code blocks" in q.issues
    assert not q.has_code_blocks


def test_quality_non_code_query_does_not_check_blocks():
    q = assembler._evaluate_quality(
        "The weather is nice today.",
        "what is the weather",
        None,
    )
    assert q.passed
    assert q.check_details.get("code_blocks_present") is True


def test_quality_urls_detected():
    q = assembler._evaluate_quality(
        "Check https://example.com for details.",
        "hello",
        None,
    )
    assert q.has_urls


def test_quality_no_urls():
    q = assembler._evaluate_quality(
        "This response has no URLs.",
        "hello",
        None,
    )
    assert not q.has_urls


def test_quality_check_details():
    q = assembler._evaluate_quality("A decent length response.", "hello", None)
    assert "non_empty" in q.check_details
    assert "sufficient_length" in q.check_details
    assert "no_false_ignorance" in q.check_details
    assert "code_blocks_present" in q.check_details
    assert "has_urls" in q.check_details


def test_quality_score_calculation():
    q = assembler._evaluate_quality("Hi", "hello", None)
    passed = sum(1 for v in q.check_details.values() if v)
    total = len(q.check_details)
    expected = (passed / total) if total > 0 else 1.0
    assert q.score == expected


def test_quality_passed_threshold():
    q_pass = assembler._evaluate_quality("A proper response that passes.", "hello", None)
    assert q_pass.passed
    q_fail = assembler._evaluate_quality("", "hello", None)
    assert not q_fail.passed


# ====================================================================
# Assemble (full pipeline)
# ====================================================================


def test_assemble_no_context():
    result = assembler.assemble("Hello world.", user_message="hi")
    assert isinstance(result, AssembledResponse)
    assert result.response == "Hello world."
    assert result.sources == []
    assert not result.has_sources
    assert result.quality.passed
    assert result.assembly_time_ms >= 0


def test_assemble_with_sources():
    ctx = StructuredContext(
        query="python",
        results=[
            ScoredResultItem(
                title="Python Docs",
                url="https://docs.python.org/3/",
                domain="docs.python.org",
                confidence=0.95,
                trust_score=0.99,
            ),
        ],
    )
    result = assembler.assemble("Use print().", ctx, "python function")
    assert "---" in result.response
    assert "**Sources:**" in result.response
    assert "Python Docs" in result.response
    assert len(result.sources) == 1
    assert result.has_sources


def test_assemble_empty_response():
    result = assembler.assemble("", user_message="hello")
    assert not result.quality.passed
    assert "Response is empty" in result.quality.issues


def test_assemble_ignorance_with_sources():
    ctx = StructuredContext(
        query="python",
        results=[ScoredResultItem(url="https://docs.python.org", title="Python Docs")],
    )
    result = assembler.assemble("I don't know.", ctx, "what is python")
    assert not result.quality.passed
    assert "Response claims ignorance despite having sources available" in result.quality.issues


def test_assemble_multiple_sources():
    ctx = StructuredContext(
        query="python",
        results=[
            ScoredResultItem(url="https://a.com", title="A", domain="a.com"),
            ScoredResultItem(url="https://b.com", title="B", domain="b.com"),
            ScoredResultItem(url="https://c.com", title="C", domain="c.com"),
        ],
    )
    result = assembler.assemble("Here is the info.", ctx, "python")
    assert len(result.sources) == 3
    assert result.has_sources
    for s in result.sources:
        assert s.position >= 1


def test_assemble_quality_metadata():
    result = assembler.assemble("A response.", user_message="hello")
    assert result.quality.response_length == 11
    assert isinstance(result.quality.has_code_blocks, bool)
    assert isinstance(result.quality.has_urls, bool)
    assert isinstance(result.quality.sources_available, int)
    assert isinstance(result.quality.check_details, dict)


def test_assemble_code_query_no_code():
    result = assembler.assemble(
        "You should write a function to do that.",
        user_message="write a python function",
    )
    assert "Code-related query but response has no code blocks" in result.quality.issues
    assert "code_blocks_present" in result.quality.check_details


def test_assemble_code_query_with_code():
    result = assembler.assemble(
        "```python\ndef hello():\n    pass\n```",
        user_message="write a python function",
    )
    assert result.quality.passed
    assert result.quality.has_code_blocks


# ====================================================================
# SourceCitation & QualityReport schema
# ====================================================================


def test_source_citation_defaults():
    s = SourceCitation()
    assert s.title == ""
    assert s.url == ""
    assert s.domain == ""
    assert s.confidence == 0.0
    assert s.trust_score == 0.0
    assert s.position == 0


def test_source_citation_with_values():
    s = SourceCitation(title="T", url="U", domain="D", confidence=0.9, trust_score=0.8, position=1)
    assert s.title == "T"
    assert s.url == "U"
    assert s.domain == "D"
    assert s.confidence == 0.9
    assert s.trust_score == 0.8
    assert s.position == 1


def test_quality_report_defaults():
    q = QualityReport()
    assert q.passed
    assert q.score == 1.0
    assert q.issues == []
    assert q.response_length == 0
    assert not q.has_code_blocks
    assert not q.has_urls
    assert q.check_details == {}
    assert q.sources_available == 0


def test_quality_report_with_values():
    q = QualityReport(
        passed=False,
        score=0.5,
        issues=["test issue"],
        response_length=100,
        has_code_blocks=True,
        has_urls=True,
        check_details={"check": True},
        sources_available=3,
    )
    assert not q.passed
    assert q.score == 0.5
    assert q.issues == ["test issue"]
    assert q.response_length == 100
    assert q.has_code_blocks
    assert q.has_urls
    assert q.check_details == {"check": True}
    assert q.sources_available == 3


def test_assembled_response_defaults():
    a = AssembledResponse()
    assert a.response == ""
    assert a.sources == []
    assert a.quality.passed
    assert not a.has_sources
    assert a.assembly_time_ms == 0.0


def test_assembled_response_with_values():
    q = QualityReport(passed=False, score=0.3, issues=["bad"])
    s = [SourceCitation(title="T", url="U")]
    a = AssembledResponse(
        response="Hello",
        sources=s,
        quality=q,
        has_sources=True,
        assembly_time_ms=1.5,
    )
    assert a.response == "Hello"
    assert len(a.sources) == 1
    assert not a.quality.passed
    assert a.has_sources
    assert a.assembly_time_ms == 1.5


# ====================================================================
# Edge cases
# ====================================================================


def test_assemble_large_source_count():
    results = [
        ScoredResultItem(url=f"https://example.com/{i}", title=f"Source {i}")
        for i in range(20)
    ]
    ctx = StructuredContext(query="test", results=results)
    result = assembler.assemble("Response.", ctx, "test")
    assert len(result.sources) == 20
    assert result.has_sources


def test_assemble_source_without_structured_context():
    result = assembler.assemble("Response.", user_message="test")
    assert result.sources == []
    assert not result.has_sources


def test_assemble_does_not_duplicate_existing_sources():
    ctx = StructuredContext(
        query="test",
        results=[ScoredResultItem(url="https://example.com", title="Test")],
    )
    response_text = "Some text.\n\n---\n\n**Sources:**\n1. [Test](https://example.com)"
    result = assembler.assemble(response_text, ctx, "test")
    assert result.response.count("**Sources:**") == 1


def test_quality_non_code_keyword_does_not_trigger():
    q = assembler._evaluate_quality(
        "Hello world.",
        "how are you",
        None,
    )
    assert q.passed
    assert q.check_details["code_blocks_present"] is True


def test_quality_code_keyword_case_insensitive():
    q = assembler._evaluate_quality(
        "Some text without code.",
        "Write A Function that adds numbers",
        None,
    )
    assert "Code-related query but response has no code blocks" in q.issues
    assert not q.check_details["code_blocks_present"]


def test_quality_false_ignorance_phrases():
    ctx = StructuredContext(
        query="test",
        results=[ScoredResultItem(url="https://example.com", title="Test")],
    )
    for phrase in _DONT_KNOW_PHRASES:
        q = assembler._evaluate_quality(phrase, "question", ctx)
        assert not q.check_details["no_false_ignorance"], f"Failed for phrase: {phrase}"


def test_code_keywords_list_used():
    assert "code" in _CODE_KEYWORDS
    assert "function" in _CODE_KEYWORDS
    assert "def " in _CODE_KEYWORDS


def test_quality_unknown_sources_not_available():
    q = assembler._evaluate_quality("I don't know.", "question", None)
    assert q.passed
    assert q.check_details["no_false_ignorance"]


def test_quality_code_query_edge_keywords():
    for kw in ["api", "endpoint", "route", "hook"]:
        q = assembler._evaluate_quality("Text.", kw, None)
        assert "Code-related query but response has no code blocks" in q.issues, (
            f"Failed to detect code query for keyword: {kw}"
        )
        assert not q.check_details["code_blocks_present"]
