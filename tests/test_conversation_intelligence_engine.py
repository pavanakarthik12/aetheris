from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.conversation_intelligence_engine import (
    ConversationIntelligenceEngine,
    FollowUpIntent,
)


def _msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def _sample_history() -> list[dict[str, str]]:
    return [
        _msg("user", "What is machine learning?"),
        _msg("assistant", "Machine learning is a subset of AI that enables systems to learn from data."),
        _msg("user", "How does supervised learning work?"),
        _msg("assistant", "Supervised learning uses labeled data to train models to predict outputs."),
    ]


@pytest.fixture
def engine() -> ConversationIntelligenceEngine:
    return ConversationIntelligenceEngine()


class TestNewTopic:
    pytestmark = pytest.mark.asyncio

    async def test_empty_message_returns_new_topic(self, engine: ConversationIntelligenceEngine) -> None:
        result = await engine.analyze("", _sample_history())
        assert result.is_follow_up is False
        assert result.intent == FollowUpIntent.NEW_TOPIC

    async def test_empty_history_returns_new_topic(self, engine: ConversationIntelligenceEngine) -> None:
        result = await engine.analyze("Tell me about AI", [])
        assert result.is_follow_up is False
        assert result.intent == FollowUpIntent.NEW_TOPIC

    async def test_no_follow_up_phrase_returns_new_topic(self, engine: ConversationIntelligenceEngine) -> None:
        result = await engine.analyze("What is the capital of France?", _sample_history())
        assert result.is_follow_up is False
        assert result.intent == FollowUpIntent.NEW_TOPIC


class TestFollowUpDetection:
    pytestmark = pytest.mark.asyncio

    async def test_elaborate_detected(self, engine: ConversationIntelligenceEngine) -> None:
        result = await engine.analyze("elaborate", _sample_history())
        assert result.is_follow_up is True
        assert result.intent == FollowUpIntent.FOLLOW_UP
        assert result.previous_assistant_response == "Supervised learning uses labeled data to train models to predict outputs."

    async def test_continue_detected(self, engine: ConversationIntelligenceEngine) -> None:
        result = await engine.analyze("continue", _sample_history())
        assert result.is_follow_up is True
        assert result.intent == FollowUpIntent.CONTINUATION

    async def test_why_detected(self, engine: ConversationIntelligenceEngine) -> None:
        result = await engine.analyze("why is that important?", _sample_history())
        assert result.is_follow_up is True
        assert result.intent == FollowUpIntent.CLARIFICATION

    async def test_compare_detected(self, engine: ConversationIntelligenceEngine) -> None:
        result = await engine.analyze("compare", _sample_history())
        assert result.is_follow_up is True
        assert result.intent == FollowUpIntent.COMPARISON

    async def test_actually_correction_detected(self, engine: ConversationIntelligenceEngine) -> None:
        result = await engine.analyze("actually, that's not what I meant", _sample_history())
        assert result.is_follow_up is True
        assert result.intent == FollowUpIntent.CORRECTION

    async def test_simplify_detected(self, engine: ConversationIntelligenceEngine) -> None:
        result = await engine.analyze("simplify that", _sample_history())
        assert result.is_follow_up is True
        assert result.intent == FollowUpIntent.REFINEMENT

    async def test_refer_about_detected(self, engine: ConversationIntelligenceEngine) -> None:
        result = await engine.analyze("about that", _sample_history())
        assert result.is_follow_up is True
        assert result.intent == FollowUpIntent.REFERENCE

    async def test_tell_me_more_detected(self, engine: ConversationIntelligenceEngine) -> None:
        result = await engine.analyze("tell me more", _sample_history())
        assert result.is_follow_up is True
        assert result.intent == FollowUpIntent.FOLLOW_UP


class TestTopicResolution:
    pytestmark = pytest.mark.asyncio

    async def test_continuation_resolves_to_previous_user(self, engine: ConversationIntelligenceEngine) -> None:
        history = [_msg("user", "What is supervised learning?"), _msg("assistant", "It is a type of ML.")]
        result = await engine.analyze("continue", history)
        assert result.resolved_topic == "What is supervised learning?"

    async def test_clarification_single_topic(self, engine: ConversationIntelligenceEngine) -> None:
        history = [_msg("user", "Explain neural networks."), _msg("assistant", "Neural networks are computing systems.")]
        result = await engine.analyze("explain that part", history)
        assert result.resolved_topic in history[1]["content"]

    async def test_comparison_topic_includes_both(self, engine: ConversationIntelligenceEngine) -> None:
        history = [_msg("user", "What is Python?"), _msg("assistant", "Python is a programming language.")]
        result = await engine.analyze("compare with Java", history)
        assert "Python" in result.resolved_topic or "Java" in result.resolved_topic

    async def test_correction_resolves_to_assistant(self, engine: ConversationIntelligenceEngine) -> None:
        history = [_msg("user", "What is an API?"), _msg("assistant", "An API is an application programming interface.")]
        result = await engine.analyze("no, I meant what is a REST API", history)
        assert result.is_follow_up is True
        assert result.resolved_topic == history[1]["content"]

    async def test_reference_extracts_topic(self, engine: ConversationIntelligenceEngine) -> None:
        history = [_msg("user", "Tell me about transformers."), _msg("assistant", "Transformers are a neural network architecture.")]
        result = await engine.analyze("regarding attention mechanisms", history)
        assert result.is_follow_up is True


class TestClarification:
    pytestmark = pytest.mark.asyncio

    async def test_ambiguous_reference_triggers_clarification(self, engine: ConversationIntelligenceEngine) -> None:
        history = [
            _msg("user", "What is Python?"),
            _msg("assistant", "Python is a language."),
            _msg("user", "What is JavaScript?"),
            _msg("assistant", "JavaScript is another language."),
            _msg("user", "What is Rust?"),
            _msg("assistant", "Rust is a systems language."),
        ]
        result = await engine.analyze("explain that part", history)
        assert result.needs_clarification is True
        assert len(result.possible_topics) > 1

    async def test_clarification_question_built(self, engine: ConversationIntelligenceEngine) -> None:
        analysis = await engine.analyze("explain that part", [
            _msg("user", "Topic A"), _msg("assistant", "Answer A"),
            _msg("user", "Topic B"), _msg("assistant", "Answer B"),
        ])
        assert analysis.needs_clarification
        question = engine.build_clarification_question(analysis)
        assert "Would you like me to elaborate on" in question

    async def test_high_confidence_no_clarification(self, engine: ConversationIntelligenceEngine) -> None:
        history = [_msg("user", "What is supervised learning?"), _msg("assistant", "It uses labeled data.")]
        result = await engine.analyze("continue", history)
        assert result.needs_clarification is False
        assert result.confidence >= 0.5


class TestContextBuilding:
    pytestmark = pytest.mark.asyncio

    async def test_follow_up_context_includes_exchange(self, engine: ConversationIntelligenceEngine) -> None:
        history = [_msg("user", "What is AI?"), _msg("assistant", "AI is artificial intelligence.")]
        analysis = await engine.analyze("elaborate", history)
        context = engine.build_follow_up_context(analysis)
        assert "Previous Question:" in context
        assert "Previous Answer:" in context
        assert "Resolved Topic:" in context
        assert "Follow-up Intent:" in context

    async def test_new_topic_returns_empty_context(self, engine: ConversationIntelligenceEngine) -> None:
        analysis = await engine.analyze("What is the weather?", _sample_history())
        context = engine.build_follow_up_context(analysis)
        assert context == ""


class TestEmptyEdgeCase:
    pytestmark = pytest.mark.asyncio

    async def test_no_history_returns_new_topic(self, engine: ConversationIntelligenceEngine) -> None:
        result = await engine.analyze("continue", [])
        assert result.is_follow_up is False

    async def test_history_after_only_user_no_assistant(self, engine: ConversationIntelligenceEngine) -> None:
        result = await engine.analyze("continue", [_msg("user", "Hello")])
        assert result.is_follow_up is True
        assert result.previous_assistant_response == ""

    async def test_single_exchange_follow_up(self, engine: ConversationIntelligenceEngine) -> None:
        history = [_msg("user", "What is Docker?"), _msg("assistant", "Docker is containerization.")]
        result = await engine.analyze("more", history)
        assert result.is_follow_up is True
        assert result.previous_user_message == "What is Docker?"
        assert result.previous_assistant_response == "Docker is containerization."


class TestMultipleExchanges:
    pytestmark = pytest.mark.asyncio

    async def test_deep_history_picks_last_exchange(self, engine: ConversationIntelligenceEngine) -> None:
        history = [
            _msg("user", "First topic"),
            _msg("assistant", "First answer"),
            _msg("user", "Second topic"),
            _msg("assistant", "Second answer"),
            _msg("user", "Third topic"),
            _msg("assistant", "Third answer"),
        ]
        result = await engine.analyze("elaborate", history)
        assert result.previous_user_message == "Third topic"
        assert result.previous_assistant_response == "Third answer"


@pytest.mark.asyncio
async def test_real_time_execution() -> None:
    engine = ConversationIntelligenceEngine()
    history = [
        {"role": "user", "content": "Explain quantum computing."},
        {"role": "assistant", "content": "Quantum computing uses qubits."},
    ]
    result = await engine.analyze("elaborate", history)
    assert result.is_follow_up is True
    assert result.execution_time_ms > 0.0
