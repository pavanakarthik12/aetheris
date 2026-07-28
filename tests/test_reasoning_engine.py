"""Tests for the Cognitive Reasoning Engine (Phase 10)."""

from __future__ import annotations

import sys
from pathlib import Path

import unittest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.schemas.reasoning import (
    CognitiveTrace,
    ComplexityLevel,
    ConfidenceLevel,
    SemanticIntentType,
)
from backend.app.services.reasoning.intent_analyzer import analyze_intent
from backend.app.services.reasoning.complexity_classifier import classify_complexity
from backend.app.services.reasoning.task_decomposer import decompose
from backend.app.services.reasoning.decision_engine import decide
from backend.app.services.reasoning.planning_engine import create_plan
from backend.app.services.reasoning.response_verifier import verify
from backend.app.services.reasoning.confidence_estimator import estimate_confidence
from backend.app.services.reasoning.pipeline import ReasoningPipeline


class IntentAnalyzerTests(unittest.TestCase):
    def test_greeting(self) -> None:
        self.assertEqual(analyze_intent("hello"), SemanticIntentType.GREETING)
        self.assertEqual(analyze_intent("Hi there!"), SemanticIntentType.GREETING)
        self.assertEqual(analyze_intent("good morning"), SemanticIntentType.GREETING)

    def test_mathematics(self) -> None:
        self.assertEqual(analyze_intent("what is 5 + 6"), SemanticIntentType.MATHEMATICS)
        self.assertEqual(analyze_intent("calculate 123 * 456"), SemanticIntentType.MATHEMATICS)
        self.assertEqual(analyze_intent("solve x + 5 = 10"), SemanticIntentType.MATHEMATICS)

    def test_memory_retrieval(self) -> None:
        self.assertEqual(
            analyze_intent("what do you know about me"),
            SemanticIntentType.MEMORY_RETRIEVAL,
        )
        self.assertEqual(
            analyze_intent("what is my name"),
            SemanticIntentType.MEMORY_RETRIEVAL,
        )
        self.assertEqual(
            analyze_intent("search my memories about python"),
            SemanticIntentType.MEMORY_RETRIEVAL,
        )

    def test_conversation_continuation(self) -> None:
        self.assertEqual(
            analyze_intent("continue our previous discussion"),
            SemanticIntentType.CONVERSATION_CONTINUATION,
        )
        self.assertEqual(
            analyze_intent("what did we just talk about"),
            SemanticIntentType.CONVERSATION_CONTINUATION,
        )

    def test_programming(self) -> None:
        self.assertEqual(
            analyze_intent("what is a closure in Python"),
            SemanticIntentType.PROGRAMMING,
        )
        self.assertEqual(
            analyze_intent("how do I define a class in Java"),
            SemanticIntentType.PROGRAMMING,
        )

    def test_debugging(self) -> None:
        self.assertEqual(
            analyze_intent("my code has a bug"),
            SemanticIntentType.DEBUGGING,
        )
        self.assertEqual(
            analyze_intent("why isn't my app working"),
            SemanticIntentType.DEBUGGING,
        )

    def test_planning(self) -> None:
        self.assertEqual(
            analyze_intent("design a scalable authentication system"),
            SemanticIntentType.PLANNING,
        )

    def test_code_generation(self) -> None:
        self.assertEqual(
            analyze_intent("write a function to sort an array"),
            SemanticIntentType.CODE_GENERATION,
        )
        self.assertEqual(
            analyze_intent("generate a python script for web scraping"),
            SemanticIntentType.CODE_GENERATION,
        )

    def test_creative_writing(self) -> None:
        self.assertEqual(
            analyze_intent("write a poem about AI"),
            SemanticIntentType.CREATIVE_WRITING,
        )

    def test_general_conversation(self) -> None:
        self.assertEqual(
            analyze_intent("I like pizza"),
            SemanticIntentType.GENERAL_CONVERSATION,
        )


class ComplexityClassifierTests(unittest.TestCase):
    def test_simple_greeting(self) -> None:
        self.assertEqual(
            classify_complexity("hello", SemanticIntentType.GREETING),
            ComplexityLevel.SIMPLE,
        )

    def test_simple_math(self) -> None:
        self.assertEqual(
            classify_complexity("what is 5 + 6", SemanticIntentType.MATHEMATICS),
            ComplexityLevel.SIMPLE,
        )

    def test_medium_explanation(self) -> None:
        self.assertEqual(
            classify_complexity("explain what a database is", SemanticIntentType.EXPLANATION),
            ComplexityLevel.MEDIUM,
        )

    def test_complex_planning(self) -> None:
        self.assertEqual(
            classify_complexity(
                "design a scalable authentication system",
                SemanticIntentType.PLANNING,
            ),
            ComplexityLevel.COMPLEX,
        )

    def test_long_message_escalates(self) -> None:
        long_msg = " ".join(["word"] * 250)
        self.assertEqual(
            classify_complexity(long_msg, SemanticIntentType.GENERAL_CONVERSATION),
            ComplexityLevel.COMPLEX,
        )

    def test_short_simple_message(self) -> None:
        self.assertEqual(
            classify_complexity("hello world", SemanticIntentType.GENERAL_CONVERSATION),
            ComplexityLevel.SIMPLE,
        )


class TaskDecomposerTests(unittest.TestCase):
    def test_no_decomposition(self) -> None:
        tasks = decompose("what is 5 + 6")
        self.assertEqual(len(tasks), 0)

    def test_multi_objective_decomposition(self) -> None:
        tasks = decompose("build a login page and explain JWT authentication")
        self.assertGreaterEqual(len(tasks), 2)
        descriptions = [t.description.lower() for t in tasks]
        self.assertTrue(any("login" in d for d in descriptions))
        self.assertTrue(any("jwt" in d for d in descriptions))

    def test_and_also_decomposition(self) -> None:
        tasks = decompose("create a user model and also add validation")
        self.assertGreaterEqual(len(tasks), 2)


class DecisionEngineTests(unittest.TestCase):
    def test_greeting_skips_memory(self) -> None:
        decision, clarification, q, planning = decide(
            SemanticIntentType.GREETING,
            ComplexityLevel.SIMPLE,
            "hello",
            False,
        )
        self.assertFalse(decision.conversation_memory)
        self.assertFalse(decision.long_term_memory)
        self.assertFalse(planning)

    def test_simple_request_skips_memory(self) -> None:
        decision, clarification, q, planning = decide(
            SemanticIntentType.MATHEMATICS,
            ComplexityLevel.SIMPLE,
            "what is 5 + 6",
            False,
        )
        self.assertFalse(decision.conversation_memory)
        self.assertFalse(decision.long_term_memory)

    def test_conversation_continuation_needs_memory(self) -> None:
        decision, clarification, q, planning = decide(
            SemanticIntentType.CONVERSATION_CONTINUATION,
            ComplexityLevel.MEDIUM,
            "continue our previous discussion",
            False,
        )
        self.assertTrue(decision.conversation_memory)
        self.assertTrue(decision.long_term_memory)

    def test_planning_needs_planning(self) -> None:
        decision, clarification, q, planning = decide(
            SemanticIntentType.PLANNING,
            ComplexityLevel.COMPLEX,
            "design a system",
            False,
        )
        self.assertTrue(planning)

    def test_complex_request_activates_planning(self) -> None:
        decision, clarification, q, planning = decide(
            SemanticIntentType.CODE_GENERATION,
            ComplexityLevel.COMPLEX,
            "build a full authentication system",
            False,
        )
        self.assertTrue(planning)

    def test_vague_request_asks_clarification(self) -> None:
        decision, clarification, q, planning = decide(
            SemanticIntentType.GENERAL_CONVERSATION,
            ComplexityLevel.MEDIUM,
            "build me a website",
            False,
        )
        self.assertTrue(clarification)
        self.assertIn("website", q.lower())


class PlanningEngineTests(unittest.TestCase):
    def test_simple_no_plan(self) -> None:
        plan = create_plan(
            SemanticIntentType.GREETING,
            ComplexityLevel.SIMPLE,
            "hello",
        )
        self.assertEqual(len(plan), 0)

    def test_complex_has_plan(self) -> None:
        plan = create_plan(
            SemanticIntentType.PLANNING,
            ComplexityLevel.COMPLEX,
            "design a system",
        )
        self.assertGreater(len(plan), 0)

    def test_medium_has_plan(self) -> None:
        plan = create_plan(
            SemanticIntentType.PLANNING,
            ComplexityLevel.MEDIUM,
            "explain something",
        )
        self.assertGreater(len(plan), 0)


class ResponseVerifierTests(unittest.TestCase):
    def test_empty_response_fails(self) -> None:
        result = verify("hello", "")
        self.assertFalse(result.passed)

    def test_valid_response_passes(self) -> None:
        result = verify("hello", "Hi there!")
        self.assertTrue(result.passed)

    def test_yes_no_question_without_answer_fails(self) -> None:
        result = verify(
            "Is Python a programming language?",
            "It is used by many developers.",
        )
        self.assertFalse(result.passed)

    def test_yes_no_question_with_answer_passes(self) -> None:
        result = verify(
            "Is Python a programming language?",
            "Yes, Python is a programming language.",
        )
        self.assertTrue(result.passed)


class ConfidenceEstimatorTests(unittest.TestCase):
    def test_math_high_confidence(self) -> None:
        trace = CognitiveTrace()
        conf = estimate_confidence(SemanticIntentType.MATHEMATICS, ComplexityLevel.SIMPLE, trace)
        self.assertEqual(conf, ConfidenceLevel.HIGH)

    def test_planning_low_confidence(self) -> None:
        trace = CognitiveTrace()
        conf = estimate_confidence(SemanticIntentType.PLANNING, ComplexityLevel.COMPLEX, trace)
        self.assertIn(conf, (ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM))


class ReasoningPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = ReasoningPipeline()

    def test_scenario1_math(self) -> None:
        """What is 5 + 6?"""
        import asyncio
        plan, trace = asyncio.run(self.pipeline.run("What is 5 + 6?"))
        self.assertEqual(plan.semantic_intent, SemanticIntentType.MATHEMATICS)
        self.assertEqual(plan.complexity, ComplexityLevel.SIMPLE)
        self.assertFalse(plan.needs_clarification)
        self.assertFalse(plan.needs_planning)
        self.assertEqual(len(plan.tasks), 0)

    def test_scenario2_memory_retrieval(self) -> None:
        """What project am I building?"""
        import asyncio
        plan, trace = asyncio.run(self.pipeline.run("What project am I building?"))
        self.assertEqual(plan.semantic_intent, SemanticIntentType.MEMORY_RETRIEVAL)
        self.assertTrue(plan.memory_sources.long_term_memory)
        self.assertFalse(plan.needs_planning)

    def test_scenario3_conversation_continue(self) -> None:
        """Continue our previous discussion."""
        import asyncio
        plan, trace = asyncio.run(self.pipeline.run("Continue our previous discussion."))
        self.assertEqual(plan.semantic_intent, SemanticIntentType.CONVERSATION_CONTINUATION)
        self.assertTrue(plan.memory_sources.conversation_memory)

    def test_scenario4_complex_planning(self) -> None:
        """Design a scalable authentication system."""
        import asyncio
        plan, trace = asyncio.run(self.pipeline.run("Design a scalable authentication system."))
        self.assertEqual(plan.semantic_intent, SemanticIntentType.PLANNING)
        self.assertEqual(plan.complexity, ComplexityLevel.COMPLEX)
        self.assertTrue(plan.needs_planning)
        self.assertGreater(len(plan.planning_steps), 0)

    def test_scenario5_task_decomposition(self) -> None:
        """Build a login page and explain JWT."""
        import asyncio
        plan, trace = asyncio.run(self.pipeline.run("Build a login page and explain JWT."))
        self.assertGreater(len(plan.tasks), 0)

    def test_cognitive_trace(self) -> None:
        import asyncio
        plan, trace = asyncio.run(self.pipeline.run("What is 5 + 6?"))
        self.assertEqual(trace.semantic_intent, SemanticIntentType.MATHEMATICS)
        self.assertEqual(trace.complexity, ComplexityLevel.SIMPLE)
        self.assertGreater(trace.reasoning_duration_ms, 0)

    def test_greeting_bypasses_memory(self) -> None:
        import asyncio
        plan, trace = asyncio.run(self.pipeline.run("hello"))
        self.assertEqual(plan.semantic_intent, SemanticIntentType.GREETING)
        self.assertFalse(plan.memory_sources.conversation_memory)
        self.assertFalse(plan.memory_sources.long_term_memory)

    def test_vague_request_clarification(self) -> None:
        import asyncio
        plan, trace = asyncio.run(self.pipeline.run("build me a website"))
        self.assertTrue(plan.needs_clarification)
        self.assertIn("website", plan.clarification_question.lower())


if __name__ == "__main__":
    unittest.main()
