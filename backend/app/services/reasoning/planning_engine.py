"""Module 6 — Planning Engine.

For medium and complex requests, creates an internal execution plan.
The plan is never exposed to the user.
"""

from __future__ import annotations

from ...schemas.reasoning import ComplexityLevel, SemanticIntentType

_PLAN_TEMPLATES: dict[SemanticIntentType, list[str]] = {
    SemanticIntentType.PLANNING: [
        "Understand the full requirements",
        "Identify key components and their relationships",
        "Determine the optimal approach",
        "Structure the response with clear sections",
        "Include practical considerations and trade-offs",
    ],
    SemanticIntentType.CODE_GENERATION: [
        "Understand the code requirements",
        "Determine the programming language and framework",
        "Design the code structure",
        "Implement with best practices",
        "Add error handling and edge cases",
    ],
    SemanticIntentType.DEBUGGING: [
        "Analyze the reported issue",
        "Identify likely root causes",
        "Determine the fix approach",
        "Verify the solution addresses the problem",
    ],
    SemanticIntentType.REASONING: [
        "Break down the problem into premises",
        "Evaluate each premise",
        "Draw logical conclusions",
        "Consider alternative perspectives",
        "Summarize the reasoning chain",
    ],
    SemanticIntentType.EXPLANATION: [
        "Identify the core concept to explain",
        "Determine the appropriate level of detail",
        "Structure explanation from basics to advanced",
        "Use examples to illustrate key points",
    ],
    SemanticIntentType.PROGRAMMING: [
        "Analyze the programming question",
        "Retrieve relevant programming knowledge",
        "Structure the answer with code examples if appropriate",
        "Explain the reasoning behind the solution",
    ],
    SemanticIntentType.CREATIVE_WRITING: [
        "Understand the creative direction",
        "Determine the tone and style",
        "Structure the piece with proper flow",
        "Refine language for impact",
    ],
}

_GENERIC_PLAN: list[str] = [
    "Understand the request",
    "Retrieve required context",
    "Determine solution strategy",
    "Generate structured response",
    "Verify completeness",
]


def create_plan(
    semantic_intent: SemanticIntentType,
    complexity: ComplexityLevel,
    message: str,
) -> list[str]:
    if complexity == ComplexityLevel.SIMPLE:
        return []

    plan = _PLAN_TEMPLATES.get(semantic_intent, _GENERIC_PLAN)

    if complexity == ComplexityLevel.COMPLEX:
        plan = plan + [
            "Consider edge cases and alternatives",
            "Add implementation details",
        ]

    return plan
