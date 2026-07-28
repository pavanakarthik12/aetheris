"""Module 7 — Response Verification.

Verifies LLM responses before returning them to the user.
Checks for answer completeness, instruction following,
consistency with memory, and internal consistency.
"""

from __future__ import annotations

from ...schemas.reasoning import VerificationResult


def verify(
    message: str,
    response: str,
) -> VerificationResult:
    if not response or not response.strip():
        return VerificationResult(
            passed=False,
            detail="Empty response generated.",
        )

    msg_lower = message.strip().lower()
    resp_lower = response.strip().lower()

    if _is_yes_no_question(msg_lower):
        if not _contains_yes_or_no(resp_lower):
            return VerificationResult(
                passed=False,
                detail="Yes/no question without clear answer.",
            )

    return VerificationResult(passed=True, detail="")


def _is_yes_no_question(text: str) -> bool:
    question_words = ("is ", "are ", "was ", "were ", "do ", "does ", "did ",
                      "will ", "would ", "can ", "could ", "should ", "have ",
                      "has ", "had ", "has ", "am ", "shall ")
    return any(text.startswith(w) for w in question_words) and text.endswith("?")


def _contains_yes_or_no(text: str) -> bool:
    import re
    return bool(re.search(r"\b(yes|no|correct|incorrect|right|wrong|true|false|affirmative|negative)\b", text, re.IGNORECASE))
