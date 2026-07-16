"""Memory normalizer — converts first-person memories to third-person user facts.

Every stored memory must represent information ABOUT THE USER.
Never store memories in first person.
The LLM must never confuse user memories with its own identity.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Patterns that detect common first-person constructs
_I_PATTERN: re.Pattern[str] = re.compile(r"\bI(?:'m| am| was| have|'ve| had|'ll| will| would|'d| could| can| do| did| shall| should|)\b", re.IGNORECASE)
_MY_PATTERN: re.Pattern[str] = re.compile(r"\bmy\b", re.IGNORECASE)
_ME_PATTERN: re.Pattern[str] = re.compile(r"\bme\b", re.IGNORECASE)
_WE_PATTERN: re.Pattern[str] = re.compile(r"\bwe\b", re.IGNORECASE)
_OUR_PATTERN: re.Pattern[str] = re.compile(r"\bour\b", re.IGNORECASE)



_KNOWN_THIRD_PERSON_PREFIXES: tuple[str, ...] = (
    "the user",
    "the user's",
    "user's",
    "the assistant",
)


def is_first_person(text: str) -> bool:
    """Return True if *text* contains first-person pronouns."""
    lowered = text.lower().strip()
    if lowered.startswith(_KNOWN_THIRD_PERSON_PREFIXES):
        return False
    if _I_PATTERN.search(lowered):
        return True
    if _MY_PATTERN.search(lowered):
        return True
    if _ME_PATTERN.search(lowered):
        return True
    if _WE_PATTERN.search(lowered):
        return True
    if _OUR_PATTERN.search(lowered):
        return True
    return False


def normalize_to_third_person(text: str, category: str = "", attribute: str = "", value: str = "") -> str:
    """Convert a first-person memory to a third-person user fact.

    Uses the provided structured data first; falls back to rule-based conversion.

    Args:
        text: The original message to normalize.
        category: Memory category (Preference, Project, Goal, etc.).
        attribute: Machine-readable attribute key.
        value: The attribute value.

    Returns:
        A third-person fact string describing the user.
    """
    if not text or not text.strip():
        return ""

    # If already third-person, return as-is
    if not is_first_person(text):
        return text.strip()

    # If structured data is available, build from it
    if attribute and value:
        fact = _build_from_structured(text, category, attribute, value)
        if fact:
            return fact

    # Rule-based conversion
    return _rule_based_convert(text.strip(), category)


def _build_from_structured(text: str, category: str, attribute: str, value: str) -> str | None:
    """Build a third-person fact from structured attribute/value data."""
    # Map the internal category to a natural language attribute phrase
    cat_lower = category.lower() if category else ""
    lower_attr = attribute.lower().replace("_", " ")

    if cat_lower == "preference":
        if "language" in lower_attr:
            return f"The user's preferred programming language is {value}."
        if "food" in lower_attr:
            return f"The user's favorite food is {value}."
        if "subject" in lower_attr:
            return f"The user's favorite subject is {value}."
        return f"The user prefers {value}."

    if cat_lower == "project":
        return f"The user is building a project called {value}."

    if cat_lower == "goal":
        return f"The user's goal is {value}."

    if cat_lower == "skill":
        return f"The user is learning {value}."

    if cat_lower == "fact":
        return f"The user's fact: {value}."

    if cat_lower == "relationship":
        return f"The user's relationship: {value}."

    if cat_lower == "achievement":
        return f"The user achieved {value}."

    if cat_lower == "event":
        return f"The user has an event: {value}."

    if cat_lower == "task":
        return f"The user has a task: {value}."

    if lower_attr in ("name", "current name"):
        return f"The user's current name is {value}."

    if lower_attr in ("occupation", "job", "work"):
        return f"The user's occupation is {value}."

    if lower_attr in ("age",):
        return f"The user's age is {value}."

    if lower_attr in ("city", "country", "location"):
        return f"The user lives in {value}."

    # Generic structured fact
    return f"The user's {lower_attr} is {value}."


def _rule_based_convert(text: str, category: str = "") -> str:
    """Convert a first-person sentence to third-person using rules.

    Handles patterns like:
      "I am building X." -> "The user is building X."
      "I'm building X." -> "The user is building X."
      "My name is X." -> "The user's name is X."
      "My favorite language is X." -> "The user's favorite language is X."
      "I like X." -> "The user likes X."
      "I started X." -> "The user started X."
      "I have X." -> "The user has X."
    """
    stripped = text.strip()

    # Pattern 1: "My X is Y" -> "The user's X is Y"
    my_match = re.match(r"\b[Mm]y\s+(.+?)\s+is\s+(.+?)[\.!\?]?\s*$", stripped)
    if my_match:
        attr = my_match.group(1).strip()
        val = my_match.group(2).strip()
        if attr.lower() == "name":
            return f"The user's current name is {val}."
        if attr.lower().startswith("favorite "):
            return f"The user's {attr} is {val}."
        if attr.lower() == "goal":
            return f"The user's goal is {val}."
        return f"The user's {attr} is {val}."

    # Pattern 2: "I am / I'm" + present participle -> "The user is" + participle
    am_match = re.match(
        r"\bI(?:'m|\s+am)\s+(.+)$",
        stripped,
        re.IGNORECASE,
    )
    if am_match:
        return f"The user is {am_match.group(1).strip()}"

    # Pattern 3: "I have / I've" + past participle -> "The user has" + participle
    have_match = re.match(
        r"\bI(?:'ve|\s+have)\s+(.+)$",
        stripped,
        re.IGNORECASE,
    )
    if have_match:
        return f"The user has {have_match.group(1).strip()}"

    # Pattern 4: "I was" + ... -> "The user was" + ...
    was_match = re.match(
        r"\bI\s+was\s+(.+)$",
        stripped,
        re.IGNORECASE,
    )
    if was_match:
        return f"The user was {was_match.group(1).strip()}"

    # Pattern 5: "I'll / I will" + verb -> "The user will" + verb
    will_match = re.match(
        r"\bI(?:'ll|\s+will)\s+(.+)$",
        stripped,
        re.IGNORECASE,
    )
    if will_match:
        return f"The user will {will_match.group(1).strip()}"

    # Pattern 6: "I'd / I would" + verb -> "The user would" + verb
    would_match = re.match(
        r"\bI(?:'d|\s+would)\s+(.+)$",
        stripped,
        re.IGNORECASE,
    )
    if would_match:
        return f"The user would {would_match.group(1).strip()}"

    # Pattern 7: "I verb" (present tense) -> "The user verbs"
    i_present = re.match(
        r"\bI\s+(\w+)\b(.+)$",
        stripped,
        re.IGNORECASE,
    )
    if i_present:
        verb = i_present.group(1).lower()
        rest = i_present.group(2)

        _PRESENT_VERB_MAP: dict[str, str] = {
            "like": "likes",
            "prefer": "prefers",
            "love": "loves",
            "hate": "hates",
            "use": "uses",
            "work": "works",
            "study": "studies",
            "live": "lives",
            "eat": "eats",
            "play": "plays",
            "read": "reads",
            "write": "writes",
            "code": "codes",
            "build": "builds",
            "create": "creates",
            "make": "makes",
            "think": "thinks",
            "know": "knows",
            "believe": "believes",
            "feel": "feels",
            "hope": "hopes",
            "plan": "plans",
            "want": "wants",
            "need": "needs",
            "have": "has",
            "do": "does",
            "go": "goes",
            "say": "says",
            "get": "gets",
            "take": "takes",
            "see": "sees",
            "come": "comes",
        }

        # Past tense verbs stay the same in third person
        if verb.endswith("ed") or verb in (
            "began", "built", "went", "made", "took",
            "wrote", "spoke", "taught", "thought",
            "brought", "bought", "sent", "found",
            "left", "lost", "met", "ran", "sat",
            "stood", "told", "understood", "won",
            "started", "learned", "finished", "completed",
        ):
            verb_conjugated = verb
        else:
            verb_conjugated = _PRESENT_VERB_MAP.get(verb, verb + "s")

        return f"The user {verb_conjugated}{rest}"

    # Fallback: wrap with contextual prefix
    if category:
        cat_phrase = _category_to_phrase(category)
        return f"The user{cat_phrase}: {stripped}"
    return f"The user said: {stripped}"


def _category_to_phrase(category: str) -> str:
    mapping: dict[str, str] = {
        "Preference": "'s preference",
        "Project": "'s project",
        "Goal": "'s goal",
        "Skill": "'s skill",
        "Relationship": "'s relationship",
        "Fact": "",
        "Achievement": "'s achievement",
        "Event": "'s event",
        "Task": "'s task",
        "Other": "",
    }
    return mapping.get(category, "")


def normalize_memory_document(
    document: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Normalize a memory document to third-person user fact.

    Uses metadata.fact.memory_fact if available, then falls back to
    rule-based conversion.
    """
    if not document:
        return ""

    # If metadata has a pre-computed memory fact, use it
    if metadata:
        fact_data = metadata.get("fact")
        if isinstance(fact_data, dict):
            mf = fact_data.get("memory_fact", "")
            if mf and not is_first_person(mf):
                return mf

        # Check for top-level memory_fact in metadata
        mf = metadata.get("memory_fact", "")
        if mf and not is_first_person(mf):
            return mf

    # If already third-person, return as-is
    if not is_first_person(document):
        return document.strip()

    # Rule-based conversion
    category = ""
    if metadata:
        category = metadata.get("category", "")
    return normalize_to_third_person(document, category=category)
