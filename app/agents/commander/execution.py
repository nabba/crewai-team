import logging
import re

logger = logging.getLogger(__name__)

# Patterns that indicate a failed or low-quality output
_QUALITY_FAILURE_PATTERNS = [
    re.compile(r"^I (?:cannot|can't|am unable to|don't|apologize)", re.IGNORECASE),
    re.compile(r"^(?:sorry|apologies|unfortunately),?\s+I", re.IGNORECASE),
    re.compile(r"^As an AI", re.IGNORECASE),
    re.compile(r"^\{.*\}$", re.DOTALL),  # raw JSON
    re.compile(r"^Traceback \(most recent call", re.IGNORECASE),
]

# Patterns that detect meta-commentary / planning output instead of actual content.
# These fire when the model describes what it WILL do rather than producing the answer.
_META_COMMENTARY_PATTERNS = [
    re.compile(r"(?:moving forward|next,?)\s+I\s+will\b", re.IGNORECASE),
    re.compile(r"^(?:the\s+)?(?:unified|final|compiled)\s+(?:research\s+)?report\s+(?:synthesizes|combines|compiles)", re.IGNORECASE),
    re.compile(r"\bI\s+will\s+(?:now\s+)?(?:assess|evaluate|review|reflect|analyze|proceed)\b", re.IGNORECASE),
    re.compile(r"^(?:I'll|I\s+will)\s+(?:start|begin|proceed)\s+(?:by|with|to)\b", re.IGNORECASE),
    re.compile(r"^(?:Let me|Allow me to)\s+(?:now\s+)?(?:assess|evaluate|compile|synthesize|review)", re.IGNORECASE),
    re.compile(r"^Here(?:'s| is) (?:my|the) (?:plan|approach|strategy)\b", re.IGNORECASE),
]


def _passes_quality_gate(result: str, crew_name: str) -> bool:
    """Quick heuristic quality check — no LLM call.

    Returns True if the result appears to be usable output.
    """
    if not result or len(result.strip()) < 20:
        return False

    text = result.strip()
    for pattern in _QUALITY_FAILURE_PATTERNS:
        if pattern.match(text):
            return False

    # Detect meta-commentary: model describing what it will do instead of doing it.
    # Short outputs (<400 chars) that match meta patterns are almost always junk.
    if len(text) < 400:
        for pattern in _META_COMMENTARY_PATTERNS:
            if pattern.search(text):
                logger.info(f"quality_gate: meta-commentary detected ({len(text)} chars)")
                return False

    # For coding tasks, expect at least a code block or code-like content
    if crew_name == "coding":
        has_code = "```" in text or "def " in text or "function " in text or "class " in text
        if not has_code and len(text) < 100:
            return False

    return True


def _generate_reflection(
    task: str, result: str, crew_name: str, trial: int
) -> str:
    """Generate a heuristic reflection on a failed output — no LLM call.

    Returns a concise reflection string that gets injected into the next attempt.
    """
    if not result or len(result.strip()) < 5:
        return (
            f"Trial {trial} produced empty or near-empty output. "
            "Try a more detailed, step-by-step approach."
        )

    text = result.strip()

    # Check for refusal patterns
    if any(p.match(text) for p in _QUALITY_FAILURE_PATTERNS[:3]):
        return (
            f"Trial {trial} produced a refusal or apology. "
            "Rephrase the task more specifically. "
            "Focus on what CAN be done rather than limitations."
        )

    # Check for raw JSON / traceback
    if text.startswith("{") or text.startswith("Traceback"):
        return (
            f"Trial {trial} returned raw technical output instead of a useful response. "
            "Format the output as clear, human-readable text."
        )

    # Check for meta-commentary (model describing what it will do instead of doing it)
    if any(p.search(text) for p in _META_COMMENTARY_PATTERNS):
        return (
            f"Trial {trial} produced meta-commentary describing WHAT you will do "
            "instead of actually doing it. Do NOT describe your plan or approach. "
            "Produce the ACTUAL content directly — the research report, the answer, "
            "the code. Start with the content itself, not a description of it."
        )

    # For coding with no code
    if crew_name == "coding" and "```" not in text:
        return (
            f"Trial {trial} did not include a code block. "
            "Include executable code in a ``` code block. "
            "Test the code before returning."
        )

    # Generic quality issue
    return (
        f"Trial {trial} output did not meet quality standards "
        f"({len(text)} chars). Try a fundamentally different approach — "
        "not a minor variation of the same strategy."
    )


def _load_past_reflexion_lessons(task: str, n: int = 3) -> list[str]:
    """Load relevant past reflexion lessons from memory."""
    try:
        from app.memory.scoped_memory import retrieve_operational
        return retrieve_operational("scope_reflexion_lessons", task, n)
    except Exception:
        logger.debug("Failed to load reflexion lessons", exc_info=True)
        return []


def _store_reflexion_success(task: str, trials: int, reflections: list[str]) -> None:
    """Store a successful reflexion outcome as a reusable lesson."""
    try:
        from app.memory.scoped_memory import store_scoped
        lesson = (
            f"SUCCESS after {trials} trials: "
            f"Task: {task[:200]}. "
            f"Winning reflection: {reflections[-1][:300] if reflections else 'N/A'}"
        )
        store_scoped(
            "scope_reflexion_lessons", lesson,
            {"type": "success", "trials": str(trials)},
            importance="high",
        )
        logger.info(f"Reflexion: stored success lesson after {trials} trials")
    except Exception:
        logger.debug("Failed to store reflexion success", exc_info=True)


def _store_reflexion_failure(task: str, trials: int, reflections: list[str]) -> None:
    """Store a failed reflexion outcome as an antipattern."""
    try:
        from app.memory.scoped_memory import store_scoped
        antipattern = (
            f"FAILURE after {trials} trials: "
            f"Task: {task[:200]}. "
            f"Reflections: {'; '.join(r[:100] for r in reflections)}"
        )
        store_scoped(
            "scope_reflexion_lessons", antipattern,
            {"type": "failure", "trials": str(trials)},
            importance="high",
        )
        logger.info(f"Reflexion: stored failure antipattern after {trials} trials")
    except Exception:
        logger.debug("Failed to store reflexion failure", exc_info=True)


def _run_proactive_scan(result: str, crew_names: str, user_input: str) -> str:
    """Run proactive trigger scan and return notes string (or empty)."""
    try:
        from app.proactive.trigger_scanner import scan_for_triggers, execute_proactive_action
        triggers = scan_for_triggers(
            crew_results={"result": result, "crews": crew_names},
            task_description=user_input,
        )
        notes = []
        for trigger in triggers[:2]:
            logger.info(f"Proactive trigger: {trigger['trigger_type']}: {trigger['description'][:80]}")
            addition = execute_proactive_action(trigger, result)
            if addition:
                notes.append(addition)
        return "\n".join(notes)
    except Exception:
        logger.debug("Proactive scan failed", exc_info=True)
        return ""
