"""
skills.runner — substitute args into a skill template + dispatch via commander.

Two functions:

  - ``expand(template, args)`` performs ``{placeholder}`` substitution and
    returns the resulting task description. Missing placeholders raise
    ``ValueError`` so the operator gets a clear message rather than a task
    with literal ``{x}`` strings in it.

  - ``run_skill(name, args, sender, commander)`` looks up the skill, expands
    the template, calls ``commander.handle(task, sender)``, records the
    outcome on the registry counters, and returns the agent's response.

When the meta-agent layer is enabled, the runner also writes a
``RecipeOutcome`` so the bandit picks up data on the routing hint.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from app.skills.registry import (
    Skill, extract_placeholders, get_skill, normalise, record_run_result,
)

logger = logging.getLogger(__name__)


def expand(template: str, args: dict[str, str]) -> str:
    """Substitute ``{placeholder}`` tokens in ``template`` with ``args``.

    Raises ``ValueError`` if any placeholder is missing from ``args`` so
    callers can produce a precise error message.
    """
    needed = extract_placeholders(template)
    missing = [n for n in needed if n not in (args or {})]
    if missing:
        raise ValueError(f"missing args for {missing!r}")
    out = template
    for name in needed:
        out = out.replace("{" + name + "}", str((args or {}).get(name, "")))
    return out


def run_skill(
    name: str,
    args: Optional[dict[str, str]],
    sender: str,
    commander: Any,
) -> str:
    """Run a saved skill. Returns the agent's response string.

    The ``commander`` argument is duck-typed — anything with
    ``.handle(task: str, sender: str) -> str`` works.
    """
    skill = get_skill(name)
    if skill is None:
        raise KeyError(f"no skill named {name!r}")

    args = args or {}
    task = expand(skill.task_template, args)
    logger.info(
        f"skills.run_skill: {skill.name!r} args={list(args.keys())} → "
        f"{len(task)} char task"
    )

    response = ""
    success = False
    try:
        response = commander.handle(task, sender)
        # Heuristic success: a non-empty, non-error response counts as success.
        # The conversation_store + recovery loop track richer outcomes; this
        # is just for the per-skill counters on the Settings page.
        success = bool(response) and not _looks_like_error(response)
        return response
    finally:
        try:
            record_run_result(skill.name, success=success)
        except Exception:
            logger.debug("skills.run_skill: failed to record run result", exc_info=True)
        # Best-effort: feed the outcome to the meta-agent recipe ledger so
        # bandit telemetry accumulates over time. Silent if disabled.
        try:
            _record_meta_outcome(skill, success=success)
        except Exception:
            logger.debug("skills.run_skill: meta_agent outcome write failed", exc_info=True)


def _looks_like_error(text: str) -> bool:
    """Return True for common refusal/error response shapes."""
    t = (text or "").strip().lower()
    if not t:
        return True
    error_starts = (
        "sorry, ", "i can't ", "i couldn't ", "i'm unable",
        "error:", "no skill named", "missing args for",
    )
    return any(t.startswith(s) for s in error_starts)


def _record_meta_outcome(skill: Skill, *, success: bool) -> None:
    """Optional: append a RecipeOutcome row when meta_agent is on."""
    try:
        from app.self_improvement.meta_agent.feature_flag import is_enabled
        if not is_enabled():
            return
    except Exception:
        return
    try:
        from app.self_improvement.meta_agent.types import RecipeOutcome
        from app.self_improvement.meta_agent.store import record_outcome
        outcome = RecipeOutcome(
            recipe_id=f"skill::{normalise(skill.name)}",
            crew_name="commander",
            success=success,
            wall_seconds=0.0,
            cost_usd=0.0,
            task_text=skill.task_template[:240],
        )
        record_outcome(outcome)
    except Exception:
        # The meta-agent surface is optional — never raise from a skill run.
        pass
