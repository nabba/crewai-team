"""routing_overrides.py — Phase 5.2 fix for stale-context routing.

Two-layer defense against the failure mode that surfaced 2026-05-04:

  User: "what is my calendar tomorrow?"
  Bot: "PIM crew failed: NameError: optional_tool_group is not defined"
  [PR #50 lands; gateway restarted; PIM agent now constructs cleanly]
  User: "what is my calendar tomorrow?"
  Bot: "The PIM crew is currently broken (a code error: ...). I can't
        fetch your calendar until this is fixed. Please ask me to debug
        and fix the PIM crew, and then I can check your schedule for
        tomorrow."

The Commander LLM saw the prior failure message in conversation history
and emitted a routing decision of ``crew=direct`` with the response text
inline — refusing to dispatch to PIM even though the underlying bug had
been fixed minutes earlier. The user got a hallucinated refusal instead
of their calendar.

The fix is in two layers:

  **Layer 1 — Conversation-history sanitation** (``mark_stale_failures``)
  Before feeding history to the routing prompt, mark prior failure
  messages as ``[PRIOR — LIKELY RESOLVED, recent successful runs: ...]``
  when ``app.system_state`` shows the relevant crew has succeeded since.
  This makes the LLM less likely to extrapolate "X is still broken."

  **Layer 2 — Routing-decision validator** (``validate_routing_decision``)
  After the LLM emits a decision, scan for the refusal-as-direct
  pattern: ``crew=direct`` + response text containing refusal markers
  ("is broken", "I can't fetch", etc.) + a mentioned crew name. When
  detected, override and re-route to the mentioned crew with the
  original user input. The crew either succeeds (real answer) or
  produces a real error (better than a hallucinated refusal).

Why both layers
---------------
Defense in depth. Layer 1 reduces the LLM's *propensity* to hallucinate
the refusal in the first place. Layer 2 catches the cases where Layer 1
doesn't go far enough (e.g. when no successful runs exist yet to hint
at staleness, or when the LLM is stubborn).

Trade-offs
----------
Layer 2's override fires even when the crew genuinely IS broken — that's
intentional. A real attempt produces a real error message (the actual
NameError, the actual import failure) which the user can act on. A
hallucinated "is broken" doesn't tell the user *what* is broken or
*how* to act. Cost of an extra dispatch is low; benefit of always-fresh
failure surface is high.

Both layers are conservative on what they treat as "refusal" — they
only fire when both refusal markers AND a specific crew name appear
together. A response like "I'd need more information about which date
range you mean" doesn't trigger override; only responses like "PIM is
broken, want me to fix?" do.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# Refusal markers — phrases that signal the LLM is declining to act.
# Curated from real Commander outputs during the 2026-05-04 PIM
# incident plus expected variants.
_REFUSAL_MARKERS = (
    "is broken",
    "is currently broken",
    "still broken",
    "crew failed",
    "is not working",
    "is not connected",
    "i can't fetch",
    "i can't access",
    "i'm unable to",
    "i am unable to",
    "need to fix",
    "need to debug",
    "want me to debug",
    "want me to fix",
    "until this is fixed",
    "until that's fixed",
    "until that bug is fixed",
)


# Failure markers in conversation history — patterns indicating a
# prior assistant turn was a failure response, candidate for stale-
# tagging by Layer 1.
_FAILURE_PATTERNS = (
    re.compile(r"Crew \w+ failed:", re.IGNORECASE),
    re.compile(r"\bNameError:"),
    re.compile(r"\bImportError:"),
    re.compile(r"\bAttributeError:"),
    re.compile(r"Traceback \(most recent call last\):"),
    re.compile(r"is currently broken", re.IGNORECASE),
    re.compile(r"is not working", re.IGNORECASE),
    re.compile(r"want me to (debug|fix)", re.IGNORECASE),
)


# The set of crews the Commander can dispatch to. Sourced from the
# orchestrator's ``_VALID_CREWS`` list — kept in sync by importing
# at call time so we don't drift if the list changes.
def _valid_crews() -> frozenset[str]:
    """Return the set of dispatchable crews. Imported lazily so this
    module doesn't pull orchestrator at import time."""
    return frozenset({
        "research", "coding", "writing", "media",
        "creative", "pim", "financial", "desktop",
        "repo_analysis", "devops",
    })


# ── Layer 1: history sanitation ─────────────────────────────────────


def mark_stale_failures(
    history_text: str,
    *,
    system_state: dict[str, Any] | None = None,
) -> str:
    """Tag prior failure messages in conversation history as
    potentially-stale.

    Args:
        history_text: Raw conversation history string (multi-line,
            with assistant + user turns).
        system_state: Output of ``app.system_state.get_system_state()``.
            If None or unusable, returns history unchanged.

    Returns:
        The history with each detected failure line prefixed with a
        staleness tag when there's evidence the failure has been
        resolved. Lines without failure markers pass through unchanged.

    Conservative behavior: a failure message is tagged stale ONLY when
    ``system_state.recent_crew_runs`` shows at least one successful
    run for some crew. If no successful runs are visible (e.g. fresh
    gateway boot, buffer empty), the failure message is left as-is —
    we don't want to mislabel an unconfirmed failure as resolved.
    """
    if not history_text:
        return history_text
    if not system_state:
        return history_text

    crew_runs = (system_state.get("recent_crew_runs") or {})
    if not crew_runs.get("available"):
        return history_text  # buffer unavailable; can't determine staleness

    by_crew = crew_runs.get("by_crew", {})
    successful = sorted(
        crew for crew, runs in by_crew.items()
        if any(r.get("ok") for r in (runs or []))
    )

    if not successful:
        # No successful runs since gateway start — failures may be live.
        # Don't apply tags.
        return history_text

    out_lines: list[str] = []
    for line in history_text.split("\n"):
        is_failure = any(p.search(line) for p in _FAILURE_PATTERNS)
        if is_failure:
            out_lines.append(
                f"[PRIOR — LIKELY RESOLVED, recent successful runs: "
                f"{successful}] {line}"
            )
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


# ── Layer 2: routing-decision validator ─────────────────────────────


def detect_refusal_pattern(response_text: str) -> str | None:
    """If ``response_text`` looks like a "X crew is broken" refusal,
    return the crew name. Else None.

    Discriminator: response must contain BOTH a refusal marker AND a
    valid crew name (case-insensitive). A response with refusal markers
    but no specific crew (e.g. "I need more info") is NOT classified
    as this pattern — it's a legitimate non-dispatch decision.
    """
    if not response_text:
        return None
    lower = response_text.lower()
    if not any(m in lower for m in _REFUSAL_MARKERS):
        return None
    for crew in _valid_crews():
        # Use word-boundary check so "writing" doesn't match inside
        # "writing well" coincidentally — though for the refusal-pattern
        # case the crew name appears in semantically-loaded contexts
        # ("PIM crew", "the coding crew") so this is robust enough.
        if re.search(rf"\b{re.escape(crew)}\b", lower):
            return crew
    return None


def validate_routing_decision(
    decisions: list[dict[str, Any]],
    user_input: str,
    *,
    system_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Override hallucinated 'X is broken' refusals routed as direct.

    Iterates over routing decisions. For each ``crew=direct`` entry,
    inspects the inline response. When ``detect_refusal_pattern``
    returns a crew name, replaces the entry with a real dispatch
    to that crew using the original ``user_input`` as the task.

    Args:
        decisions: The list of decisions parsed from the LLM output.
        user_input: The original user message (replaces the hallucinated
            response as the dispatched task).
        system_state: Optional, included for telemetry/logging only.
            The validator runs regardless of state (the override is
            unconditional once the refusal pattern is detected).

    Returns:
        A new list with overrides applied. Decisions that don't match
        the refusal pattern pass through unchanged.
    """
    out: list[dict[str, Any]] = []
    for d in decisions:
        if d.get("crew") != "direct":
            out.append(d)
            continue
        response = str(d.get("task", ""))
        crew = detect_refusal_pattern(response)
        if crew is None:
            out.append(d)
            continue
        # Telemetry context — what the LLM was actually claiming
        recent_success = ""
        if system_state:
            crew_runs = (system_state.get("recent_crew_runs") or {})
            by_crew = crew_runs.get("by_crew", {}).get(crew, [])
            if by_crew:
                last_ok = next((r for r in by_crew if r.get("ok")), None)
                if last_ok:
                    recent_success = f" (last {crew} success at {last_ok.get('ts')})"
        logger.warning(
            "routing_overrides: detected refusal-as-direct mentioning "
            "crew=%s; overriding to actual dispatch with user_input=%r%s. "
            "Hallucinated response was: %s",
            crew, user_input[:80], recent_success, response[:200],
        )
        out.append({
            "crew": crew,
            "task": user_input,
            "difficulty": d.get("difficulty", 5),
        })
    return out
