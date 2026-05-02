"""
capability_router.py — between routing decision and dispatch.

Built for the 2026-05-02 audit (Week 3 / Shift 2).  Phase 3 of the audit
found that the commander's routing was not capability-aware: it chose
crews based on track record + ToM heuristics + a hand-tuned LLM prompt,
but never checked whether the routed-to crew actually had the tools the
task needs.  Result: dispatches went to the coding crew for satellite-
imagery work, the coding crew lacked ``gee_run_script``, and the
dispatch failed silently for 20 minutes.

Week 2's declarative tool registry (``register_tool_factory``) made
per-crew capabilities computable from the source of truth — see
``base_crew.get_crew_capabilities``.  This module closes the loop:

  1. ``categorize_task(task_text)`` — small LLM call that returns the
     subset of capability categories the task plausibly requires.
  2. ``precondition_check(crew_name, task_text)`` — combines the above
     with ``get_crew_capabilities(crew_name)`` and reports whether the
     crew can actually do the work.

The orchestrator wires these in between routing and dispatch.  When the
check fails, the orchestrator can: (a) escalate to a different crew
that DOES have the categories, (b) inject the missing tools into the
routed crew (where possible), or (c) report "missing capability" to
the user with a suggestion.  Today we ship (a) + diagnostic logging
for (c); (b) is deferred until we see how the user wants graceful
degradation to behave.

The classifier itself is local-LLM (qwen3.5:35b via Ollama) by default
— the call is small (~50 tokens completion) and runs in <2s, well
inside the routing budget.  Cloud fallback if local is down.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.crews.base_crew import (
    CAPABILITY_DESCRIPTIONS,
    get_all_crew_capabilities,
    get_crew_capabilities,
)

logger = logging.getLogger(__name__)


# Token-budget cap for the classifier completion.  ~50 tokens covers
# 3-5 category names + JSON wrapping; longer responses are truncated.
_CLASSIFIER_MAX_TOKENS = 80

# Cap on returned categories — prevents the LLM from claiming "this
# task needs everything", which makes the precondition check vacuous.
_MAX_CATEGORIES = 4


# Cache classifier responses by exact task text — same task in a session
# shouldn't burn two LLM calls.  Bounded to prevent unbounded growth.
_CLASSIFIER_CACHE: dict[str, list[str]] = {}
_CLASSIFIER_CACHE_CAP = 256


def _build_classifier_prompt(task_text: str) -> str:
    """Render the classifier's user-input.  System content is set in
    ``categorize_task`` so the LLM provider sees a clean role split."""
    cats_lines = []
    for cat, desc in CAPABILITY_DESCRIPTIONS.items():
        cats_lines.append(f"  - {cat}: {desc}")
    cats_block = "\n".join(cats_lines)
    return (
        "You decide which capability categories a user task requires.\n\n"
        "Categories (with one-line descriptions):\n"
        f"{cats_block}\n\n"
        f"User task: {task_text[:600]}\n\n"
        f"Return ONLY a JSON array of {_MAX_CATEGORIES} or fewer category "
        "names that this task plausibly REQUIRES.  Be strict — only include "
        "a category if the task cannot reasonably be completed without it.  "
        "Example output: [\"geospatial\", \"document_gen\"]"
    )


def _parse_classifier_output(raw: str) -> list[str]:
    """Extract the JSON array of category names from the LLM output.

    Tolerant of common shapes — bare JSON array, JSON-in-code-block,
    LLMs that wrap with extra prose.  Returns an empty list when nothing
    parseable is found (caller treats empty as "no capability info,
    skip the precondition check").
    """
    if not raw:
        return []
    # Try direct JSON parse first
    text = raw.strip()
    # Strip code fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", text, flags=re.MULTILINE).strip()
    # Find the first JSON array in the text
    m = re.search(r"\[[^\[\]]*\]", text)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    out: list[str] = []
    valid = set(CAPABILITY_DESCRIPTIONS)
    for item in parsed:
        if not isinstance(item, str):
            continue
        cat = item.strip().lower()
        if cat in valid and cat not in out:
            out.append(cat)
        if len(out) >= _MAX_CATEGORIES:
            break
    return out


def categorize_task(task_text: str) -> list[str]:
    """Return the capability categories *task_text* requires.

    Uses a small local-LLM call (qwen3.5:35b via Ollama by default)
    with a short prompt.  Cached by exact task text to amortize the
    cost across reroutes / retries.

    Returns an empty list when the LLM is unavailable, returns
    nothing parseable, or names no recognized categories — callers
    should treat empty as "no capability info, skip the check"
    rather than "task needs nothing".
    """
    if not task_text or not task_text.strip():
        return []
    key = task_text.strip()[:600]
    cached = _CLASSIFIER_CACHE.get(key)
    if cached is not None:
        return list(cached)

    try:
        from app.llm_factory import create_specialist_llm
        llm = create_specialist_llm(
            max_tokens=_CLASSIFIER_MAX_TOKENS,
            role="planner",
            task_hint="capability-classification",
        )
        prompt = _build_classifier_prompt(key)
        raw = str(llm.call(prompt)).strip()
    except Exception as exc:
        logger.warning(
            "capability_router.categorize_task: LLM call failed (%s); "
            "returning empty (precondition check will be skipped)",
            exc,
        )
        return []

    cats = _parse_classifier_output(raw)
    # Cache regardless of result — even an empty result avoids re-trying
    # a degenerate parse on the same task.
    _CLASSIFIER_CACHE[key] = cats
    while len(_CLASSIFIER_CACHE) > _CLASSIFIER_CACHE_CAP:
        _CLASSIFIER_CACHE.pop(next(iter(_CLASSIFIER_CACHE)))

    if cats:
        logger.info(
            "capability_router: task '%s...' → required=%s",
            key[:60], cats,
        )
    return cats


def precondition_check(
    crew_name: str,
    task_text: str,
) -> tuple[bool, list[str], list[str]]:
    """Check whether *crew_name* has every capability *task_text* needs.

    Returns ``(passed, required_categories, missing_categories)``:
      * ``passed`` — True when missing is empty OR the classifier
        returned nothing (we can't check what we don't know).
      * ``required_categories`` — what the task needs (from classifier).
      * ``missing_categories`` — required minus the crew's actual
        capabilities (from the registry).
    """
    required = categorize_task(task_text)
    if not required:
        # Classifier had nothing useful — fall through, don't block.
        return True, [], []
    available = get_crew_capabilities(crew_name)
    missing = [c for c in required if c not in available]
    passed = not missing
    if not passed:
        logger.warning(
            "precondition_check: crew=%s missing=%s required=%s available=%s",
            crew_name, missing, required, sorted(available),
        )
    return passed, required, missing


def find_crew_for_categories(
    required: list[str],
    *,
    exclude: tuple[str, ...] = (),
) -> str | None:
    """Return the first crew (in CREW_TO_AGENTS order) whose
    capabilities cover every category in *required*.

    Used by the orchestrator's escalation path — when a routing
    decision fails the precondition check, find a crew that DOES
    have what's needed.  ``exclude`` lets the caller skip crews
    already tried (e.g. the original failed routing target).

    Returns None when no crew has everything required — caller
    should fall back to legacy behaviour or report to user.
    """
    if not required:
        return None
    req_set = set(required)
    crew_caps = get_all_crew_capabilities()
    for crew, caps in crew_caps.items():
        if crew in exclude:
            continue
        if req_set.issubset(caps):
            return crew
    return None
