"""
retry_decision.py — single decision point for "what to do after vetting".

Built for Week 4 of the 2026-05-02 audit (orchestrator reroute structural
rewrite).  Pre-Week-4 the orchestrator's vetting-failed branch had:

  * A keyword scan (_vetting_signals_wrong_crew) for "wrong crew" verdicts
  * A reroute via _route(exclude_crew=...) on wrong-crew signals
  * A post-filter that dropped the excluded crew from the router output
  * A fallback that re-ran the SAME excluded crew when the post-filter
    left the list empty (Week 2.6 Fix B finally caught this and made it
    deliver-as-is instead)
  * A separate Theory-of-Mind swap rule that wanted to act on track-record
    but was blocked by a `cross-task-type` guard (added in Apr 20's
    `1c64005` patch)
  * A re-vet of the retry result that compared against the original
    by length-ratio heuristic
  * A retry-attempt flag (`self._vetting_retry_attempted`) to prevent
    infinite loops

Each rule fired independently in a long if/elif/else.  The Estonia
verification dispatches showed the rules contradicting each other:
vetting said WRONG CREW → reroute → LLM router stubbornly picked the
excluded crew → post-filter dropped → fallback ran the same crew with
the original retry_task → silent loop.  Week 2.5 + 2.6 patched the
specific failure mode but left the structural mess in place.

Week 4 collapses the rules into ONE decision function that returns ONE
typed action.  The orchestrator's vetting-failed block becomes:

    action, target_crew, reason = decide_retry(
        crew_name=crew_name,
        difficulty=difficulty,
        vet_passed=_vet_passed,
        vet_issues=_vet_issues,
        attempt_count=int(self._vetting_retry_attempted),
        original_task=user_input,
        retry_task_text=d.get("task", user_input),
    )

The function also produces a single-line `reason` string for telemetry,
so we can later mine the disagreement pattern (e.g. "vetting said
wrong-crew but capability-router said same-crew is fine").
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# Phrases that signal "wrong crew" from a vetting verdict.  Lifted
# verbatim from the prior _vetting_signals_wrong_crew so behaviour is
# byte-equivalent on the keyword side; the new function ALSO consults
# the capability router (Week 3) so a vetting verdict alone no longer
# determines the reroute.
_WRONG_CREW_PHRASES = (
    "wrong crew",
    "should be routed to",
    "not the right crew",
    "should have been",
    "expected the {} crew",  # e.g. "expected the research crew"
    "needed a {} crew",
)


class RetryAction(str, Enum):
    """What the orchestrator should do next.

    KEEP_ORIGINAL    — vet passed (or we've exhausted retries).  Deliver
                       the current final_result as-is.
    RETRY_SAME_CREW  — same crew, but with a reflexion hint built from
                       vet_issues.  Use when vet failed on data quality
                       (truncation, missing sources) but the crew choice
                       seems fine.
    RETRY_DIFFERENT_CREW
                     — vet flagged WRONG CREW AND we have a target crew
                       (from capability router or fallback heuristic).
                       Target crew name is in the second tuple slot.
    DELIVER_WITH_NOTE
                     — already retried once + still failing.  Deliver
                       the original with a "vetting flagged: ..." caveat
                       appended.  Beats looping forever.
    """
    KEEP_ORIGINAL = "keep_original"
    RETRY_SAME_CREW = "retry_same_crew"
    RETRY_DIFFERENT_CREW = "retry_different_crew"
    DELIVER_WITH_NOTE = "deliver_with_note"


@dataclass(frozen=True)
class RetryDecision:
    action: RetryAction
    target_crew: Optional[str]
    reason: str


def _vetting_signals_wrong_crew(vet_text: str, crew_name: str) -> bool:
    """Keyword scan over the vetting verdict text.  Same heuristic as
    the pre-Week-4 _vetting_signals_wrong_crew in orchestrator.py —
    kept here so the new module is the single source of truth for
    retry-decision inputs."""
    if not vet_text:
        return False
    text = vet_text.lower()
    for phrase_template in _WRONG_CREW_PHRASES:
        phrase = phrase_template.format("X")  # neutralise the format slot
        # Match phrases where {} was filled with a crew name OR the
        # template has no slot.
        if "{}" in phrase_template:
            # Try common crew names in the slot
            for c in ("research", "coding", "writing", "media", "creative",
                      "pim", "financial", "desktop", "repo_analysis", "devops"):
                if c == crew_name:
                    continue
                if phrase_template.format(c) in text:
                    return True
        else:
            if phrase in text:
                return True
    # Final fallback — explicit "WRONG CREW" verdict markers
    return "wrong crew" in text


def _build_reflexion_hint(vet_issues: list[str], max_chars: int = 1200) -> str:
    """Render vet_issues as a compact reflexion hint to prepend to the
    retry task.  Bounded so the retry prompt doesn't blow up."""
    if not vet_issues:
        return ""
    bullets = []
    for issue in vet_issues[:6]:
        clean = " ".join(str(issue).split())
        if not clean:
            continue
        bullets.append(f"  - {clean[:200]}")
    if not bullets:
        return ""
    block = "\n".join(bullets)
    return (
        "VETTING FLAGGED THESE ISSUES IN YOUR PREVIOUS ATTEMPT — "
        "address each before re-submitting:\n"
        f"{block}\n\n"
    )[:max_chars]


def decide_retry(
    *,
    crew_name: str,
    difficulty: int,
    vet_passed: bool,
    vet_text: str = "",
    vet_issues: Optional[list[str]] = None,
    attempt_count: int = 0,
    capability_router=None,  # callable: (crew_name, task_text) -> alt_crew | None
    task_text: str = "",
) -> RetryDecision:
    """Decide what the orchestrator should do after vetting.

    Single decision point that consumes every signal we have today
    (vetting outcome + verdict-text wrong-crew scan + capability
    router suggestion) and returns one typed RetryDecision.

    Parameters
    ----------
    crew_name : str
        The crew that just produced the answer being vetted.
    difficulty : int
        Task difficulty (1-10).  Retries only fire at difficulty ≥ 7
        — lower-stakes tasks deliver the current result regardless.
    vet_passed : bool
        Vetting outcome.  True → KEEP_ORIGINAL immediately.
    vet_text : str
        Full text of the vetting verdict; scanned for wrong-crew
        phrases.  Optional but improves precision.
    vet_issues : list[str]
        Structured issues from vetting (used to build the reflexion
        hint for RETRY_SAME_CREW).
    attempt_count : int
        How many times we've already retried this dispatch.  ≥ 1
        forces DELIVER_WITH_NOTE — we don't loop forever.
    capability_router : callable | None
        Optional function ``(crew_name, task_text) -> Optional[str]``
        returning an alternative crew name from the Week 3 capability
        router.  When provided AND vetting failed AND the alternative
        differs from crew_name, takes precedence over the keyword
        wrong-crew scan.
    task_text : str
        The task being retried; passed to capability_router.
    """
    if vet_passed:
        return RetryDecision(
            RetryAction.KEEP_ORIGINAL, None,
            "vetting passed — no retry needed",
        )
    if difficulty < 7:
        return RetryDecision(
            RetryAction.KEEP_ORIGINAL, None,
            f"difficulty {difficulty} below retry floor (7)",
        )
    if attempt_count >= 1:
        return RetryDecision(
            RetryAction.DELIVER_WITH_NOTE, None,
            f"already retried {attempt_count} time(s); avoid loop",
        )

    # First retry attempt — figure out same-crew vs different-crew.
    issues = vet_issues or []
    capability_alt: Optional[str] = None
    if capability_router is not None:
        try:
            capability_alt = capability_router(crew_name, task_text)
        except Exception as exc:
            logger.debug("capability_router lookup failed: %s", exc)
            capability_alt = None

    keyword_wrong_crew = _vetting_signals_wrong_crew(vet_text, crew_name)

    if capability_alt and capability_alt != crew_name:
        reason_bits = []
        if keyword_wrong_crew:
            reason_bits.append("vet keyword=wrong_crew")
        reason_bits.append(f"capability router → {capability_alt}")
        return RetryDecision(
            RetryAction.RETRY_DIFFERENT_CREW, capability_alt,
            "; ".join(reason_bits),
        )
    if keyword_wrong_crew:
        # Vetting said wrong crew but capability router has no
        # alternative — log the disagreement (or absence) and retry
        # same-crew.  Better to re-attempt with reflexion than to
        # ABANDON when no other crew can do the work.
        return RetryDecision(
            RetryAction.RETRY_SAME_CREW, None,
            "vet keyword=wrong_crew but no capability alternative; "
            "retrying same crew with reflexion",
        )
    return RetryDecision(
        RetryAction.RETRY_SAME_CREW, None,
        f"vet failed with {len(issues)} issue(s); retrying same crew "
        "with reflexion hint",
    )


def build_retry_task(
    original_task: str,
    vet_issues: list[str] | None = None,
) -> str:
    """Compose the retry task text — original task plus reflexion hint
    derived from vet_issues.  Returns the task text only (no prefix);
    caller wraps it in their own `Implement the following...` template.
    """
    hint = _build_reflexion_hint(vet_issues or [])
    if not hint:
        return original_task
    return f"{hint}{original_task}"
