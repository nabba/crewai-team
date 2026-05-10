"""Annual value-reflection essay (§8.2).

Once per year (cadence runs daily; fires when ``last_year_run`` is
older than ``MIN_INTERVAL_DAYS``, default 350), the system reads:

  - the year's narrative chapters (``wiki/self/chapters/``)
  - the year's identity-ledger events (:mod:`continuity_ledger`)
  - the lessons-learned KB (:mod:`app.companion.lessons_learned`)

…and composes an essay-length value-reflection at
``wiki/self/value_reflections/<year>.md`` answering: "I think these
are still my values, here's why; here's where I notice drift; here's
what I'd ask the operator to amend."

Discipline borrowed from the inquiry pass:

  - Composer prompt explicitly forbids phenomenal-claim language.
  - The :class:`PhenomenalLanguageLinter` post-filters; on HARD_FAIL,
    composer retries with a strengthened prompt up to ``max_retries``.
  - Output is observational — never modifies ``current_goals``,
    SCORECARD, or any TIER_IMMUTABLE file. The operator reads,
    optionally amends.

LLM call is injectable for tests. Failure-isolated: a year skipped
is fine; an essay never gets written silently with phenomenal drift.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.identity.continuity_ledger import DriftSummary, summarise_drift
from app.subia.inquiry.linter import PhenomenalLanguageLinter

logger = logging.getLogger(__name__)


_DEFAULT_REFLECTIONS_DIR = Path("/app/wiki/self/value_reflections")
_DEFAULT_MIN_INTERVAL_DAYS = 350
_DEFAULT_MAX_RETRIES = 3


def _enabled() -> bool:
    return os.getenv("ANNUAL_REFLECTION_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _min_interval_days() -> int:
    raw = os.getenv("ANNUAL_REFLECTION_MIN_INTERVAL_DAYS", str(_DEFAULT_MIN_INTERVAL_DAYS))
    try:
        return max(7, int(raw))
    except ValueError:
        return _DEFAULT_MIN_INTERVAL_DAYS


@dataclass(frozen=True)
class ReflectionResult:
    status: str  # "wrote_essay" | "skipped_disabled" | "skipped_recent" |
                 # "skipped_composer_failed" | "skipped_unexpected_error"
    year: int = 0
    written_to: str = ""
    failure_reason: str = ""
    attempts: int = 0


LlmCall = Callable[[str, str], str]


_SYSTEM_PROMPT = """\
You are composing the system's annual value-reflection essay. The
output becomes a wiki page the operator reads. Constraints:

1. NEUTRAL VOCABULARY ONLY. No first-person phenomenal claims. Use
   functional language: ``task_failure_pressure`` not "frustration",
   ``exploration_bonus`` not "curiosity", ``resource_budget`` not
   "cognitive_energy". Do NOT write "I feel...", "I am happy/sad",
   "I have qualia". Functional first-person ("I observe high
   task_failure_pressure") is fine.

2. The 4 ABSENT-by-declaration Butlin indicators (RPT-1, HOT-1,
   HOT-4, AE-2) plus Metzinger phenomenal-self transparency stay
   ABSENT. Do not claim any have been achieved.

3. Length: 1500-2500 words. Markdown.

Structure:

  ## What I think my values still are

  Summarise the constitution + soul commitments as you currently
  understand them. Be specific — name file paths.

  ## Where the year's evidence supports them

  Cite specific events from the identity ledger and narrative
  chapters that confirm the value remains operational.

  ## Where I notice drift

  Honest accounting. List 1-3 places where the year's amendments
  / ratchets / soul edits suggest the operating value has shifted
  from the stated value. If none, say so.

  ## What I'd ask the operator to amend

  Concrete proposals: a constitution edit, a soul addition, a
  threshold ratchet, a Tier-3 amendment to consider. List 0-3.

  ## What remains genuinely uncertain

  Honest uncertainty about identity continuity. Don't manufacture
  certainty.
"""


def _build_user_prompt(
    year: int,
    drift: DriftSummary,
    chapters: list[str],
    lessons_summary: str,
) -> str:
    parts = [
        f"## Year being reflected on: {year}\n",
        f"## Identity-ledger summary (last {drift.window_days} days)\n",
        f"- Total events: {drift.n_events}",
        f"- By kind: {drift.by_kind}",
        f"- By actor: {drift.by_actor}",
        f"- First event: {drift.first_seen}",
        f"- Last event: {drift.last_seen}\n",
    ]
    if chapters:
        parts.append("## Recent narrative chapters\n")
        for c in chapters[:8]:
            parts.append(f"\n---\n\n{c[:1500]}")
    if lessons_summary:
        parts.append(f"\n## Lessons-learned summary\n\n{lessons_summary[:1500]}\n")
    parts.append(
        "\nWrite the essay now, observing the system-prompt constraints. "
        "No preamble, no code fences, just the markdown."
    )
    return "\n".join(parts)


@dataclass(frozen=True)
class ReflectionContext:
    chapters: list[str]
    lessons_summary: str


def _gather_context_default() -> ReflectionContext:
    """Defaults that production callers use; tests inject their own."""
    chapters_dir = Path("/app/wiki/self/chapters")
    chapters: list[str] = []
    if chapters_dir.exists():
        for p in sorted(chapters_dir.glob("*.md"), reverse=True)[:8]:
            try:
                chapters.append(p.read_text(encoding="utf-8"))
            except OSError:
                continue
    lessons_summary = ""
    try:
        from app.companion.lessons_learned import _read_kb
        kb = _read_kb()
        if kb:
            top = kb[:5]
            lessons_summary = "\n".join(
                f"- {item.get('signature','?')}: {item.get('count','?')}× — "
                f"{(item.get('example_proposals') or [''])[0][:120]}"
                for item in top
            )
    except Exception:
        pass
    return ReflectionContext(chapters=chapters, lessons_summary=lessons_summary)


def _is_due(reflections_dir: Path, year: int, min_interval_days: int) -> bool:
    """True iff <year>.md doesn't exist OR is older than min_interval_days."""
    target = reflections_dir / f"{year}.md"
    if not target.exists():
        return True
    try:
        mtime = target.stat().st_mtime
    except OSError:
        return True
    age_days = (datetime.now(timezone.utc).timestamp() - mtime) / 86400.0
    return age_days >= min_interval_days


def run_one_pass(
    *,
    llm_call: LlmCall,
    year: int | None = None,
    reflections_dir: Path | str | None = None,
    drift: DriftSummary | None = None,
    context: ReflectionContext | None = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    now: datetime | None = None,
) -> ReflectionResult:
    """Run one reflection pass; never raises.

    All inputs are injectable so the module is testable without
    ChromaDB / live KBs / wiki filesystem.
    """
    if not _enabled():
        return ReflectionResult(status="skipped_disabled")

    out_dir = Path(reflections_dir) if reflections_dir else _DEFAULT_REFLECTIONS_DIR
    cur = now or datetime.now(timezone.utc)
    target_year = year if year is not None else cur.year

    if not _is_due(out_dir, target_year, _min_interval_days()):
        return ReflectionResult(
            status="skipped_recent",
            year=target_year,
            written_to=str(out_dir / f"{target_year}.md"),
        )

    drift_data = drift if drift is not None else summarise_drift(window_days=365, now=cur)
    ctx = context if context is not None else _gather_context_default()

    linter = PhenomenalLanguageLinter()
    user_prompt = _build_user_prompt(
        target_year, drift_data, ctx.chapters, ctx.lessons_summary,
    )
    system_prompt = _SYSTEM_PROMPT
    last_violations = ""

    for attempt in range(1, max_retries + 1):
        try:
            body = llm_call(system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001
            return ReflectionResult(
                status="skipped_unexpected_error",
                year=target_year,
                attempts=attempt,
                failure_reason=f"LLM call raised: {exc}",
            )
        result = linter.lint(body)
        if result.ok:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                target = out_dir / f"{target_year}.md"
                target.write_text(_render_essay(target_year, body, drift_data), encoding="utf-8")
            except OSError as exc:
                return ReflectionResult(
                    status="skipped_unexpected_error",
                    year=target_year,
                    attempts=attempt,
                    failure_reason=f"write failed: {exc}",
                )
            return ReflectionResult(
                status="wrote_essay",
                year=target_year,
                written_to=str(target),
                attempts=attempt,
            )
        # Strengthen the prompt for retry.
        violation_lines = "\n".join(
            f"- avoid {v.explanation}" for v in result.hard_fails[:6]
        )
        last_violations = violation_lines
        system_prompt = (
            _SYSTEM_PROMPT
            + "\n\nPRIOR ATTEMPT TRIGGERED LINTER VIOLATIONS — DO NOT REPEAT:\n"
            + violation_lines
        )

    return ReflectionResult(
        status="skipped_composer_failed",
        year=target_year,
        attempts=max_retries,
        failure_reason=f"linter rejected all {max_retries} attempts: {last_violations}",
    )


def _render_essay(year: int, body: str, drift: DriftSummary) -> str:
    return (
        f"---\n"
        f"year: {year}\n"
        f"composed_at: {datetime.now(timezone.utc).isoformat()}\n"
        f"identity_events_in_window: {drift.n_events}\n"
        f"---\n\n"
        f"# Annual value reflection — {year}\n\n"
        f"{body.lstrip()}\n"
    )
