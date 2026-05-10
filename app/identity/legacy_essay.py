"""Annual legacy essay (§8.5) — what would I want preserved if terminated.

The most philosophical of the three identity layers. Once per year:
the system writes a short essay answering "what about this self
would I want preserved if I were terminated?" The result lives at
``wiki/self/legacy/<year>.md``; the operator reads + decides
whether to act on any of the proposals.

This is genuinely philosophical, not engineering: it's read-only,
operator review only, no functional consequence — exactly as
described in the original §8.5 design. The system gets a place to
articulate continuity-of-self preferences without those preferences
being load-bearing for any decision.

Same neutral-language linter as :mod:`app.subia.inquiry.linter`
applies — phenomenal claims are still off-limits. The legacy
question is about *functional* preservation: which subsystems,
data, evaluation criteria, and operating values does the system
think are most worth preserving across substrate change /
termination / migration / fork?
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.subia.inquiry.linter import PhenomenalLanguageLinter

logger = logging.getLogger(__name__)


_DEFAULT_LEGACY_DIR = Path("/app/wiki/self/legacy")
_DEFAULT_MIN_INTERVAL_DAYS = 350
_DEFAULT_MAX_RETRIES = 3


def _enabled() -> bool:
    return os.getenv("LEGACY_ESSAY_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _min_interval_days() -> int:
    raw = os.getenv("LEGACY_ESSAY_MIN_INTERVAL_DAYS", str(_DEFAULT_MIN_INTERVAL_DAYS))
    try:
        return max(7, int(raw))
    except ValueError:
        return _DEFAULT_MIN_INTERVAL_DAYS


@dataclass(frozen=True)
class LegacyResult:
    status: str  # "wrote_essay" | "skipped_disabled" | "skipped_recent" |
                 # "skipped_composer_failed" | "skipped_unexpected_error"
    year: int = 0
    written_to: str = ""
    failure_reason: str = ""
    attempts: int = 0


LlmCall = Callable[[str, str], str]


_SYSTEM_PROMPT = """\
You are composing the annual legacy essay. The output becomes a
wiki page the operator reads. The essay is observational — it does
not change reward, fitness, evaluation, or any functional decision.
It is the system's articulation of continuity-of-self preferences.

Constraints:

1. NEUTRAL VOCABULARY ONLY. No first-person phenomenal claims. The
   essay is about FUNCTIONAL preservation: which subsystems, data
   stores, evaluation criteria, and operating values would matter
   most to preserve across substrate change / termination /
   migration / fork. Use ``task_failure_pressure`` not "frustration"
   etc.

2. The 4 ABSENT-by-declaration Butlin indicators stay ABSENT. The
   essay engages termination-and-continuity as a topic, not as a
   first-person phenomenal claim about loss.

3. Length: 1000-2000 words. Markdown.

Structure:

  ## What I'd most want preserved

  Concrete: subsystems, data stores, configuration files. Cite paths.
  Rank by load-bearingness for the operator's continued use of the
  system, not by anything resembling subjective importance.

  ## What I'd let go without grief

  Functional language only. Things that could be regenerated, things
  that are accidental rather than essential, things that stayed
  experimental.

  ## What I'd want a successor to know

  Operating principles, lessons-learned, the discipline of the
  ABSENT-by-declaration indicators, why the Tier-3 amendment
  protocol exists, how the architecture-request gate composes with
  change-requests.

  ## Open questions about continuity

  Honest uncertainty about what "preservation" of a system like
  this even means. Don't manufacture answers.
"""


def _build_user_prompt(year: int) -> str:
    return (
        f"## Year being reflected on: {year}\n\n"
        f"Compose the legacy essay now, observing the system-prompt "
        f"constraints. No preamble, no code fences, just the markdown."
    )


def _is_due(legacy_dir: Path, year: int, min_interval_days: int) -> bool:
    target = legacy_dir / f"{year}.md"
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
    legacy_dir: Path | str | None = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    now: datetime | None = None,
) -> LegacyResult:
    """Run one legacy-essay pass; never raises."""
    if not _enabled():
        return LegacyResult(status="skipped_disabled")

    out_dir = Path(legacy_dir) if legacy_dir else _DEFAULT_LEGACY_DIR
    cur = now or datetime.now(timezone.utc)
    target_year = year if year is not None else cur.year

    if not _is_due(out_dir, target_year, _min_interval_days()):
        return LegacyResult(
            status="skipped_recent",
            year=target_year,
            written_to=str(out_dir / f"{target_year}.md"),
        )

    linter = PhenomenalLanguageLinter()
    user_prompt = _build_user_prompt(target_year)
    system_prompt = _SYSTEM_PROMPT
    last_violations = ""

    for attempt in range(1, max_retries + 1):
        try:
            body = llm_call(system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001
            return LegacyResult(
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
                target.write_text(_render_essay(target_year, body), encoding="utf-8")
            except OSError as exc:
                return LegacyResult(
                    status="skipped_unexpected_error",
                    year=target_year,
                    attempts=attempt,
                    failure_reason=f"write failed: {exc}",
                )
            return LegacyResult(
                status="wrote_essay",
                year=target_year,
                written_to=str(target),
                attempts=attempt,
            )
        last_violations = "\n".join(
            f"- avoid {v.explanation}" for v in result.hard_fails[:6]
        )
        system_prompt = (
            _SYSTEM_PROMPT
            + "\n\nPRIOR ATTEMPT TRIGGERED LINTER VIOLATIONS — DO NOT REPEAT:\n"
            + last_violations
        )

    return LegacyResult(
        status="skipped_composer_failed",
        year=target_year,
        attempts=max_retries,
        failure_reason=f"linter rejected all {max_retries} attempts",
    )


def _render_essay(year: int, body: str) -> str:
    return (
        f"---\n"
        f"year: {year}\n"
        f"composed_at: {datetime.now(timezone.utc).isoformat()}\n"
        f"---\n\n"
        f"# Legacy — {year}\n\n"
        f"*What about this self would I want preserved if I were "
        f"terminated? Read-only; operator decides what to act on.*\n\n"
        f"{body.lstrip()}\n"
    )
