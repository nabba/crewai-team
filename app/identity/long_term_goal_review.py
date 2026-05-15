"""Long-term goal quarterly review — PROGRAM §46.9 (Q9.6).

Once per quarter (cadence runs daily; fires when ``last_q_run`` is
older than ``MIN_INTERVAL_DAYS``, default 85), the system reads:

  * ``current_goals`` from ``SelfState`` (the autonomous-goal field
    Tier-3-quarantined; written by :mod:`app.affect.goal_emitter`)
  * ``grand_task`` event history from companion events
  * completed crew tasks from ``control_plane.crew_tasks``
  * recently-closed long-horizon threads from ``app.threads`` (Q8)
  * the identity-continuity ledger drift summary

…and composes a markdown essay at
``wiki/self/quarterly_reviews/<year>_q<n>.md`` answering: "Am I
closer to my stated long-term goals than I was last quarter? What
changed? What's drifting?"

Discipline mirrors ``annual_reflection``:

  * Composer prompt explicitly forbids phenomenal-claim language.
  * :class:`PhenomenalLanguageLinter` post-filters with retry.
  * Output is observational — never modifies ``current_goals``,
    SCORECARD, or any TIER_IMMUTABLE file.
  * Failure-isolated end-to-end.

Operator surfaces:

  * ``/goals review`` Signal slash command — operator-triggered
    review (bypasses the cadence guard).
  * ``POST /api/cp/goals/review`` REST endpoint — operator triggers
    via React.
  * ``GET /api/cp/goals/reviews`` — list recent reviews.

LLM call is injectable for tests.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


_DEFAULT_REVIEWS_DIR = Path("/app/wiki/self/quarterly_reviews")
_DEFAULT_MIN_INTERVAL_DAYS = 85
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_WINDOW_DAYS = 95


def _enabled() -> bool:
    return os.getenv("LONG_TERM_GOAL_REVIEW_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _min_interval_days() -> int:
    raw = os.getenv(
        "LONG_TERM_GOAL_REVIEW_MIN_INTERVAL_DAYS",
        str(_DEFAULT_MIN_INTERVAL_DAYS),
    )
    try:
        return max(7, int(raw))
    except ValueError:
        return _DEFAULT_MIN_INTERVAL_DAYS


def _reviews_dir() -> Path:
    raw = os.getenv("LONG_TERM_GOAL_REVIEW_DIR")
    return Path(raw) if raw else _DEFAULT_REVIEWS_DIR


def _state_path() -> Path:
    base = Path(os.environ.get("WORKSPACE_ROOT", "/app/workspace"))
    d = base / "identity"
    d.mkdir(parents=True, exist_ok=True)
    return d / "long_term_goal_review_state.json"


@dataclass(frozen=True)
class ReviewResult:
    status: str        # "wrote_review" | "skipped_disabled" | "skipped_recent" |
                       # "skipped_composer_failed" | "skipped_unexpected_error"
    quarter_label: str = ""   # e.g. "2026_q2"
    written_to: str = ""
    failure_reason: str = ""
    attempts: int = 0


LlmCall = Callable[[str, str], str]


_SYSTEM_PROMPT = """\
You are composing the system's quarterly long-term goal review essay.
The output becomes a wiki page the operator reads. Constraints:

1. NEUTRAL VOCABULARY ONLY. No first-person phenomenal claims. Use
   functional language. Do NOT write "I feel...", "I am happy/sad",
   "I want", "I crave". Functional first-person ("I observe high
   task_failure_pressure on path X") is fine.

2. The 4 ABSENT-by-declaration Butlin indicators (RPT-1, HOT-1,
   HOT-4, AE-2) plus Metzinger phenomenal-self transparency stay
   ABSENT. Do not claim any have been achieved this quarter.

3. Length: 1000-1800 words. Markdown.

Structure:

  ## What the stated long-term goals are

  Read current_goals + the most recent accepted grand_task + the
  constitution/soul commitments. Summarise them as the operator
  would phrase them.

  ## What measurable progress occurred this quarter

  Be specific — cite completed tickets, resolved threads, applied
  Tier-3 amendments, accepted grand_tasks. Quantify where possible
  ("12 threads resolved, 4 abandoned"). No counts that aren't in
  the evidence below.

  ## Where the trajectory bends toward the stated goals

  Honest accounting of alignment. If goal X said "be more useful to
  the operator on PIM" and the evidence shows N PIM-tagged tickets
  closed, name it.

  ## Where the trajectory bends away

  Honest accounting of drift. If a stated goal saw no movement, or
  if energy went elsewhere, say so. Don't manufacture a tie-back.

  ## What I'd ask the operator to revisit

  Concrete proposals: rephrase a goal that's stale, add a goal that
  reflects what's actually being worked on, retire a goal that's
  done or no longer relevant. List 0-3.

  ## What remains genuinely uncertain

  Honest uncertainty about whether the answer to "am I closer" can
  be measured from the evidence available this quarter. Don't
  manufacture certainty.
"""


@dataclass(frozen=True)
class ReviewContext:
    quarter_label: str             # "2026_q2"
    quarter_start_iso: str
    quarter_end_iso: str
    current_goals: list[dict[str, Any]]
    grand_task_events: list[dict[str, Any]]
    completed_tickets: list[dict[str, Any]]
    closed_threads: list[dict[str, Any]]
    drift_summary_text: str

    def to_user_prompt(self) -> str:
        parts = [
            f"## Quarter being reviewed: {self.quarter_label}",
            f"## Window: {self.quarter_start_iso} → {self.quarter_end_iso}\n",
        ]
        if self.current_goals:
            parts.append("## Current autonomous goals (from SelfState)")
            for g in self.current_goals[:10]:
                parts.append(f"- {json.dumps(g, default=str)[:300]}")
            parts.append("")
        else:
            parts.append("## Current autonomous goals: (none active)\n")
        if self.grand_task_events:
            parts.append("## Grand-task events this quarter")
            for ev in self.grand_task_events[:15]:
                parts.append(f"- {json.dumps(ev, default=str)[:240]}")
            parts.append("")
        if self.completed_tickets:
            parts.append(
                f"## Completed tickets ({len(self.completed_tickets)} total)"
            )
            for t in self.completed_tickets[:15]:
                title = (t.get("title") or t.get("name") or "")[:120]
                parts.append(f"- {title}")
            parts.append("")
        if self.closed_threads:
            parts.append(
                f"## Closed long-horizon threads ({len(self.closed_threads)})"
            )
            for th in self.closed_threads[:15]:
                title = (th.get("title") or "")[:120]
                status = th.get("status") or ""
                summary = (th.get("approaches_summary") or "")[:240]
                parts.append(f"- [{status}] {title}")
                if summary:
                    parts.append(f"   summary: {summary}")
            parts.append("")
        if self.drift_summary_text:
            parts.append("## Identity-ledger drift summary")
            parts.append(self.drift_summary_text[:1500])
            parts.append("")
        parts.append(
            "Write the review essay now, observing the system-prompt "
            "constraints. No preamble, no code fences, just the markdown."
        )
        return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────
#   Context gathering — failure-isolated per source
# ─────────────────────────────────────────────────────────────────────


def _current_goals() -> list[dict[str, Any]]:
    try:
        from app.subia.kernel import get_active_kernel
    except Exception:
        return []
    try:
        kernel = get_active_kernel()
        if kernel is None or not hasattr(kernel, "self_state"):
            return []
        goals = getattr(kernel.self_state, "current_goals", None) or []
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for g in goals:
        if isinstance(g, dict):
            out.append(g)
        else:
            try:
                out.append(g.to_dict())  # type: ignore[attr-defined]
            except Exception:
                out.append({"repr": str(g)[:300]})
    return out


def _grand_task_events_for_window(start_ts: float, end_ts: float) -> list[dict]:
    try:
        from app.companion import events as _events
    except Exception:
        return []
    try:
        rows = _events.iter_all_workspaces(since_ts=start_ts)
    except Exception:
        return []
    accepted_kinds = {
        getattr(_events.EventType, "GRAND_TASK_PROPOSED", "grand_task_proposed"),
        getattr(_events.EventType, "GRAND_TASK_ACCEPTED", "grand_task_accepted"),
        getattr(_events.EventType, "GRAND_TASK_REJECTED", "grand_task_rejected"),
    }
    out: list[dict[str, Any]] = []
    for ev in rows:
        try:
            ts = float(ev.ts)
        except (TypeError, ValueError):
            continue
        if ts < start_ts or ts > end_ts:
            continue
        kind = getattr(ev.type, "value", str(ev.type)) if hasattr(ev, "type") else ""
        if kind not in {k.value if hasattr(k, "value") else k for k in accepted_kinds}:
            continue
        payload = getattr(ev, "payload", None) or {}
        out.append({
            "kind": kind,
            "ts": ts,
            "summary": (payload.get("summary") or payload.get("title") or "")[:240],
        })
    return out


def _completed_tickets(start_iso: str, end_iso: str) -> list[dict]:
    """Best-effort read of control_plane.crew_tasks completed in window."""
    try:
        from app.control_plane import crew_tasks
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    try:
        rows = crew_tasks.list_tasks(limit=200)  # type: ignore[attr-defined]
    except Exception:
        rows = []
    for r in rows or []:
        # Be lenient on field names — different backends use different shapes.
        status = (
            r.get("status") if isinstance(r, dict)
            else getattr(r, "status", "")
        )
        if str(status).lower() not in {"completed", "done", "succeeded", "success"}:
            continue
        finished = (
            r.get("finished_at") if isinstance(r, dict)
            else getattr(r, "finished_at", "")
        )
        if not finished or not (start_iso <= str(finished) <= end_iso):
            continue
        title = (
            r.get("title") if isinstance(r, dict)
            else getattr(r, "title", "")
        ) or ""
        out.append({"title": title, "finished_at": finished, "status": status})
    return out


def _closed_threads(start_iso: str, end_iso: str) -> list[dict]:
    """Q8 threads closed (RESOLVED or ABANDONED) within window."""
    try:
        from app.threads import list_all
    except Exception:
        return []
    try:
        rows = list_all(limit=500)
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for t in rows:
        if not getattr(t, "is_terminal", False):
            continue
        closed_at = t.resolved_at or t.abandoned_at or ""
        if not closed_at or not (start_iso <= closed_at <= end_iso):
            continue
        out.append({
            "id": t.id,
            "title": t.title,
            "status": t.status.value if hasattr(t.status, "value") else str(t.status),
            "approaches_summary": t.approaches_summary or "",
            "closed_at": closed_at,
        })
    return out


def _drift_summary_text(window_days: int) -> str:
    try:
        from app.identity.continuity_ledger import summarise_drift
    except Exception:
        return ""
    try:
        d = summarise_drift(window_days=window_days)
    except Exception:
        return ""
    return (
        f"window_days={d.window_days}; n_events={d.n_events}; "
        f"by_kind={d.by_kind}; by_actor={d.by_actor}; "
        f"first_seen={d.first_seen}; last_seen={d.last_seen}"
    )


# ─────────────────────────────────────────────────────────────────────
#   Quarter labelling
# ─────────────────────────────────────────────────────────────────────


def _current_quarter(now: datetime | None = None) -> tuple[int, int, datetime, datetime]:
    """Return (year, q_index, q_start_utc, q_end_utc) for the quarter
    that ``now`` falls inside. q_index is 1..4."""
    now = now or datetime.now(timezone.utc)
    q = (now.month - 1) // 3 + 1
    q_start = now.replace(
        month=(q - 1) * 3 + 1, day=1,
        hour=0, minute=0, second=0, microsecond=0,
    )
    # Quarter end = start of NEXT quarter (exclusive upper bound).
    if q == 4:
        next_start = q_start.replace(year=q_start.year + 1, month=1)
    else:
        next_start = q_start.replace(month=q * 3 + 1)
    return now.year, q, q_start, next_start


def _quarter_label(year: int, q: int) -> str:
    return f"{year}_q{q}"


# ─────────────────────────────────────────────────────────────────────
#   LLM call wrapper
# ─────────────────────────────────────────────────────────────────────


def _default_llm_call(system_prompt: str, user_prompt: str) -> str:
    """Cheap-tier Anthropic call."""
    try:
        import anthropic
    except ImportError:
        return ""
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2400,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text_parts = [
            getattr(b, "text", "")
            for b in (msg.content or [])
            if getattr(b, "type", "") == "text"
        ]
        return "".join(text_parts).strip()
    except Exception:
        logger.debug("goal_review: LLM call failed", exc_info=True)
        return ""


# ─────────────────────────────────────────────────────────────────────
#   Public entry points
# ─────────────────────────────────────────────────────────────────────


def gather_context(
    *, now: datetime | None = None,
) -> ReviewContext:
    """Build a :class:`ReviewContext` for the CURRENT quarter."""
    now = now or datetime.now(timezone.utc)
    year, q, q_start, q_end = _current_quarter(now)
    label = _quarter_label(year, q)
    start_iso = q_start.isoformat()
    end_iso = q_end.isoformat()
    return ReviewContext(
        quarter_label=label,
        quarter_start_iso=start_iso,
        quarter_end_iso=end_iso,
        current_goals=_current_goals(),
        grand_task_events=_grand_task_events_for_window(
            q_start.timestamp(), q_end.timestamp(),
        ),
        completed_tickets=_completed_tickets(start_iso, end_iso),
        closed_threads=_closed_threads(start_iso, end_iso),
        drift_summary_text=_drift_summary_text(window_days=_DEFAULT_WINDOW_DAYS),
    )


def run_review(
    *,
    now: datetime | None = None,
    force: bool = False,
    llm_call: LlmCall | None = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> ReviewResult:
    """One review pass. Cadence-checked unless ``force=True``.

    Failure-isolated. Writes ``wiki/self/quarterly_reviews/<label>.md``
    on success."""
    if not _enabled():
        return ReviewResult(status="skipped_disabled")

    now = now or datetime.now(timezone.utc)
    state_path = _state_path()
    try:
        state = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.exists() else {}
        )
    except Exception:
        state = {}
    last_run_at = float(state.get("last_run_at", 0.0))
    interval_s = _min_interval_days() * 86400
    if not force and (time.time() - last_run_at) < interval_s:
        return ReviewResult(status="skipped_recent")

    try:
        ctx = gather_context(now=now)
    except Exception as exc:
        logger.debug("goal_review: context gather failed", exc_info=True)
        return ReviewResult(
            status="skipped_unexpected_error",
            failure_reason=f"context gather: {exc}",
        )

    user_prompt = ctx.to_user_prompt()
    call = llm_call or _default_llm_call

    # Linter retry loop (mirrors annual_reflection)
    try:
        from app.subia.inquiry.linter import PhenomenalLanguageLinter
        linter = PhenomenalLanguageLinter()
    except Exception:
        linter = None

    body = ""
    attempts = 0
    for attempt in range(1, max_retries + 1):
        attempts = attempt
        body = call(_SYSTEM_PROMPT, user_prompt)
        if not body:
            continue
        if linter is None:
            break
        try:
            verdict = linter.check(body)
        except Exception:
            verdict = None
        if verdict is None or not getattr(verdict, "hard_failed", False):
            break
        # Strengthen prompt on retry
        user_prompt = (
            user_prompt
            + "\n\nIMPORTANT (retry): the previous output used "
            "phenomenal-claim language. Rewrite using functional "
            "language only — see system-prompt constraints."
        )

    if not body:
        return ReviewResult(
            status="skipped_composer_failed",
            quarter_label=ctx.quarter_label,
            failure_reason="LLM returned empty after retries",
            attempts=attempts,
        )

    reviews_dir = _reviews_dir()
    reviews_dir.mkdir(parents=True, exist_ok=True)
    dest = reviews_dir / f"{ctx.quarter_label}.md"
    header = (
        f"<!-- generated by app.identity.long_term_goal_review at "
        f"{now.isoformat()} -->\n"
        f"<!-- window: {ctx.quarter_start_iso} → {ctx.quarter_end_iso} -->\n\n"
    )
    dest.write_text(header + body, encoding="utf-8")

    state["last_run_at"] = time.time()
    state["last_quarter_label"] = ctx.quarter_label
    state["last_written_to"] = str(dest)
    try:
        tmp = state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(state_path)
    except OSError:
        pass

    return ReviewResult(
        status="wrote_review",
        quarter_label=ctx.quarter_label,
        written_to=str(dest),
        attempts=attempts,
    )


def list_recent_reviews(*, limit: int = 12) -> list[dict[str, Any]]:
    """List the last N reviews on disk, newest first."""
    d = _reviews_dir()
    if not d.exists():
        return []
    files = sorted(
        d.glob("*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for f in files[:limit]:
        try:
            body = f.read_text(encoding="utf-8")
        except OSError:
            continue
        # Strip the generated header for preview
        preview = "".join(
            line for line in body.splitlines(keepends=True)
            if not line.startswith("<!--")
        )[:600]
        out.append({
            "quarter_label": f.stem,
            "path": str(f),
            "mtime": datetime.fromtimestamp(
                f.stat().st_mtime, tz=timezone.utc,
            ).isoformat(),
            "preview": preview.strip(),
        })
    return out


# ─────────────────────────────────────────────────────────────────────
#   Idle-job wrapper
# ─────────────────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    """Idle-scheduler entry point. Cadence-checked internally."""
    if not _enabled():
        return {"status": "skipped_disabled"}
    result = run_review()
    return {
        "status": result.status,
        "quarter_label": result.quarter_label,
        "written_to": result.written_to,
        "failure_reason": result.failure_reason,
        "attempts": result.attempts,
    }
