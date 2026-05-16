"""Daily briefing — proactive Signal digest at fixed local times.

Three flavours, all opt-out per env var:

  * **morning** (default 07:00) — calendar (next 24 h) + top-3 urgent
    unread + open project tickets + companion ideas.
  * **evening** (default 18:00) — wrap of completed events / unhandled
    flagged mail / open ticket reminders.
  * **weekly** (default Mon 09:00) — last week's highlights + this
    week's calendar density + workspace activity.

Cadence guards inside ``run()`` ensure each flavour fires at most once
per scheduled window per local day. State at
``workspace/life_companion/daily_briefing.json``::

    {
      "last_morning_at": "2026-05-09",
      "last_evening_at": "2026-05-09",
      "last_weekly_at": "2026-W19",
    }

All data sources fail soft: a missing Calendar token, an empty inbox,
or a down ticket DB just gets a "(none)" line in the digest.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from app.life_companion._common import (
    audit_event,
    background_enabled,
    feature_enabled,
    read_state_json,
    send_signal_alert,
    user_email_address,
    write_state_json,
)

logger = logging.getLogger(__name__)

_STATE_FILE = "daily_briefing.json"

# Default local-clock windows. The cadence check in run() picks the
# flavour whose window the wall clock is in (with a ±15 min tolerance).
_TOLERANCE_MIN = 15


def _parse_hhmm(value: str, default: tuple[int, int]) -> tuple[int, int]:
    try:
        h, m = value.split(":")
        return (int(h), int(m))
    except Exception:
        return default


def _morning_time() -> tuple[int, int]:
    return _parse_hhmm(os.getenv("LIFE_COMPANION_BRIEFING_MORNING", "07:00"), (7, 0))


def _evening_time() -> tuple[int, int]:
    return _parse_hhmm(os.getenv("LIFE_COMPANION_BRIEFING_EVENING", "18:00"), (18, 0))


def _weekly_dow() -> int:
    """0=Mon ... 6=Sun"""
    table = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
    raw = os.getenv("LIFE_COMPANION_BRIEFING_WEEKLY_DOW", "MON").upper().strip()
    return table.get(raw, 0)


def _weekly_time() -> tuple[int, int]:
    return _parse_hhmm(os.getenv("LIFE_COMPANION_BRIEFING_WEEKLY_TIME", "09:00"), (9, 0))


def _now_local() -> datetime:
    """Local-clock datetime. The system already runs in the operator's
    timezone; ``datetime.now()`` (no tzinfo) is the right call.
    """
    return datetime.now()


def _within_window(now: datetime, target_h: int, target_m: int) -> bool:
    """Within ±_TOLERANCE_MIN of the target HH:MM."""
    target = now.replace(hour=target_h, minute=target_m, second=0, microsecond=0)
    delta = abs((now - target).total_seconds()) / 60
    return delta <= _TOLERANCE_MIN


def _which_flavour(now: datetime) -> str | None:
    """Decide which (if any) flavour should fire right now."""
    # Weekly takes priority when it's the configured day + window.
    if now.weekday() == _weekly_dow() and _within_window(now, *_weekly_time()):
        return "weekly"
    if _within_window(now, *_morning_time()):
        return "morning"
    if _within_window(now, *_evening_time()):
        return "evening"
    return None


# ── Data collectors (each fail-soft) ──────────────────────────────────────


def _gather_calendar_24h() -> list[str]:
    """Lines for the next 24 h of calendar events. Empty list on failure."""
    try:
        from app.tools.gcal_tools import _list_events
    except Exception:
        return []
    try:
        now = datetime.now(timezone.utc)
        events = _list_events(
            max_results=15,
            time_min=now.isoformat().replace("+00:00", "Z"),
            time_max=(now + timedelta(hours=24)).isoformat().replace("+00:00", "Z"),
        ) or []
    except Exception:
        return []

    lines = []
    for ev in events[:10]:
        start = ev.get("start") or ""
        summary = ev.get("summary") or "(untitled)"
        loc = ev.get("location") or ""
        loc_part = f" @ {loc[:30]}" if loc else ""
        # Strip date component when it's the same day as now (already implicit).
        try:
            t = start.split("T", 1)[1][:5] if "T" in start else start[:10]
        except Exception:
            t = start[:16]
        lines.append(f"  • {t} — {summary[:60]}{loc_part}")
    return lines


def _gather_top_emails(n: int = 3) -> list[str]:
    """Top-N urgent unread bullets. Empty on failure."""
    try:
        from app.tools.gmail_tools import _list_recent
        from app.tools.email_importance import EmailHeaders, score_email
    except Exception:
        return []
    try:
        stubs = _list_recent(limit=20, query="in:inbox is:unread") or []
    except Exception:
        return []
    if not stubs:
        return []

    user_addr = user_email_address()
    important_senders_raw = os.getenv("EMAIL_IMPORTANT_SENDERS", "")
    senders = tuple(
        p.strip().lower() for p in important_senders_raw.split(",") if p.strip()
    )

    scored = []
    for stub in stubs:
        h = EmailHeaders(
            from_=stub.get("from", ""),
            subject=stub.get("subject", ""),
            unread=True,
        )
        try:
            r = score_email(h, user_address=user_addr, important_senders=senders)
            scored.append((r.score, stub))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)
    lines = []
    for score, stub in scored[:n]:
        sender = (stub.get("from") or "(unknown)")[:50]
        subj = (stub.get("subject") or "(no subject)")[:60]
        lines.append(f"  • [{score:.1f}] {sender}: {subj}")
    return lines


def _gather_open_tickets(n: int = 5) -> list[str]:
    """Open tickets across the active venture. Soft fail."""
    try:
        from app.control_plane import db as cp_db
    except Exception:
        return []
    try:
        rows = cp_db.execute(
            """
            SELECT title, status, project_id
              FROM control_plane.tickets
             WHERE status NOT IN ('done', 'cancelled', 'archived')
             ORDER BY updated_at DESC
             LIMIT %s
            """,
            (n,),
            fetch=True,
        ) or []
    except Exception:
        return []

    lines = []
    for row in rows:
        title = (row.get("title") or "")[:60]
        status = row.get("status") or ""
        project = row.get("project_id") or ""
        lines.append(f"  • [{status}] {title}  ({project})")
    return lines


def _gather_top_interests(n: int = 5) -> list[str]:
    """Top-N topics from the interest_model profile (Phase F #6).

    Empty list when the profile hasn't been generated yet — not an
    error, just means interest_model hasn't run.
    """
    try:
        from app.companion.interest_model import current_profile
    except Exception:
        return []
    try:
        profile = current_profile()
    except Exception:
        return []
    topics = profile.get("topics") or []
    out: list[str] = []
    for t in topics[:n]:
        if not isinstance(t, dict):
            continue
        name = (t.get("name") or "").strip()
        score = t.get("score")
        if name and score is not None:
            out.append(f"  • {name} ({score:.2f})")
    return out


def _gather_health_summary() -> list[str]:
    """One-line health bullets for the daily briefing. Soft-fail.

    Returns ``[]`` when health ingestion is disabled or no records exist
    yet, so the briefing reads identical to the pre-§5.1 version. Only
    *summary statistics* reach this layer — never raw records — keeping
    the privacy invariant of the health subsystem intact.
    """
    try:
        from app.health.anomaly import detect_anomalies
        from app.health.summary import summarise_window
    except Exception:
        return []
    try:
        s = summarise_window(days=7)
    except Exception:
        return []
    if not s.record_counts or all(v == 0 for v in s.record_counts.values()):
        return []
    lines: list[str] = []
    if s.steps_per_day_mean > 0:
        lines.append(f"  • {s.steps_per_day_mean:,.0f} steps/day (7d avg)")
    if s.sleep_hours_per_night_mean is not None:
        lines.append(
            f"  • {s.sleep_hours_per_night_mean:.1f}h sleep/night "
            f"({s.sleep_nights_observed} nights observed)"
        )
    if s.hr_resting_p10_bpm is not None:
        lines.append(
            f"  • resting HR ~{s.hr_resting_p10_bpm:.0f} bpm (p10 proxy)"
        )
    if s.workouts_count > 0:
        lines.append(
            f"  • {s.workouts_count} workouts, "
            f"{s.workouts_distance_km_total:.1f} km total"
        )
    try:
        anomalies = detect_anomalies()
    except Exception:
        anomalies = []
    for a in anomalies:
        lines.append(f"  ⚠️ {a.description}")
    return lines


def _gather_codeable_papers(*, n: int = 3) -> list[str]:
    """Q10.2 (PROGRAM §46.14) — queued codeable paper-experiment
    ideas. Reads ``workspace/proposed_experiments.jsonl`` and picks
    the top-N most-recent rows with ``codeable=true``. Empty list
    when the ledger is missing or no codeable rows are present.

    Soft fail — never blocks the briefing.
    """
    from pathlib import Path as _P
    ledger = _P("/app/workspace/proposed_experiments.jsonl")
    if not ledger.exists():
        return []
    try:
        import json as _json
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict] = []
    # Walk newest-first (file is append-only).
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            r = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        if not isinstance(r, dict):
            continue
        if not r.get("codeable"):
            continue
        rows.append(r)
        if len(rows) >= n:
            break
    if not rows:
        return []
    out: list[str] = []
    for r in rows:
        title = (r.get("title") or "Untitled")[:80]
        rel = float(r.get("relevance") or 0.0)
        out.append(f"  📜 {title}  (rel {rel:.2f})")
        scaffold = r.get("scaffold") or {}
        purpose = (scaffold.get("driver_purpose") or "")[:140]
        if purpose:
            out.append(f"     → {purpose}")
    return out


def _gather_travel_block() -> list[str]:
    """Q9.3 (PROGRAM §46.6) — upcoming travel surfaced from TripIt
    + flight-status snapshots. Returns one block of markdown lines
    (header + per-segment) or ``[]`` when nothing in the window.
    Soft-fail."""
    try:
        from app.life_companion.travel import format_for_briefing
    except Exception:
        return []
    try:
        text = format_for_briefing(window_days=14)
    except Exception:
        return []
    if not text:
        return []
    return [text]


def _gather_people_insights(n: int = 5) -> list[str]:
    """Q4.2 (PROGRAM §42 L1) — people showing cross-modal convergence.
    Only emits when master switch ON. Soft fail."""
    try:
        from app.companion.person_model import current_profile
        prof = current_profile() or {}
    except Exception:
        return []
    if not prof.get("enabled"):
        return []
    people = prof.get("people") or []
    # Surface those with ≥3 modalities active in the recent window.
    convergent = [
        p for p in people
        if p.get("modality_count", 0) >= 3
        and p.get("total_occurrences", 0) >= 5
    ]
    if not convergent:
        return []
    lines: list[str] = []
    for p in convergent[:n]:
        display = (p.get("display_names") or [""])[0] or p.get("person_id", "?")
        mods = p.get("modality_count", 0)
        total = p.get("total_occurrences", 0)
        lines.append(f"  • {display[:40]} — {mods} modalities × {total} hits")
    return lines


def _gather_person_suggestions(n: int = 3) -> list[str]:
    """Q4.2 (PROGRAM §42 L3 + L4.4) — operator-facing nudges from the
    person-suggestions emitter. Shares the 3-per-briefing cap with
    L4.4 graph-suggestions (the rate limit lives in the emitter)."""
    try:
        from app.companion.person_suggestions import generate_suggestions
        sugs = generate_suggestions() or []
    except Exception:
        return []
    if not sugs:
        return []
    lines: list[str] = []
    for s in sugs[:n]:
        text = (s.get("text") or "")[:160]
        lines.append(f"  • {text}")
    return lines


def _gather_queued_notifications(n: int = 10) -> tuple[list[str], list[float]]:
    """Q4.1 (PROGRAM §41.4) — pull arbiter-queued notifications for
    the digest. Returns ``(formatted_lines, ts_set)`` where ts_set is
    the event timestamps the caller should mark as consumed after
    the briefing fires successfully.

    Newest-first; truncates to n entries. Soft fail."""
    try:
        from app.notify.fatigue import pending_digest_entries
    except Exception:
        return [], []
    try:
        events = pending_digest_entries(window_hours=24.0) or []
    except Exception:
        return [], []
    if not events:
        return [], []
    # Sort newest-first, take n.
    events.sort(key=lambda e: float(e.get("ts") or 0), reverse=True)
    events = events[:n]
    lines: list[str] = []
    ts_set: list[float] = []
    for e in events:
        title = (e.get("title") or "").strip()
        body = (e.get("body") or "").strip()
        topic = e.get("topic")
        # Compose a tight one-liner.
        if title and body:
            display = f"{title}: {body[:70]}"
        elif title:
            display = title
        elif body:
            display = body[:80]
        else:
            display = "(notification body empty)"
        if topic:
            display = f"[{topic}] {display}"
        lines.append(f"  • {display}")
        try:
            ts_set.append(float(e.get("ts") or 0))
        except (TypeError, ValueError):
            continue
    return lines, ts_set


def _mark_digest_consumed(ts_set: list[float]) -> None:
    """Best-effort cleanup after the briefing fires. Failure is non-
    fatal — worst case is the items resurface in the next digest,
    which is observably better than dropping them silently."""
    if not ts_set:
        return
    try:
        from app.notify.fatigue import mark_digest_consumed
        mark_digest_consumed(set(ts_set))
    except Exception:
        pass


def _gather_cross_modal_insights(n: int = 3) -> list[str]:
    """Q4#15 (PROGRAM §41) — proactive insights from the cross-modal
    pattern detector. Surfaces topics that crossed ≥3 modalities at
    high strength in the recent window. Soft fail."""
    try:
        from app.companion.cross_modal_patterns import list_recent_patterns
    except Exception:
        return []
    try:
        patterns = list_recent_patterns(n=n, min_strength=0.7) or []
    except Exception:
        return []
    if not patterns:
        return []
    lines: list[str] = []
    for p in patterns[:n]:
        topic = (p.get("topic") or "")[:60]
        mods = p.get("modalities") or []
        total = p.get("occurrences_total") or 0
        window = p.get("window_days") or 0
        lines.append(
            f"  • {topic} — {len(mods)} modalities × {total} hits / {window}d "
            f"({', '.join(mods[:4])})"
        )
    return lines


def _gather_open_tensions(n: int = 5) -> list[str]:
    """Q4#16 (PROGRAM §41) — open questions the operator left with the
    companion. Sorted newest-touched first; filters to OPEN status +
    freshness ≥ 0.3 (drops decayed-stale OPENs). Soft fail."""
    try:
        from app.companion.tensions import list_tensions, STATUS_OPEN
    except Exception:
        return []
    try:
        tensions = list_tensions(status=STATUS_OPEN, min_freshness=0.3) or []
    except Exception:
        return []
    if not tensions:
        return []
    lines: list[str] = []
    for t in tensions[:n]:
        q = (t.question or "")[:90]
        # Source-count hint helps operator gauge whether material accumulated.
        n_sources = len(t.sources or [])
        if n_sources > 0:
            lines.append(f"  • {q}  ({n_sources} note{'s' if n_sources != 1 else ''})")
        else:
            lines.append(f"  • {q}")
    return lines


def _gather_sentience_digest() -> list[str]:
    """Q5.4.2 — weekly digest of sentience-experiment observations.

    Surfaces the top-1 finding from each of the four modules when
    data exists. Empty list when nothing happened this week → the
    section disappears entirely from the briefing (the discipline
    that keeps every other section clean).

    Each line is OPAQUE (counts, never identities) — same discipline
    the modules themselves enforce in their GW publishes."""
    lines: list[str] = []
    # AE-2 — top high-density associations
    try:
        from app.sentience_experiments.ae2_causal_credit import list_recent
        assocs = list_recent(n=5) or []
        if assocs:
            top = assocs[0]
            n_strong = sum(
                1 for a in assocs
                if float(a.get("outcome_density_ratio", 0)) >= 5.0
            )
            if n_strong:
                lines.append(
                    f"  • AE-2: {n_strong} rare-event causal "
                    f"association{'s' if n_strong != 1 else ''} "
                    f"(top density ratio "
                    f"{float(top.get('outcome_density_ratio', 0)):.1f}×)"
                )
    except Exception:
        pass
    # HOT-1 — trace-level patterns (baseline_drift, attractor_lock)
    try:
        from app.sentience_experiments.hot1_meta_affect import list_recent
        patterns = list_recent(n=10) or []
        trace_level = [
            p for p in patterns
            if p.get("pattern_kind") in ("baseline_drift", "attractor_lock")
        ]
        if trace_level:
            top = trace_level[0]
            lines.append(
                f"  • HOT-1: {top.get('pattern_kind')} pattern "
                f"({top.get('n_occurrences')} obs over "
                f"{float(top.get('span_days', 0)):.1f}d)"
            )
    except Exception:
        pass
    # HOT-4 — flagged reasoning-chain steps (this week only).
    # Q5.6 (PROGRAM §43.6) — without the since_iso filter, a quiet
    # HOT-4 history would return N flagged rows from MONTHS ago and
    # the "this week" prose would be misleading. The other three
    # module digests are naturally time-bounded by their data
    # semantics; HOT-4 was the only one that needed explicit
    # windowing.
    try:
        from app.sentience_experiments.hot4_metacog_monitor import list_recent_flagged
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        week_ago = (_dt.now(_tz.utc) - _td(days=7)).isoformat()
        flagged = list_recent_flagged(n=20, since_iso=week_ago) or []
        if flagged:
            lines.append(
                f"  • HOT-4: {len(flagged)} unusual reasoning-chain "
                f"step{'s' if len(flagged) != 1 else ''} flagged this week"
            )
    except Exception:
        pass
    # RPT-1 — calibration state
    try:
        from app.sentience_experiments.rpt1_self_calibration import (
            load_calibration_state,
        )
        state = load_calibration_state() or {}
        reports = state.get("reports") or {}
        if reports:
            # Lowest Brier (best calibrated) wins the highlight.
            best_kind = min(
                reports.keys(),
                key=lambda k: float(reports[k].get("brier_score", 1.0)),
            )
            best = reports[best_kind]
            lines.append(
                f"  • RPT-1: {len(reports)} calibrated kind"
                f"{'s' if len(reports) != 1 else ''}; "
                f"best={best_kind!r} Brier="
                f"{float(best.get('brier_score', 0)):.3f}"
            )
    except Exception:
        pass
    return lines


def _gather_resilience_drill_digest() -> list[str]:
    """Q6.3 — weekly digest of resilience-drill status.

    Surfaces: count of drills past-due, count run successfully in the
    last 7 days, top failing drill name (if any). Section disappears
    entirely when nothing actionable.
    """
    try:
        import app.resilience_drills.drills  # noqa: F401 — populate registry
        from app.resilience_drills.audit import (
            days_since_last_success, iter_results,
        )
        from app.resilience_drills.protocol import get_registry, drill_enabled
    except Exception:
        return []
    registry = get_registry()
    if not registry.list_specs():
        return []
    # Per-drill staleness.
    stale: list[tuple[str, float]] = []
    for spec in registry.list_specs():
        if not drill_enabled(spec):
            continue
        days = days_since_last_success(spec.name)
        threshold = spec.cadence_days + spec.grace_days
        if days is None or days > threshold:
            stale.append((spec.name, days if days is not None else float("inf")))
    # Successful runs this week.
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    week_ago_iso = (_dt.now(_tz.utc) - _td(days=7)).isoformat()
    ran_this_week = 0
    failed_this_week: list[str] = []
    for row in iter_results(since_iso=week_ago_iso):
        status = row.get("status") or ""
        if status == "pass":
            ran_this_week += 1
        elif status in ("fail", "error"):
            failed_this_week.append(row.get("drill_name", "?"))
    if ran_this_week == 0 and not stale and not failed_this_week:
        return []
    lines: list[str] = []
    if ran_this_week:
        lines.append(
            f"  • {ran_this_week} drill"
            f"{'s' if ran_this_week != 1 else ''} passed this week"
        )
    if failed_this_week:
        lines.append(
            f"  • {len(failed_this_week)} drill"
            f"{'s' if len(failed_this_week) != 1 else ''} FAILED: "
            f"{', '.join(sorted(set(failed_this_week))[:3])}"
        )
    if stale:
        names = ", ".join(
            f"{n} ({d:.0f}d)" if d != float("inf") else f"{n} (never)"
            for n, d in sorted(stale)[:3]
        )
        lines.append(
            f"  • {len(stale)} drill"
            f"{'s' if len(stale) != 1 else ''} past-due: {names}"
        )
    return lines


def _gather_companion_surfaced() -> list[str]:
    """Recent companion ideas surfaced to the user (last 24 h). Soft fail."""
    try:
        from app.companion import idea_store as _idea_store
    except Exception:
        return []
    try:
        # The idea_store API varies by version; fall back gracefully if a
        # ``recent_surfaced`` accessor isn't available.
        if hasattr(_idea_store, "recent_surfaced"):
            ideas = _idea_store.recent_surfaced(hours=24) or []
        elif hasattr(_idea_store, "list_recent"):
            ideas = _idea_store.list_recent(hours=24) or []
        else:
            return []
    except Exception:
        return []

    lines = []
    for idea in ideas[:5]:
        if isinstance(idea, dict):
            txt = (idea.get("text") or idea.get("title") or "")[:80]
            ws = idea.get("workspace_id") or ""
            lines.append(f"  • {txt}  ({ws})")
    return lines


# ── Compose ──────────────────────────────────────────────────────────────


def _compose_morning() -> tuple[str, list[float]]:
    """Compose the morning briefing.

    Q4.1 (PROGRAM §41.4) returns a 2-tuple: ``(body, consume_ts_list)``
    where ``consume_ts_list`` is the timestamps of arbiter-queued
    notifications that should be marked consumed AFTER successful
    send. The caller (``run()``) only marks them on Signal-send
    success so failed briefings preserve the queue.
    """
    cal = _gather_calendar_24h()
    mail = _gather_top_emails(n=3)
    tickets = _gather_open_tickets(n=5)
    health = _gather_health_summary()
    tensions = _gather_open_tensions(n=5)
    queued_lines, queued_ts = _gather_queued_notifications(n=10)

    parts = ["☀️  Morning briefing\n"]
    parts.append("📅 Today's events:")
    parts.extend(cal or ["  • (none scheduled)"])
    parts.append("\n📬 Urgent unread:")
    parts.extend(mail or ["  • (inbox clean)"])
    parts.append("\n🎯 Open tickets:")
    parts.extend(tickets or ["  • (no open tickets)"])
    insights = _gather_cross_modal_insights(n=3)
    if insights:
        # Q4#15 — proactive insights from cross-modal pattern detector.
        parts.append("\n💡 Proactive insights:")
        parts.extend(insights)
    if tensions:
        # Q4#16 — open questions you left with me. Only show when
        # there's something to surface; the section disappears when
        # the list is empty so the briefing stays clean.
        parts.append("\n❓ Open questions you left with me:")
        parts.extend(tensions)
    people_insights = _gather_people_insights(n=5)
    if people_insights:
        # Q4.2 L1 — people showing cross-modal convergence
        parts.append("\n🧑 People showing up:")
        parts.extend(people_insights)
    person_suggestions = _gather_person_suggestions(n=3)
    if person_suggestions:
        # Q4.2 L3 + L4.4 — operator-facing nudges (combined cap of 3)
        parts.append("\n💬 Suggestions:")
        parts.extend(person_suggestions)
    if queued_lines:
        # Q4#17 — notifications the arbiter deferred. Pull them out
        # of the fatigue queue into the digest so "queue_for_digest"
        # actually surfaces somewhere.
        parts.append("\n📨 Queued notifications (deferred by arbiter):")
        parts.extend(queued_lines)
    if health:
        parts.append("\n❤️  Health (7d):")
        parts.extend(health)
    travel = _gather_travel_block()
    if travel:
        # Q9.3 — upcoming TripIt segments + flight status. The block
        # already carries its own header; just join below.
        parts.append("")
        parts.extend(travel)
    codeable = _gather_codeable_papers(n=3)
    if codeable:
        # Q10.2 (PROGRAM §46.14) — paper-experiment queued ideas.
        # Only surfaces papers the LLM marked codeable; non-codeable
        # papers stay in the regular weekly digest.
        parts.append("\n💻 Queued codeable paper ideas:")
        parts.extend(codeable)
    return "\n".join(parts), queued_ts


def _compose_evening() -> tuple[str, list[float]]:
    cal = _gather_calendar_24h()  # also covers tonight + tomorrow morning
    mail = _gather_top_emails(n=3)
    surfaced = _gather_companion_surfaced()
    health = _gather_health_summary()
    queued_lines, queued_ts = _gather_queued_notifications(n=10)

    parts = ["🌙 Evening wrap\n"]
    parts.append("📅 Tomorrow:")
    parts.extend(cal or ["  • (no events)"])
    parts.append("\n📬 Still flagged:")
    parts.extend(mail or ["  • (inbox clean)"])
    parts.append("\n💡 Companion surfaced today:")
    parts.extend(surfaced or ["  • (no surfaced ideas)"])
    if queued_lines:
        parts.append("\n📨 Queued notifications (deferred by arbiter):")
        parts.extend(queued_lines)
    if health:
        parts.append("\n❤️  Health (7d):")
        parts.extend(health)
    travel = _gather_travel_block()
    if travel:
        parts.append("")
        parts.extend(travel)
    return "\n".join(parts), queued_ts


def _compose_weekly() -> tuple[str, list[float]]:
    cal = _gather_calendar_24h()
    tickets = _gather_open_tickets(n=8)
    surfaced = _gather_companion_surfaced()
    interests = _gather_top_interests(n=5)
    health = _gather_health_summary()

    parts = ["🗓 Weekly review\n"]
    parts.append("📅 Next 24h:")
    parts.extend(cal or ["  • (no events)"])
    parts.append("\n🎯 Open tickets:")
    parts.extend(tickets or ["  • (no open tickets)"])
    parts.append("\n💡 Companion surfaced last week:")
    parts.extend(surfaced or ["  • (none)"])
    if health:
        parts.append("\n❤️  Health (7d):")
        parts.extend(health)
    if interests:
        # Phase F #6: only surface interests in the weekly digest —
        # daily morning/evening cadences don't need this noise.
        parts.append("\n🧭 Topics you've cared about:")
        parts.extend(interests)
    # Q5.4.2 — sentience digest. Weekly cadence is the right one:
    # the four modules accumulate data slowly, and daily would be
    # too noisy. Section disappears when nothing happened this week.
    sentience = _gather_sentience_digest()
    if sentience:
        parts.append("\n🔬 Self-observation (week):")
        parts.extend(sentience)
    # Q6.3 — resilience drills digest. Section disappears when
    # nothing actionable (no past-due, no recent failures, no recent
    # successes — i.e. truly nothing happened).
    drills = _gather_resilience_drill_digest()
    if drills:
        parts.append("\n🛡 Resilience drills (week):")
        parts.extend(drills)
    # Weekly composer doesn't pull from the queue — daily covers that.
    return "\n".join(parts), []


# ── Cadence-aware entry point ─────────────────────────────────────────────


def _key_for(flavour: str, now: datetime) -> str:
    """Idempotency key. Daily for morning/evening; weekly for weekly."""
    if flavour == "weekly":
        # ISO week token like "2026-W19" — same week shares the key.
        iso = now.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    return now.strftime("%Y-%m-%d")


def _last_key(state: dict, flavour: str) -> str:
    return state.get(f"last_{flavour}_at", "") or ""


def _set_last_key(state: dict, flavour: str, key: str) -> None:
    state[f"last_{flavour}_at"] = key


_COMPOSERS = {
    "morning": _compose_morning,
    "evening": _compose_evening,
    "weekly": _compose_weekly,
}


def run() -> None:
    """Cadence-aware tick. Idempotent within each scheduled window."""
    if not feature_enabled("briefing"):
        return
    if not background_enabled():
        return

    now = _now_local()
    flavour = _which_flavour(now)
    if flavour is None:
        return

    state = read_state_json(_STATE_FILE, {})
    key = _key_for(flavour, now)
    if _last_key(state, flavour) == key:
        return  # already sent this window

    composer = _COMPOSERS.get(flavour)
    if composer is None:
        return

    try:
        body, consume_ts = composer()
    except Exception:
        logger.debug("daily_briefing: composer %s raised", flavour, exc_info=True)
        return

    audit_event(
        "daily_briefing_send",
        flavour=flavour,
        key=key,
        body_chars=len(body),
        queued_notifications_included=len(consume_ts),
    )
    sent = send_signal_alert(body, tag=f"daily_briefing_{flavour}")
    if sent:
        _set_last_key(state, flavour, key)
        write_state_json(_STATE_FILE, state)
        # Q4.1 — mark queued notifications consumed ONLY after the
        # briefing actually fires. A failed Signal send preserves the
        # queue for the next cadence window.
        if consume_ts:
            _mark_digest_consumed(consume_ts)
