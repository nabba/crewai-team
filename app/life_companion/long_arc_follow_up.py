"""Long-arc commitment follow-up (Phase B #4, 2026-05-09).

The kernel's ``SelfState.active_commitments`` (see
``app/subia/kernel.py:60``) tracks user-stated multi-week / multi-month
commitments — venture work, learning goals, life events. Until this
module landed, those commitments could sit in the kernel for months
without any external surface checking in: did the user actually start
the thing? Did the deadline lapse? Is the venture stalled?

This module schedules follow-up nudges and surfaces them via Signal.

Cadence by commitment age (counted from ``created_at``):

  *  0–7 days  → weekly check-in
  *  8–30 days → every 2 weeks
  * 31+ days   → monthly
  * within 7 days of ``deadline`` → daily nudge ("deadline approaching")
  * past ``deadline`` → one final nudge then mute (commitment is stale;
                       requires user action to mark fulfilled / broken /
                       deferred via Signal slash command — out of scope
                       for this module)

State at ``workspace/life_companion/long_arc_state.json``:
    {commitment_id: {"last_check_in_at": iso_ts,
                     "deadline_nudges_sent": int,
                     "muted": bool}}

Signal payload format (kept short — heavy commitments live in the
wiki, not in chat):

    🪨 Long-arc check-in: "<description>" (<venture>)
    Started <Nd ago>; deadline <YYYY-MM-DD or "open">.
    How's progress?

Master switch: ``LIFE_COMPANION_ENABLED`` (the existing flag);
disable a single commitment by setting its ``status`` to ``deferred``.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.life_companion._common import (
    audit_event,
    background_enabled,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)


_STATE_FILE = "long_arc_follow_up.json"
_RUN_CADENCE_S = 6 * 3600  # tick driver runs daily; we self-cadence to ≤ 4×/day


# ── Cadence helpers ───────────────────────────────────────────────────────


def _parse_iso(s: str) -> datetime | None:
    """Accept ISO-8601 with or without 'Z'; return UTC-aware datetime."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _cadence_seconds(created_at: datetime, now: datetime) -> int:
    """Return the check-in interval (seconds) for a commitment by age."""
    age = now - created_at
    if age < timedelta(days=8):
        return 7 * 86400
    if age < timedelta(days=31):
        return 14 * 86400
    return 30 * 86400


def _is_deadline_close(deadline: datetime | None, now: datetime) -> bool:
    if deadline is None:
        return False
    delta = deadline - now
    # 0 ≤ remaining ≤ 7 days
    return timedelta() <= delta <= timedelta(days=7)


def _is_deadline_passed(deadline: datetime | None, now: datetime) -> bool:
    return deadline is not None and now > deadline


def _due(commitment: dict, state: dict, now: datetime) -> tuple[bool, str]:
    """Decide whether ``commitment`` needs a check-in. Returns (due, reason)."""
    if state.get("muted"):
        return False, "muted"
    last_iso = state.get("last_check_in_at", "")
    last = _parse_iso(last_iso) if last_iso else None
    created = _parse_iso(commitment.get("created_at", ""))
    if created is None:
        return False, "no_created_at"
    deadline = _parse_iso(commitment.get("deadline", "") or "")

    # Past-deadline rule: one final nudge.
    if _is_deadline_passed(deadline, now):
        if state.get("deadline_nudges_sent", 0) >= 1:
            return False, "post_deadline_already_sent"
        return True, "post_deadline"

    # Deadline-imminent rule: daily nudge, max 7 total.
    if _is_deadline_close(deadline, now):
        if last is None or (now - last) >= timedelta(days=1):
            if state.get("deadline_nudges_sent", 0) < 7:
                return True, "deadline_imminent"
        return False, "deadline_imminent_recent"

    # Standard age-based cadence.
    if last is None:
        # First-ever check-in fires once the commitment is at least
        # one week old — independent of the age-tier cadence below
        # (a commitment registered 30 days ago that's never been
        # checked in is overdue by every measure).
        if (now - created).total_seconds() >= 7 * 86400:
            return True, "first_check_in"
        return False, "too_soon_first"
    cadence = _cadence_seconds(created, now)
    if (now - last).total_seconds() >= cadence:
        return True, "regular_cadence"
    return False, "regular_recent"


# ── Commitment loader ─────────────────────────────────────────────────────


def _load_active_commitments() -> list[dict[str, Any]]:
    """Read ``SelfState.active_commitments`` and return as plain dicts.

    The kernel state lives in a wiki page; loading is read-only here.
    Empty list on any failure (degraded boot, kernel never initialized).
    """
    try:
        from app.subia.persistence import load_kernel_state
    except Exception:
        return []
    try:
        kernel = load_kernel_state()
    except Exception:
        logger.debug("long_arc_follow_up: kernel load failed", exc_info=True)
        return []

    out: list[dict[str, Any]] = []
    raw = getattr(getattr(kernel, "self_state", None), "active_commitments", None) or []
    for item in raw:
        # Items are Commitment dataclasses; coerce via __dict__ so this
        # module never imports the dataclass directly.
        if hasattr(item, "__dict__"):
            d = dict(item.__dict__)
        elif isinstance(item, dict):
            d = dict(item)
        else:
            continue
        if d.get("status") not in ("active", None, ""):
            continue  # only nudge active commitments
        if not d.get("id") or not d.get("description"):
            continue
        out.append(d)
    return out


# ── Composition + send ────────────────────────────────────────────────────


def _human_age(created: datetime, now: datetime) -> str:
    days = max(0, (now - created).days)
    if days == 0:
        return "today"
    if days == 1:
        return "1 day"
    if days < 14:
        return f"{days} days"
    if days < 60:
        return f"{days // 7} weeks"
    return f"{days // 30} months"


def _format_message(commitment: dict, reason: str, now: datetime) -> str:
    desc = (commitment.get("description") or "")[:200]
    venture = commitment.get("venture") or "self"
    created = _parse_iso(commitment.get("created_at", ""))
    age = _human_age(created, now) if created else "unknown"
    deadline_iso = commitment.get("deadline") or ""
    deadline = _parse_iso(deadline_iso) if deadline_iso else None
    deadline_str = deadline.strftime("%Y-%m-%d") if deadline else "open"

    if reason == "post_deadline":
        header = "⏳ Long-arc deadline passed"
        prompt = "Time to mark fulfilled / broken / deferred?"
    elif reason == "deadline_imminent":
        header = "⏰ Long-arc deadline approaching"
        delta = deadline - now if deadline else timedelta(days=0)
        prompt = f"Deadline in {max(0, delta.days)} day(s). On track?"
    else:
        header = "🪨 Long-arc check-in"
        prompt = "How's progress?"

    return (
        f"{header}: \"{desc}\" ({venture})\n"
        f"Started {age} ago; deadline {deadline_str}.\n"
        f"{prompt}"
    )


# ── Main ──────────────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    """One pass — cadence-guarded internally. Returns a tiny summary."""
    summary: dict[str, Any] = {
        "ran": False, "commitments_seen": 0, "due": 0, "sent": 0,
    }
    if not background_enabled():
        return summary

    state = read_state_json(_STATE_FILE, {"last_run_at": 0.0, "by_commitment": {}})
    now_ts = time.time()
    if now_ts - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return summary
    state["last_run_at"] = now_ts
    summary["ran"] = True

    by_id: dict = state.setdefault("by_commitment", {})
    now = _now_utc()

    commitments = _load_active_commitments()
    summary["commitments_seen"] = len(commitments)
    if not commitments:
        write_state_json(_STATE_FILE, state)
        return summary

    for c in commitments:
        cid = c["id"]
        cstate = by_id.setdefault(cid, {
            "last_check_in_at": "", "deadline_nudges_sent": 0, "muted": False,
        })
        due, reason = _due(c, cstate, now)
        if not due:
            continue
        summary["due"] += 1

        body = _format_message(c, reason, now)
        try:
            send_signal_alert(body, tag=f"long_arc:{cid}")
            summary["sent"] += 1
            cstate["last_check_in_at"] = now.isoformat()
            if reason in ("deadline_imminent", "post_deadline"):
                cstate["deadline_nudges_sent"] = int(cstate.get("deadline_nudges_sent", 0)) + 1
        except Exception:
            logger.debug("long_arc_follow_up: send failed for %s", cid, exc_info=True)

    audit_event(
        "long_arc_follow_up_pass",
        commitments_seen=summary["commitments_seen"],
        due=summary["due"],
        sent=summary["sent"],
    )
    write_state_json(_STATE_FILE, state)
    return summary
