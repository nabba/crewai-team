"""Routine detector — surface day-of-week + time-of-day patterns.

Scans the affect-tagged episode log at
``workspace/affect/episode_affect_tags.jsonl`` (each line is one
completed task with ``ts`` + ``crew`` + ``agent_id`` +
``task_preview``). Clusters episodes by (day-of-week, hour-bucket,
crew) and flags any cluster that fires regularly enough to look like a
routine.

Detection is deterministic and cheap: count episodes per
(weekday, hour-bucket, crew) tuple over the last 4–8 weeks; any tuple
with ≥4 occurrences over that window AND ≥60 % concentration to the
weekday is flagged. The state file persists detected routines so
further passes only alert on new or changed ones.

Output:

  * Persisted to ``workspace/life_companion/routines.json``.
  * Signal alert when a NEW routine is first detected.
  * Reminder alert (cooldown ≥ 6 h) when a routine's window is
    approaching and no episode has fired yet today.

The detector runs nightly (cadence enforced inside ``run()``).
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.life_companion._common import (
    audit_event,
    background_enabled,
    feature_enabled,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)

_STATE_FILE = "routines.json"
_EPISODE_LOG = (
    Path(__file__).resolve().parents[2] / "workspace" / "affect"
    / "episode_affect_tags.jsonl"
)

# How far back to look. 8 weeks gives us at least 8 samples per weekly
# routine without dragging in too much stale behaviour.
_LOOKBACK_DAYS = 56

# A cluster needs at least this many episodes over the window to be a
# routine, AND at least this fraction of them on the same weekday.
_MIN_OCCURRENCES = 4
_MIN_WEEKDAY_FRACTION = 0.60

# Cadence: one detection pass per ~20 h (so it runs at most ~once/day
# even on a chatty idle scheduler).
_DETECT_CADENCE_S = 20 * 3600

# Reminder cadence: at most once per (routine_id, day).
_REMINDER_CADENCE_S = 6 * 3600

# Hour buckets — 4-hour windows.
_HOUR_BUCKETS = [
    (0, 4, "night"),
    (4, 8, "early-morning"),
    (8, 12, "morning"),
    (12, 16, "afternoon"),
    (16, 20, "evening"),
    (20, 24, "late-night"),
]

_WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _bucket_for(hour: int) -> str:
    for lo, hi, name in _HOUR_BUCKETS:
        if lo <= hour < hi:
            return name
    return "?"


def _bucket_window(name: str) -> tuple[int, int] | None:
    for lo, hi, n in _HOUR_BUCKETS:
        if n == name:
            return (lo, hi)
    return None


def _read_episodes() -> list[dict[str, Any]]:
    """Read affect-tagged episodes within the lookback window."""
    if not _EPISODE_LOG.exists():
        return []
    cutoff = datetime.now().astimezone() - timedelta(days=_LOOKBACK_DAYS)
    out: list[dict[str, Any]] = []
    try:
        with _EPISODE_LOG.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                ts = rec.get("ts") or ""
                if not ts:
                    continue
                try:
                    when = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    continue
                if when < cutoff:
                    continue
                out.append(rec)
    except Exception:
        logger.debug("routine_detector: read failed", exc_info=True)
    return out


def _cluster(episodes: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict]]:
    """Bucket by (weekday, hour-bucket, crew). Local time."""
    clusters: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for ep in episodes:
        ts = ep.get("ts") or ""
        try:
            when = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone()
        except Exception:
            continue
        wd = _WEEKDAY_NAMES[when.weekday()]
        hb = _bucket_for(when.hour)
        crew = (ep.get("crew") or ep.get("agent_id") or "unknown").lower()
        clusters[(wd, hb, crew)].append(ep)
    return clusters


def _detect(clusters: dict[tuple[str, str, str], list[dict]]) -> list[dict]:
    """Filter clusters down to confirmed routines.

    A routine is a (weekday, bucket, crew) cluster with ≥ N episodes
    where the same crew also appears on multiple weeks (so a one-off
    flurry on a single date doesn't graduate into a "routine").
    """
    routines: list[dict] = []
    for (wd, hb, crew), eps in clusters.items():
        if len(eps) < _MIN_OCCURRENCES:
            continue

        # Concentration check: of all this crew's episodes in any bucket,
        # what fraction landed on this weekday + bucket?
        same_crew_total = sum(
            len(v) for k, v in clusters.items() if k[2] == crew
        )
        if same_crew_total == 0:
            continue
        concentration = len(eps) / same_crew_total
        if concentration < _MIN_WEEKDAY_FRACTION:
            continue

        # Multi-week proof: episodes need to span ≥ 2 distinct ISO weeks.
        weeks = set()
        for ep in eps:
            try:
                w = datetime.fromisoformat(
                    (ep.get("ts") or "").replace("Z", "+00:00"),
                ).astimezone().isocalendar()
                weeks.add((w[0], w[1]))
            except Exception:
                continue
        if len(weeks) < 2:
            continue

        # Pick the most common task_preview prefix as a label.
        prevs = Counter(
            ((ep.get("task_preview") or "")[:60]).strip() for ep in eps
        )
        label = prevs.most_common(1)[0][0] if prevs else ""

        routines.append({
            "id": f"{wd}__{hb}__{crew}",
            "weekday": wd,
            "bucket": hb,
            "crew": crew,
            "occurrences": len(eps),
            "weeks_observed": len(weeks),
            "concentration": round(concentration, 2),
            "label": label,
        })
    return routines


def _diff_routines(
    prev: dict[str, dict], current: list[dict],
) -> tuple[list[dict], list[str]]:
    """Return ``(newly_detected, dropped_ids)``."""
    cur_by_id = {r["id"]: r for r in current}
    new = [r for r in current if r["id"] not in prev]
    dropped = [pid for pid in prev if pid not in cur_by_id]
    return new, dropped


def _format_new_routines(new: list[dict]) -> str:
    head = f"🔁 Detected {len(new)} new routine(s):\n"
    bullets = []
    for r in new:
        win = _bucket_window(r["bucket"])
        win_str = f"{win[0]:02d}:00–{win[1]:02d}:00" if win else r["bucket"]
        label = (r.get("label") or "").strip() or f"{r['crew']} work"
        bullets.append(
            f"  • {r['weekday']} {win_str}: {label} "
            f"(crew={r['crew']}, n={r['occurrences']}, "
            f"weeks={r['weeks_observed']}, "
            f"concentration={r['concentration']:.0%})"
        )
    return head + "\n".join(bullets)


def _approaching_routines(
    routines: list[dict], now: datetime, *, lookahead_min: int = 30,
) -> list[dict]:
    """Return routines whose window starts within ``lookahead_min`` minutes
    of ``now`` (same weekday). Used for the proactive nudge.
    """
    today = _WEEKDAY_NAMES[now.weekday()]
    out = []
    for r in routines:
        if r["weekday"] != today:
            continue
        win = _bucket_window(r["bucket"])
        if not win:
            continue
        bucket_start = now.replace(hour=win[0], minute=0, second=0, microsecond=0)
        delta_min = (bucket_start - now).total_seconds() / 60
        if 0 <= delta_min <= lookahead_min:
            out.append({**r, "minutes_until": int(delta_min)})
    return out


def run() -> None:
    """One pass — detection + nudge. Cadence-checked."""
    if not feature_enabled("routines"):
        return
    if not background_enabled():
        return

    now_real = time.time()
    state = read_state_json(_STATE_FILE, {
        "last_detect_at": 0.0,
        "routines": {},
        "last_reminder_at": {},
    })

    # ── Detection (slow path, cadenced) ───────────────────────────────
    if now_real - float(state.get("last_detect_at", 0)) >= _DETECT_CADENCE_S:
        episodes = _read_episodes()
        clusters = _cluster(episodes)
        current = _detect(clusters)

        prev = state.get("routines", {}) or {}
        new, dropped = _diff_routines(prev, current)

        # Persist current set indexed by id.
        state["routines"] = {r["id"]: r for r in current}
        state["last_detect_at"] = now_real

        audit_event(
            "routine_detector_pass",
            episodes=len(episodes),
            clusters=len(clusters),
            routines=len(current),
            new=len(new),
            dropped=len(dropped),
        )

        if new:
            send_signal_alert(_format_new_routines(new), tag="routine_detector")
            audit_event("routine_detector_new", count=len(new))

        write_state_json(_STATE_FILE, state)

    # ── Nudge (fast path, every tick) ─────────────────────────────────
    routines = list((state.get("routines") or {}).values())
    if not routines:
        return

    now_local = datetime.now()
    upcoming = _approaching_routines(routines, now_local)
    if not upcoming:
        return

    last_reminders = state.setdefault("last_reminder_at", {})
    today_key = now_local.strftime("%Y-%m-%d")
    fresh: list[dict] = []
    for r in upcoming:
        last = last_reminders.get(r["id"], "")
        if last == today_key:
            continue
        last_reminders[r["id"]] = today_key
        fresh.append(r)

    if not fresh:
        return

    body_lines = ["⏰ Upcoming routine(s) in the next 30 min:"]
    for r in fresh:
        win = _bucket_window(r["bucket"])
        win_str = f"{win[0]:02d}:00–{win[1]:02d}:00" if win else r["bucket"]
        label = (r.get("label") or "").strip() or f"{r['crew']} work"
        body_lines.append(
            f"  • {win_str}: {label} (crew={r['crew']}, "
            f"normally {r['occurrences']}× over {r['weeks_observed']} weeks)"
        )
    send_signal_alert("\n".join(body_lines), tag="routine_detector_nudge")
    audit_event("routine_detector_nudge", count=len(fresh))
    write_state_json(_STATE_FILE, state)
