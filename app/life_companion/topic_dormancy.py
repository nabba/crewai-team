"""Topic-dormancy long-arc nudge (Phase G #3, 2026-05-09).

Companion to ``long_arc_follow_up.py`` (Phase B #4). The follow-up
module nudges on EXPLICIT commitments (``SelfState.active_commitments``).
This module nudges on IMPLICIT topics: things the operator was deeply
into months ago but hasn't touched in weeks — "you were heavy on
forest carbon for 3 months → silent for the last 8 weeks. Still
blocked, or moved on?"

How:

  1. ``interest_model`` now retains a per-pass timeseries at
     ``workspace/companion/interest_history.jsonl`` — append-only
     ``{ts, name, score}`` rows, capped via ``app.utils.jsonl_retention``.
  2. This module reads the history (last 365 d), groups by topic name,
     and computes:
        * ``peak_score_old`` — max score in the [60, 365] day window.
        * ``avg_score_recent`` — mean score in the last 14 d.
     A topic is **dormant** when ``peak_score_old > 1.0`` AND
     ``avg_score_recent < 0.3`` (a clear before/after drop).
  3. Per-topic dedup: 30 days. Once nudged, won't re-nudge for
     a month — long enough that the operator can either re-engage
     or explicitly dismiss via ``/topic mute <name>``.
  4. Output: one Signal message per pass listing up to 3 dormant
     topics. Quiet on no-dormancy.

State at ``workspace/life_companion/topic_dormancy.json``::

    {"alerted": {"<topic>": {"last_alert_at": <ts>, "muted": <bool>}}}

The ``muted`` flag is set by ``/topic mute <name>`` (see
``commander/commands.py``). Muted topics never re-alert.

Cadence: daily.
Master switch: ``TOPIC_DORMANCY_ENABLED`` (default ON).
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
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


_STATE_FILE = "topic_dormancy.json"
_HISTORY_PATH = Path("/app/workspace/companion/interest_history.jsonl")
_RUN_CADENCE_S = 24 * 3600

_PEAK_LOOKBACK_DAYS = 365
_PEAK_OLDEST_DAYS = 60        # peak measured in [60, 365] day window
_RECENT_DAYS = 14
_PEAK_THRESHOLD = 1.0
_RECENT_THRESHOLD = 0.3
_MIN_PEAK_SAMPLES = 4         # require ≥N old observations to trust peak
_DEDUP_WINDOW_S = 30 * 86400
_TOP_N_ALERT = 3


def _enabled() -> bool:
    return os.getenv("TOPIC_DORMANCY_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


# ── History reader ───────────────────────────────────────────────────────


def _read_history(now_ts: float) -> dict[str, list[tuple[float, float]]]:
    """Group history rows by topic. Returns ``{name: [(ts, score), ...]}``.

    Filters to the last ``_PEAK_LOOKBACK_DAYS`` so the in-memory list
    stays bounded.
    """
    if not _HISTORY_PATH.exists():
        return {}
    cutoff = now_ts - _PEAK_LOOKBACK_DAYS * 86400
    out: dict[str, list[tuple[float, float]]] = {}
    try:
        with _HISTORY_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                ts_iso = row.get("ts", "")
                try:
                    dt = datetime.fromisoformat(str(ts_iso).replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue
                ts = dt.timestamp()
                if ts < cutoff:
                    continue
                name = (row.get("name") or "").strip().lower()
                score = float(row.get("score") or 0.0)
                if not name:
                    continue
                out.setdefault(name, []).append((ts, score))
    except OSError:
        logger.debug("topic_dormancy: history read failed", exc_info=True)
    return out


# ── Detection ────────────────────────────────────────────────────────────


def _is_dormant(
    points: list[tuple[float, float]], now_ts: float,
) -> tuple[bool, dict]:
    """Decide whether ``points`` shows a dormancy pattern. Returns
    (is_dormant, evidence_dict). ``evidence_dict`` is meant for the
    Signal message body."""
    if not points:
        return False, {}
    recent_cutoff = now_ts - _RECENT_DAYS * 86400
    peak_cutoff = now_ts - _PEAK_OLDEST_DAYS * 86400

    old_scores = [s for ts, s in points if ts <= peak_cutoff]
    recent_scores = [s for ts, s in points if ts >= recent_cutoff]

    if len(old_scores) < _MIN_PEAK_SAMPLES:
        return False, {}
    peak_old = max(old_scores)
    avg_recent = (
        sum(recent_scores) / len(recent_scores)
        if recent_scores else 0.0
    )
    if peak_old < _PEAK_THRESHOLD:
        return False, {}
    if avg_recent >= _RECENT_THRESHOLD:
        return False, {}

    # Find when peak was last seen (most recent OLD observation that
    # equalled or exceeded peak * 0.9). That's the "deep on this"
    # window we surface to the operator.
    near_peak = [(ts, s) for ts, s in points if s >= peak_old * 0.9]
    near_peak.sort(key=lambda kv: kv[0])
    last_peak_ts = near_peak[-1][0] if near_peak else points[-1][0]
    days_since_peak = max(0, int((now_ts - last_peak_ts) / 86400))
    return True, {
        "peak_score": round(peak_old, 2),
        "recent_avg_score": round(avg_recent, 2),
        "days_since_peak": days_since_peak,
        "n_old_samples": len(old_scores),
        "n_recent_samples": len(recent_scores),
    }


# ── Composition ──────────────────────────────────────────────────────────


def _format_alert(records: list[tuple[str, dict]]) -> str:
    lines = [
        f"🪨 Topic dormancy: {len(records)} topic(s) you were deep on "
        f"have gone quiet:\n",
    ]
    for name, ev in records[:_TOP_N_ALERT]:
        days = ev["days_since_peak"]
        lines.append(
            f"  • \"{name}\" — peak {ev['peak_score']} "
            f"~{days} days ago, "
            f"recent avg {ev['recent_avg_score']}"
        )
    if len(records) > _TOP_N_ALERT:
        lines.append(f"  …and {len(records) - _TOP_N_ALERT} more")
    lines.append("")
    lines.append(
        "Still working on these? Or shelved? Reply, or "
        "`/topic mute <name>` to silence further nudges."
    )
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ran": False, "topics_examined": 0,
        "dormant": 0, "alerted": [], "sent": False,
    }
    if not _enabled() or not background_enabled():
        return summary

    state = read_state_json(_STATE_FILE, {
        "last_run_at": 0.0, "alerted": {},
    })
    now_ts = time.time()
    if now_ts - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return summary
    state["last_run_at"] = now_ts
    summary["ran"] = True

    history = _read_history(now_ts)
    summary["topics_examined"] = len(history)
    if not history:
        write_state_json(_STATE_FILE, state)
        return summary

    alerted_state: dict = state.setdefault("alerted", {})
    candidates: list[tuple[str, dict]] = []
    for name, points in history.items():
        prev = alerted_state.get(name) or {}
        if prev.get("muted"):
            continue
        last_ts = float(prev.get("last_alert_at", 0))
        if now_ts - last_ts < _DEDUP_WINDOW_S:
            continue
        is_dormant, ev = _is_dormant(points, now_ts)
        if is_dormant:
            candidates.append((name, ev))

    summary["dormant"] = len(candidates)

    if candidates:
        # Sort by days_since_peak ASC (most-recently-deep first — those
        # are the ones the operator might still re-engage on).
        candidates.sort(key=lambda kv: kv[1]["days_since_peak"])

        body = _format_alert(candidates)
        try:
            send_signal_alert(body, tag="topic_dormancy")
            summary["sent"] = True
            for name, _ev in candidates[:_TOP_N_ALERT]:
                alerted_state.setdefault(name, {})["last_alert_at"] = now_ts
                summary["alerted"].append(name)
        except Exception:
            logger.debug(
                "topic_dormancy: alert send failed", exc_info=True,
            )

    write_state_json(_STATE_FILE, state)
    audit_event(
        "topic_dormancy_pass",
        topics_examined=summary["topics_examined"],
        dormant=summary["dormant"],
        alerted=summary["alerted"],
        sent=summary["sent"],
    )
    return summary


# ── Operator API: mute / unmute (consumed by /topic slash command) ───────


def mute(topic: str) -> bool:
    """Suppress further dormancy alerts for ``topic``. Returns True."""
    state = read_state_json(_STATE_FILE, {"alerted": {}})
    norm = (topic or "").strip().lower()
    if not norm:
        return False
    state.setdefault("alerted", {}).setdefault(norm, {})["muted"] = True
    write_state_json(_STATE_FILE, state)
    return True


def unmute(topic: str) -> bool:
    """Resume dormancy alerts for ``topic``. Returns True if previously muted."""
    state = read_state_json(_STATE_FILE, {"alerted": {}})
    norm = (topic or "").strip().lower()
    if not norm:
        return False
    entry = state.setdefault("alerted", {}).setdefault(norm, {})
    was_muted = bool(entry.get("muted"))
    entry["muted"] = False
    write_state_json(_STATE_FILE, state)
    return was_muted
