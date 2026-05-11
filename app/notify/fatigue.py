"""Notification fatigue tracking — recent send/ack history.

PROGRAM §41 (2026-05-11) — Q4 Item 17 supporting module.

Tracks a rolling window of recent notifications keyed by tag/topic so
the arbiter can compute "we've sent N notifications about X in the
last 4 hours, was the user responding to them?" without inventing
data from thin air.

Storage: ``workspace/notify/fatigue_state.json`` — single JSON file
with a bounded list of recent events. Cap at 500 entries so the file
stays small even at high notification rates.

Goodhart guard: ack-state is recorded BUT NEVER becomes the dominant
input to arbitration. The acknowledged-rate is an advisory signal,
not a learning target. The arbiter's primary inputs remain
interest-model salience + welfare envelope + critical bypass.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_MAX_EVENTS = 500
_STATE_FILE = Path("/app/workspace/notify/fatigue_state.json")
_lock = threading.Lock()


def _default_state_file() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "notify" / "fatigue_state.json"
    except Exception:
        return _STATE_FILE


@dataclass
class NotifyEvent:
    ts: float          # epoch seconds
    tag: str           # notification tag
    topic: str | None  # optional topic correlation (interest_model name)
    decision: str      # "send_now" | "queue_for_digest" | "suppress_low_value"
    salience_score: float | None = None
    ack_state: str | None = None    # "acked" | "unacked" | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── State I/O ────────────────────────────────────────────────────────────


def _load(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or _default_state_file()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return []
    except (OSError, json.JSONDecodeError):
        return []


def _save(events: list[dict[str, Any]], path: Path | None = None) -> None:
    p = path or _default_state_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(events, indent=2), encoding="utf-8")
    tmp.replace(p)


def record_event(
    *,
    tag: str,
    topic: str | None,
    decision: str,
    salience_score: float | None = None,
    path: Path | None = None,
) -> None:
    """Append a notification decision to the fatigue store. Caps the
    list at _MAX_EVENTS (drops oldest)."""
    event = NotifyEvent(
        ts=time.time(),
        tag=tag,
        topic=topic,
        decision=decision,
        salience_score=salience_score,
        ack_state=None,
    )
    with _lock:
        events = _load(path)
        events.append(event.to_dict())
        # Drop oldest if over cap.
        if len(events) > _MAX_EVENTS:
            events = events[-_MAX_EVENTS:]
        try:
            _save(events, path)
        except OSError:
            logger.debug("notify.fatigue: save failed", exc_info=True)


def mark_acked(tag: str, window_seconds: float = 3600.0, path: Path | None = None) -> int:
    """Mark recent same-tag events as acked. Operator reaction →
    feedback_router → here. Returns count marked.

    Window default 1h matches typical "I see this notification" reaction
    latency. Beyond that, the operator is reacting to something else
    that just happened to share the tag."""
    with _lock:
        events = _load(path)
        cutoff = time.time() - window_seconds
        marked = 0
        for e in reversed(events):
            if e.get("tag") != tag:
                continue
            if float(e.get("ts") or 0) < cutoff:
                break
            if e.get("ack_state") is None:
                e["ack_state"] = "acked"
                marked += 1
        if marked > 0:
            try:
                _save(events, path)
            except OSError:
                pass
        return marked


# ── Read API (used by arbiter) ───────────────────────────────────────────


def recent_count(window_hours: float = 4.0, path: Path | None = None) -> int:
    """How many notifications fired (sent_now decision) in the recent
    window? High count = throttle harder."""
    events = _load(path)
    cutoff = time.time() - window_hours * 3600.0
    return sum(
        1 for e in events
        if float(e.get("ts") or 0) >= cutoff
        and e.get("decision") == "send_now"
    )


def recent_count_by_topic(topic: str, window_hours: float = 24.0, path: Path | None = None) -> int:
    """How many send_now decisions for THIS topic in the window?
    Per-topic rate-limit input."""
    if not topic:
        return 0
    events = _load(path)
    cutoff = time.time() - window_hours * 3600.0
    topic_lower = topic.lower()
    return sum(
        1 for e in events
        if float(e.get("ts") or 0) >= cutoff
        and e.get("decision") == "send_now"
        and (e.get("topic") or "").lower() == topic_lower
    )


def daily_suppression_rate(path: Path | None = None) -> tuple[int, int, float]:
    """Returns ``(suppressed_today, total_decisions_today, rate)``.
    Used by the suppression-rate ceiling guard."""
    events = _load(path)
    cutoff = time.time() - 24 * 3600.0
    suppressed = 0
    total = 0
    for e in events:
        if float(e.get("ts") or 0) < cutoff:
            continue
        total += 1
        if e.get("decision") == "suppress_low_value":
            suppressed += 1
    rate = (suppressed / total) if total > 0 else 0.0
    return suppressed, total, rate


def list_recent(window_hours: float = 168.0, path: Path | None = None) -> list[dict[str, Any]]:
    """Read recent events for the operator-facing suppression review.
    Default window is 1 week."""
    events = _load(path)
    cutoff = time.time() - window_hours * 3600.0
    return [e for e in events if float(e.get("ts") or 0) >= cutoff]
