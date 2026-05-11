"""Weekly review surface for notification arbiter decisions.

PROGRAM §41 (2026-05-11) — Q4 Item 17 supporting monitor.

Walks the notify fatigue store on a weekly cadence and produces an
operator-facing review: how many notifications fired, queued,
suppressed; which topics dominated each bucket; whether the
suppression-rate ceiling kicked in.

Goodhart guard surface: this lets the operator notice if the arbiter
is silently filtering out things that should have surfaced. Without
this review, suppression decisions are invisible — exactly the
failure mode where bad arbitration logic compounds.

Cadence: weekly (7d). Cheap; scans a bounded JSONL.
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any

from app.life_companion._common import (
    audit_event,
    background_enabled,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)

_STATE_FILE = "notify_suppression_review.json"
_RUN_CADENCE_S = 7 * 24 * 3600   # weekly


def run() -> None:
    """One pass. Cadence-guarded internally."""
    if not background_enabled():
        return

    state = read_state_json(_STATE_FILE, {"last_run_at": 0.0})
    now = time.time()
    if now - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return
    state["last_run_at"] = now

    try:
        from app.notify.fatigue import list_recent, daily_suppression_rate
    except Exception:
        logger.debug(
            "notify_suppression_review: fatigue module unavailable",
            exc_info=True,
        )
        write_state_json(_STATE_FILE, state)
        return

    try:
        events = list_recent(window_hours=168.0) or []   # 1 week
    except Exception:
        events = []

    if not events:
        # First week of operation or arbiter not used yet. Skip alert.
        state["last_summary"] = {"events": 0}
        write_state_json(_STATE_FILE, state)
        return

    by_decision = Counter()
    topics_suppressed: Counter[str] = Counter()
    topics_sent: Counter[str] = Counter()
    for e in events:
        if not isinstance(e, dict):
            continue
        decision = (e.get("decision") or "").strip()
        by_decision[decision] += 1
        topic = (e.get("topic") or "").strip()
        if topic:
            if decision == "suppress_low_value":
                topics_suppressed[topic] += 1
            elif decision == "send_now":
                topics_sent[topic] += 1

    try:
        _suppressed, _total, daily_rate = daily_suppression_rate()
    except Exception:
        daily_rate = 0.0

    summary: dict[str, Any] = {
        "events": len(events),
        "by_decision": dict(by_decision),
        "top_suppressed_topics": topics_suppressed.most_common(5),
        "top_sent_topics": topics_sent.most_common(5),
        "today_suppression_rate": round(daily_rate, 3),
    }
    audit_event("notify_suppression_review_pass", **summary)

    # Send a digest Signal alert. Operator gets one weekly nudge to
    # review what's been filtered.
    body_parts = [
        "🔭 Weekly notification arbiter review:",
        f"  • Sent: {by_decision.get('send_now', 0)}",
        f"  • Queued: {by_decision.get('queue_for_digest', 0)}",
        f"  • Suppressed: {by_decision.get('suppress_low_value', 0)}",
    ]
    if topics_suppressed:
        body_parts.append("\n  Top suppressed topics:")
        for topic, count in topics_suppressed.most_common(3):
            body_parts.append(f"    – {topic} ({count}×)")
    body_parts.append(
        f"\n  Today's suppression rate: {daily_rate:.0%} "
        f"(ceiling 30%)."
    )
    body_parts.append(
        "\n  Review at /cp/companion → Insights, or query "
        "`GET /api/cp/notify/fatigue` to drill into the raw events."
    )
    body = "\n".join(body_parts)

    try:
        send_signal_alert(body, tag="notify_suppression_review")
    except Exception:
        logger.debug(
            "notify_suppression_review: alert failed", exc_info=True,
        )

    state["last_summary"] = summary
    write_state_json(_STATE_FILE, state)
