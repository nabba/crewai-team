"""Notification arbitration — "is this worth interrupting?"

PROGRAM §41 (2026-05-11) — Q4 Item 17.

The existing ``notify`` API is fire-and-forget. Every wrapped
function fires Signal + Web Push on completion regardless of context.
The arbiter is an OPT-IN pre-filter that decides one of three
outcomes for a pending notification:

  * **SEND_NOW**           — fire immediately (high salience or critical)
  * **QUEUE_FOR_DIGEST**   — defer to the next briefing cycle
  * **SUPPRESS_LOW_VALUE** — drop entirely (low salience + high fatigue)

Inputs (all read-only, failure-isolated):

  * interest_model salience for the topic
  * cross_modal_patterns strength (boost if matching)
  * open companion tensions matching topic (boost)
  * affect welfare envelope (critical breach → only critical alerts)
  * fatigue: recent send_now count global + per-topic
  * recent operator interaction recency (engaged user = lower fatigue)

Critical bypass: ``critical=True`` ALWAYS sends, no arbitration. Used
for security alerts, runbook fail-loud paths, welfare-breach
notifications. The arbiter must NEVER suppress a critical alert.

Goodhart guards:
  * Daily suppression rate cap at 30%. If we hit it, the NEXT
    notification force-sends regardless of score.
  * Ack-rate is recorded but is NOT a primary input. The arbiter
    optimises for safety guards, not "what the user reacted to."
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from typing import Any

from app.notify.fatigue import (
    record_event, recent_count, recent_count_by_topic,
    daily_suppression_rate,
)

logger = logging.getLogger(__name__)


DECISION_SEND_NOW = "send_now"
DECISION_QUEUE = "queue_for_digest"
DECISION_SUPPRESS = "suppress_low_value"


_MAX_DAILY_SUPPRESSION_RATE = 0.30
_TOPIC_BURST_THRESHOLD = 5          # ≥5 in 24h triggers per-topic throttle
_TOTAL_BURST_THRESHOLD = 8          # ≥8 in 4h triggers global throttle


@dataclass
class ArbitrationResult:
    decision: str                   # "send_now" / "queue_for_digest" / "suppress_low_value"
    reason: str                     # one-line operator-readable
    salience_score: float           # 0..1 — what the arbiter computed
    inputs: dict[str, Any]          # the signals that fed the decision

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Input collectors ─────────────────────────────────────────────────────


def _interest_score(topic: str) -> float:
    """0..1 score from the interest_model for this topic. Returns 0.0
    if the model is unavailable or the topic isn't tracked."""
    if not topic:
        return 0.0
    try:
        from app.companion.interest_model import current_profile
        prof = current_profile() or {}
        topics = prof.get("topics") or []
        topic_lower = topic.lower()
        for t in topics:
            if isinstance(t, dict) and (t.get("name") or "").lower() == topic_lower:
                # interest_model scores are typically [0, 5+]; normalize.
                raw = float(t.get("score") or 0.0)
                return max(0.0, min(1.0, raw / 5.0))
    except Exception:
        logger.debug("notify.arbiter: interest lookup failed", exc_info=True)
    return 0.0


def _pattern_boost(topic: str) -> float:
    """0..0.3 boost if there's a recent cross-modal pattern for this
    topic. Caps modestly so a single signal doesn't dominate."""
    if not topic:
        return 0.0
    try:
        from app.companion.cross_modal_patterns import list_recent_patterns
        patterns = list_recent_patterns(n=20, min_strength=0.7) or []
        topic_lower = topic.lower()
        for p in patterns:
            if (p.get("topic") or "").lower() == topic_lower:
                # Pattern strength is already 0..1; cap contribution at 0.3.
                return min(0.3, float(p.get("strength") or 0.0) * 0.3)
    except Exception:
        logger.debug("notify.arbiter: pattern lookup failed", exc_info=True)
    return 0.0


def _person_centrality_boost(topic: str) -> float:
    """Q4.2 (PROGRAM §42 L2) — if the topic references a person whose
    centrality is high, modest salience boost. Capped at 0.15 so it
    doesn't dominate. Off when L2 disabled."""
    if not topic:
        return 0.0
    try:
        from app.companion.person_centrality import centrality_for
        # If topic looks like an email, use it directly.
        if "@" in topic:
            return min(0.15, centrality_for(topic) * 0.15)
    except Exception:
        logger.debug("notify.arbiter: centrality lookup failed", exc_info=True)
    return 0.0


def _bridge_boost(topic: str) -> float:
    """Q4.2 (PROGRAM §42 L4.3) — if the topic mentions a person who
    is currently a bridge/cut-vertex, modest salience boost. Capped
    at 0.10. Off when L4.3 disabled."""
    if not topic:
        return 0.0
    try:
        from app.companion.graph_features.bridges import is_bridge_or_cut
        if "@" in topic and is_bridge_or_cut(topic):
            return 0.10
    except Exception:
        logger.debug("notify.arbiter: bridge lookup failed", exc_info=True)
    return 0.0


def _tension_boost(topic: str) -> float:
    """0..0.2 boost if topic matches an open tension question."""
    if not topic:
        return 0.0
    try:
        from app.companion.tensions import list_tensions, STATUS_OPEN
        opens = list_tensions(status=STATUS_OPEN, min_freshness=0.0) or []
        topic_lower = topic.lower()
        for t in opens:
            if topic_lower in (t.question or "").lower():
                return 0.2
    except Exception:
        logger.debug("notify.arbiter: tension lookup failed", exc_info=True)
    return 0.0


def _welfare_breaching() -> bool:
    """True iff the operator is currently in a critical-valence
    welfare state. We use the affect runtime to read recent breaches
    rather than re-computing — that's the canonical source.

    Failure-isolated: if affect isn't available, return False
    (don't arbitrarily restrict notifications)."""
    try:
        from app.affect import welfare
        # read_audit returns recent breaches; we only care if any
        # critical-severity entries are very recent.
        recent = welfare.read_audit(limit=10) or []
        cutoff = time.time() - 600.0  # 10 min window
        for r in recent:
            if not isinstance(r, dict):
                continue
            sev = (r.get("severity") or "").lower()
            ts_str = r.get("ts") or ""
            if sev != "critical" or not ts_str:
                continue
            try:
                from datetime import datetime
                ts = datetime.fromisoformat(
                    ts_str.replace("Z", "+00:00"),
                ).timestamp()
                if ts >= cutoff:
                    return True
            except (ValueError, TypeError):
                continue
    except Exception:
        logger.debug("notify.arbiter: welfare check failed", exc_info=True)
    return False


# ── Decision logic ───────────────────────────────────────────────────────


def arbitrate_notification(
    *,
    title: str,
    body: str = "",
    topic: str | None = None,
    critical: bool = False,
    tag: str = "andrusai",
    metadata: dict[str, Any] | None = None,
) -> ArbitrationResult:
    """Decide what to do with a pending notification. Failure-isolated:
    on ANY exception, defaults to SEND_NOW (better to be noisy than
    silent on edge cases).

    Critical bypass is checked FIRST so welfare/security alerts never
    hit arbitration logic that could degrade.
    """
    # 0. Critical bypass — always send. No arbitration.
    if critical:
        result = ArbitrationResult(
            decision=DECISION_SEND_NOW,
            reason="critical=True bypass",
            salience_score=1.0,
            inputs={"critical": True},
        )
        _record(tag, topic, result, title=title, body=body)
        return result

    # 1. Welfare guard — if operator is in critical-valence state,
    # only critical alerts go through. Non-critical queues.
    try:
        if _welfare_breaching():
            result = ArbitrationResult(
                decision=DECISION_QUEUE,
                reason="welfare envelope breaching; non-critical queued",
                salience_score=0.0,
                inputs={"welfare_breaching": True},
            )
            _record(tag, topic, result, title=title, body=body)
            return result
    except Exception:
        # Welfare check itself failed; don't restrict notifications.
        pass

    # 2. Suppression-rate ceiling — if we've suppressed too much today,
    # force-send to recover ground truth on what we're filtering.
    try:
        suppressed, total, rate = daily_suppression_rate()
        if total >= 5 and rate >= _MAX_DAILY_SUPPRESSION_RATE:
            result = ArbitrationResult(
                decision=DECISION_SEND_NOW,
                reason=(
                    f"suppression-rate ceiling: {suppressed}/{total} "
                    f"({rate:.0%}) >= {_MAX_DAILY_SUPPRESSION_RATE:.0%}; "
                    f"force-sending to maintain ground truth"
                ),
                salience_score=0.5,
                inputs={
                    "suppressed_today": suppressed,
                    "total_today": total,
                    "suppression_rate": rate,
                },
            )
            _record(tag, topic, result, title=title, body=body)
            return result
    except Exception:
        pass

    # 3. Compute salience score.
    interest = _interest_score(topic or "") if topic else 0.0
    pattern = _pattern_boost(topic or "") if topic else 0.0
    tension = _tension_boost(topic or "") if topic else 0.0
    # Q4.2 — person-correlation salience inputs (capped contributions
    # so they augment but never dominate). Failure-isolated; off when
    # corresponding levels disabled.
    centrality = _person_centrality_boost(topic or "")
    bridge = _bridge_boost(topic or "")
    score = min(1.0, interest + pattern + tension + centrality + bridge)

    # 4. Fatigue inputs.
    try:
        recent_global = recent_count(window_hours=4.0)
        recent_topic = recent_count_by_topic(topic or "", window_hours=24.0)
    except Exception:
        recent_global = 0
        recent_topic = 0

    inputs: dict[str, Any] = {
        "interest_score": round(interest, 3),
        "pattern_boost": round(pattern, 3),
        "tension_boost": round(tension, 3),
        "centrality_boost": round(centrality, 3),
        "bridge_boost": round(bridge, 3),
        "computed_salience": round(score, 3),
        "recent_global_4h": recent_global,
        "recent_topic_24h": recent_topic,
    }

    # 5. Decision tree.
    if score >= 0.7:
        decision = DECISION_SEND_NOW
        reason = f"high salience {score:.2f}"
    elif score >= 0.4 and recent_global < _TOTAL_BURST_THRESHOLD:
        decision = DECISION_SEND_NOW
        reason = f"medium salience {score:.2f}; recent_global below burst threshold"
    elif score >= 0.4 and recent_topic < _TOPIC_BURST_THRESHOLD:
        decision = DECISION_QUEUE
        reason = f"medium salience {score:.2f}; queue for digest (global burst)"
    elif recent_global >= _TOTAL_BURST_THRESHOLD:
        decision = DECISION_SUPPRESS
        reason = (
            f"low salience {score:.2f} during burst "
            f"({recent_global} sent in last 4h)"
        )
    else:
        decision = DECISION_QUEUE
        reason = f"low salience {score:.2f}; queue for digest"

    result = ArbitrationResult(
        decision=decision,
        reason=reason,
        salience_score=score,
        inputs=inputs,
    )
    _record(tag, topic, result, title=title, body=body)
    return result


def _record(
    tag: str,
    topic: str | None,
    result: ArbitrationResult,
    *,
    title: str | None = None,
    body: str | None = None,
) -> None:
    """Best-effort fatigue-store append. Never raises.

    Q4.1: title + body are forwarded to ``record_event`` so the
    ``queue_for_digest`` path retains the body for later digest
    assembly. The fatigue helper retains them only for that decision
    kind — send_now/suppress paths drop the body for storage discipline.
    """
    try:
        record_event(
            tag=tag,
            topic=topic,
            decision=result.decision,
            salience_score=result.salience_score,
            title=title,
            body=body,
        )
    except Exception:
        logger.debug("notify.arbiter: fatigue record failed", exc_info=True)
