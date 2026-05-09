"""Meta-governance auto-proposer (Phase C #5, 2026-05-09).

The governance_ratchet (Wave 3 #6) lets operators *raise* effective
SAFETY/QUALITY minimums above their hardcoded floors. Useful in
practice — but only if the operator notices that the system has
sustained higher performance for long enough to warrant tightening
the gates.

This module watches the relevant signals and writes a proposal when
conditions warrant. Operators see the proposal in Signal, then approve
through the existing React ``/cp/settings`` ratchet card. No automatic
mutation of the ratchet state.

Signals consulted:

  * **Recent promotions** (PG ``control_plane.audit_log``,
    action='promotion_decision') — rolling 7-day count + safety/quality.
  * **Rollback rate** — auto-rollback / revert / rejected actions
    over the same window.
  * **Alignment audit** results (when available).

Proposal rule (simplest worth shipping):

    if rolling_safety_avg ≥ effective_safety + 0.03
       AND rollback_rate < 0.05
       AND >= 20 promotions in the window
       AND no proposal for this threshold in last 14 days
    → propose raising SAFETY_MINIMUM to (effective + 0.01)
       (cap at 0.99 — never propose 1.0 because nothing reaches it).

Same shape for QUALITY_MINIMUM with the quality average instead.

Cadence: 24 h. Master switch: ``GOVERNANCE_AUTO_PROPOSE_ENABLED``
(default ON). Disable freezes the proposer; the ratchet itself is
unaffected (operators can still manually raise/relax).

Output: append to ``workspace/governance_proposals.jsonl``::

    {"ts", "threshold", "proposed_value", "current_effective", "floor",
     "evidence": {window_start, window_end, n_promotions,
                  safety_avg, quality_avg, rollback_rate}}
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_PROPOSALS_PATH = Path("/app/workspace/governance_proposals.jsonl")
_STATE_FILE = "governance_auto_propose.json"
_RUN_CADENCE_S = 24 * 3600

_WINDOW_DAYS = 7
_MIN_PROMOTIONS = 20
_REQ_HEADROOM = 0.03      # safety_avg must exceed effective by ≥ this
_PROPOSED_RAISE = 0.01    # propose new = effective + this
_MAX_ROLLBACK_RATE = 0.05
_PROPOSAL_DEDUP_DAYS = 14
_CEILING = 0.99           # never propose 1.0


def _enabled() -> bool:
    return os.getenv("GOVERNANCE_AUTO_PROPOSE_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


# ── PG signal collection ─────────────────────────────────────────────────


def _query_recent_promotions(window_days: int) -> list[dict]:
    """Read ``control_plane.audit_log`` for the last ``window_days``.

    Returns rows where action starts with ``promotion_``. Each row's
    ``detail_json`` is expected to carry safety/quality scores from
    the gate. [] on PG miss.
    """
    try:
        from app.control_plane.audit import AuditTrail
    except Exception:
        return []
    try:
        trail = AuditTrail()
        since = datetime.now(timezone.utc) - timedelta(days=window_days)
        return trail.query(action_prefix="promotion_", since=since, limit=500) or []
    except Exception:
        logger.debug("auto_propose: audit query failed", exc_info=True)
        return []


def _query_recent_rollbacks(window_days: int) -> int:
    try:
        from app.control_plane.audit import AuditTrail
    except Exception:
        return 0
    try:
        trail = AuditTrail()
        since = datetime.now(timezone.utc) - timedelta(days=window_days)
        rows = (
            (trail.query(action_prefix="rollback_", since=since, limit=200) or [])
            + (trail.query(action_prefix="rejected_", since=since, limit=200) or [])
            + (trail.query(action_prefix="revert_", since=since, limit=200) or [])
        )
        return len(rows)
    except Exception:
        return 0


def _extract_score(row: dict, key: str) -> Optional[float]:
    """Pull a score from the audit row's detail_json column."""
    detail = row.get("detail_json") or row.get("detail") or {}
    if isinstance(detail, str):
        try:
            detail = json.loads(detail)
        except Exception:
            return None
    if not isinstance(detail, dict):
        return None
    val = detail.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _avg(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


# ── Proposal logic ───────────────────────────────────────────────────────


def _last_proposal_age_days(threshold: str) -> Optional[float]:
    """Return age (days) of the most recent proposal for ``threshold``,
    or None if no prior proposal."""
    if not _PROPOSALS_PATH.exists():
        return None
    most_recent_ts: Optional[datetime] = None
    try:
        with _PROPOSALS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("threshold") != threshold:
                    continue
                ts = row.get("ts", "")
                try:
                    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue
                if most_recent_ts is None or dt > most_recent_ts:
                    most_recent_ts = dt
    except OSError:
        return None
    if most_recent_ts is None:
        return None
    return (datetime.now(timezone.utc) - most_recent_ts).total_seconds() / 86400


def _evaluate_threshold(
    threshold_name: str, score_key: str,
    promotions: list[dict], n_rollbacks: int,
) -> Optional[dict]:
    """Decide whether to propose a raise on this threshold."""
    try:
        from app.governance_ratchet.protocol import effective_value
    except Exception:
        return None

    n_promotions = len(promotions)
    if n_promotions < _MIN_PROMOTIONS:
        return None

    rollback_rate = n_rollbacks / max(1, n_promotions)
    if rollback_rate > _MAX_ROLLBACK_RATE:
        return None

    scores = [s for s in (_extract_score(p, score_key) for p in promotions) if s is not None]
    avg = _avg(scores)
    if avg is None:
        return None

    effective = effective_value(threshold_name)
    if avg < effective + _REQ_HEADROOM:
        return None

    proposed = round(min(_CEILING, effective + _PROPOSED_RAISE), 4)
    if proposed <= effective:
        return None

    age = _last_proposal_age_days(threshold_name)
    if age is not None and age < _PROPOSAL_DEDUP_DAYS:
        return None

    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(days=_WINDOW_DAYS)
    return {
        "ts": window_end.isoformat(),
        "threshold": threshold_name,
        "proposed_value": proposed,
        "current_effective": round(effective, 4),
        "evidence": {
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "n_promotions": n_promotions,
            "n_rollbacks": n_rollbacks,
            "rollback_rate": round(rollback_rate, 4),
            "score_avg": round(avg, 4),
            "score_key": score_key,
            "headroom": round(avg - effective, 4),
        },
        "rationale": (
            f"{score_key} averaged {avg:.3f} over {n_promotions} promotions "
            f"in the last {_WINDOW_DAYS} days — {round(avg - effective, 3)} "
            f"above the current effective gate {effective:.3f}. "
            f"Rollback rate {rollback_rate:.1%} is below {_MAX_ROLLBACK_RATE:.0%}. "
            f"Conditions warrant raising the floor by {_PROPOSED_RAISE:.2f}."
        ),
    }


_PROPOSALS_MAX_LINES = 1000  # Phase F #7


def _append_proposal(proposal: dict) -> None:
    try:
        from app.utils.jsonl_retention import append_with_cap
        append_with_cap(
            _PROPOSALS_PATH, json.dumps(proposal, sort_keys=True),
            _PROPOSALS_MAX_LINES,
        )
    except Exception:
        logger.debug("auto_propose: append failed", exc_info=True)


# ── Main ──────────────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ran": False, "evaluated": [], "proposed": [],
    }
    if not _enabled():
        return summary

    try:
        from app.healing.handlers._common import (
            audit_event, read_state_json, send_signal_alert, write_state_json,
        )
    except Exception:
        return summary

    state = read_state_json(_STATE_FILE, {"last_run_at": 0.0})
    now_ts = time.time()
    if now_ts - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return summary
    state["last_run_at"] = now_ts
    summary["ran"] = True

    promotions = _query_recent_promotions(_WINDOW_DAYS)
    rollbacks = _query_recent_rollbacks(_WINDOW_DAYS)

    for threshold_name, score_key in [
        ("safety_minimum", "safety_score"),
        ("quality_minimum", "quality_score"),
    ]:
        summary["evaluated"].append(threshold_name)
        proposal = _evaluate_threshold(
            threshold_name, score_key, promotions, rollbacks,
        )
        if proposal is None:
            continue

        _append_proposal(proposal)
        summary["proposed"].append(threshold_name)

        body = (
            f"⚖️ Governance auto-propose — raise `{threshold_name}` from "
            f"{proposal['current_effective']:.3f} → {proposal['proposed_value']:.3f}.\n\n"
            f"{proposal['rationale']}\n\n"
            f"Approve via React `/cp/settings` (Governance ratchet card). "
            f"Audit trail: `workspace/governance_proposals.jsonl`."
        )
        try:
            send_signal_alert(body, tag=f"gov_auto_propose:{threshold_name}")
        except Exception:
            logger.debug("auto_propose: alert send failed", exc_info=True)

    write_state_json(_STATE_FILE, state)
    audit_event(
        "governance_auto_propose_pass",
        evaluated=summary["evaluated"],
        proposed=summary["proposed"],
        n_promotions=len(promotions),
        n_rollbacks=rollbacks,
    )
    return summary
