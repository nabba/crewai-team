"""Goodhart enforcing-mode auto-proposer (Phase D #3, 2026-05-09).

The Goodhart hard-gate (Wave 3 #2) ships in three modes:

    Off  →  Advisory  →  Enforcing

Default is **Advisory**: the gate scores severity but does NOT block
promotion. The user explicitly wanted a 2-week observation window
before flipping to **Enforcing** so they could verify the false-
positive rate before letting it block real work.

This module observes the gate during the Advisory window and proposes
the flip when the conditions warrant. It never flips the runtime
setting itself — operator approves through the existing React
``/cp/settings`` Goodhart-hard-gate card.

Algorithm:

  1. Skip entirely if gate is already Off (operator disabled it) or
     Enforcing (already flipped).
  2. Walk ``workspace/goodhart_reports.json`` for the last
     ``OBSERVATION_DAYS`` (default 14) of detected gaming signals.
  3. Walk PG ``control_plane.audit_log`` for promotion decisions in
     the same window. For each ``promotion_decision`` row, check
     whether the gate FLAGGED it as high-severity AND whether it
     would have BLOCKED it under enforcing.
  4. Compute:
        * ``flag_count``     — high-severity signals
        * ``promotion_count``— total promotion decisions
        * ``would_block_pct``— % of promotions that would have been
                               blocked if gate had been enforcing
        * ``avg_severity``   — distribution of severities
  5. Propose the flip when ALL of:
        * ≥ ``MIN_OBSERVATIONS`` promotions in the window
        * ``would_block_pct`` ≤ ``MAX_BLOCK_PCT`` (default 5%)
        * highest sustained severity ≤ medium (no critical incidents)
        * no proposal in the last 14 days

Cadence: 24 h. Runs as a LIGHT idle job in companion/loop.

Output: append to ``workspace/governance_proposals.jsonl`` with
``threshold = "goodhart_enforcing"`` so operators see one unified
proposal stream.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_PROPOSALS_PATH = Path("/app/workspace/governance_proposals.jsonl")
_GOODHART_REPORTS = Path("/app/workspace/goodhart_reports.json")
_STATE_FILE = "goodhart_enforcing_proposer.json"
_RUN_CADENCE_S = 24 * 3600

_OBSERVATION_DAYS = 14
_MIN_OBSERVATIONS = 30          # ≥ N promotion decisions in window
_MAX_BLOCK_PCT = 0.05           # gate would have blocked < 5% of them
_PROPOSAL_DEDUP_DAYS = 14


def _enabled() -> bool:
    return os.getenv("GOODHART_ENFORCING_PROPOSER_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


# ── Mode reads ────────────────────────────────────────────────────────────


def _gate_mode() -> str:
    """Return one of "off" | "advisory" | "enforcing"."""
    try:
        from app.runtime_settings import (
            get_goodhart_hard_gate_disabled,
            get_goodhart_hard_gate_enforcing,
        )
    except Exception:
        return "advisory"
    if get_goodhart_hard_gate_disabled():
        return "off"
    if get_goodhart_hard_gate_enforcing():
        return "enforcing"
    return "advisory"


# ── Signal/promotion scans ────────────────────────────────────────────────


def _read_recent_gaming_signals(window_days: int) -> list[dict]:
    """Pull the gaming signals detected in the last ``window_days``."""
    if not _GOODHART_REPORTS.exists():
        return []
    try:
        rows = json.loads(_GOODHART_REPORTS.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            return []
    except Exception:
        return []
    cutoff = time.time() - window_days * 86400
    out: list[dict] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        ts = r.get("detected_at")
        try:
            if ts is None or float(ts) < cutoff:
                continue
        except (TypeError, ValueError):
            continue
        out.append(r)
    return out


def _read_recent_promotions(window_days: int) -> list[dict]:
    """Pull promotion-decision audit rows for the window. [] on PG miss."""
    try:
        from app.control_plane.audit import AuditTrail
    except Exception:
        return []
    try:
        trail = AuditTrail()
        since = datetime.now(timezone.utc) - timedelta(days=window_days)
        return trail.query(action_prefix="promotion_", since=since, limit=500) or []
    except Exception:
        return []


# ── Decision ──────────────────────────────────────────────────────────────


def _highest_severity(signals: list[dict]) -> str:
    """Return the maximum severity over the signal list."""
    rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    best = "none"
    for s in signals:
        sev = str(s.get("severity") or "").lower()
        if rank.get(sev, 0) > rank.get(best, 0):
            best = sev
    return best


def _last_proposal_age_days() -> float | None:
    if not _PROPOSALS_PATH.exists():
        return None
    most_recent: datetime | None = None
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
                if row.get("threshold") != "goodhart_enforcing":
                    continue
                ts = row.get("ts", "")
                try:
                    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue
                if most_recent is None or dt > most_recent:
                    most_recent = dt
    except OSError:
        return None
    if most_recent is None:
        return None
    return (datetime.now(timezone.utc) - most_recent).total_seconds() / 86400


def _build_proposal(
    signals: list[dict], promotions: list[dict], window_days: int,
) -> dict | None:
    n_promotions = len(promotions)
    if n_promotions < _MIN_OBSERVATIONS:
        return None

    high_signals = [s for s in signals if s.get("severity") == "high"]
    medium_signals = [s for s in signals if s.get("severity") == "medium"]
    # Conservative model: assume each high signal blocks 1 promotion.
    # (The actual `evaluate_promotion` joins against per-promotion
    # severity in detail_json; for the proposer we use this proxy.)
    would_block = len(high_signals)
    block_pct = would_block / max(1, n_promotions)
    if block_pct > _MAX_BLOCK_PCT:
        return None

    highest = _highest_severity(signals)
    if highest == "high" and len(high_signals) > 2:
        # If we accumulated multiple high-severity events, that's a
        # signal NOT to flip yet — there's a real gaming pattern that
        # needs investigation first.
        return None

    age = _last_proposal_age_days()
    if age is not None and age < _PROPOSAL_DEDUP_DAYS:
        return None

    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(days=window_days)
    return {
        "ts": window_end.isoformat(),
        "threshold": "goodhart_enforcing",
        "proposed_value": "enforcing",
        "current_effective": "advisory",
        "evidence": {
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "n_promotions": n_promotions,
            "n_signals_total": len(signals),
            "n_high": len(high_signals),
            "n_medium": len(medium_signals),
            "would_block_pct": round(block_pct, 4),
            "highest_severity": highest,
        },
        "rationale": (
            f"Goodhart hard-gate has been in Advisory mode for "
            f"{window_days} days. Over {n_promotions} promotion "
            f"decisions, only {len(high_signals)} would have been "
            f"blocked under Enforcing ({block_pct:.1%} ≤ "
            f"{_MAX_BLOCK_PCT:.0%}). Highest severity in the window: "
            f"{highest}. Conditions warrant flipping to Enforcing — "
            f"approve via React `/cp/settings`."
        ),
    }


# Shares retention with auto_propose since both write to the same file.
_PROPOSALS_MAX_LINES = 1000  # Phase F #7


def _append_proposal(proposal: dict) -> None:
    try:
        from app.utils.jsonl_retention import append_with_cap
        append_with_cap(
            _PROPOSALS_PATH, json.dumps(proposal, sort_keys=True),
            _PROPOSALS_MAX_LINES,
        )
    except Exception:
        logger.debug(
            "goodhart_enforcing_proposer: append failed", exc_info=True,
        )


# ── Main ──────────────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ran": False, "mode": "", "proposed": False,
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

    mode = _gate_mode()
    summary["mode"] = mode
    if mode != "advisory":
        # Off → operator chose to disable; respect that.
        # Enforcing → already done; nothing to propose.
        write_state_json(_STATE_FILE, state)
        return summary

    signals = _read_recent_gaming_signals(_OBSERVATION_DAYS)
    promotions = _read_recent_promotions(_OBSERVATION_DAYS)
    proposal = _build_proposal(signals, promotions, _OBSERVATION_DAYS)
    if proposal is None:
        write_state_json(_STATE_FILE, state)
        return summary

    _append_proposal(proposal)
    summary["proposed"] = True

    body = (
        f"⚖️ Goodhart hard-gate auto-propose: ready to flip "
        f"`Advisory → Enforcing`.\n\n"
        f"{proposal['rationale']}\n\n"
        f"Audit trail: `workspace/governance_proposals.jsonl`."
    )
    try:
        send_signal_alert(body, tag="gov_auto_propose:goodhart_enforcing")
    except Exception:
        logger.debug(
            "goodhart_enforcing_proposer: alert failed", exc_info=True,
        )

    # Q1.4 (PROGRAM §40.4) — also publish to the SubIA Global Workspace
    # so the consciousness layer sees a pending substrate-governance
    # event, not just the operator. High-salience dispositional event:
    # the gating regime that decides which promotions ship is about to
    # tighten. Best-effort; failure non-fatal (Signal alert already
    # surfaced the proposal).
    _publish_proposal_to_gw(proposal)

    write_state_json(_STATE_FILE, state)
    audit_event(
        "goodhart_enforcing_propose_pass",
        proposed=True,
        n_promotions=len(promotions),
        n_signals=len(signals),
    )
    return summary


def _publish_proposal_to_gw(proposal: dict) -> None:
    """Best-effort GW publish. Never raises."""
    try:
        from app.workspace_publish import publish_to_workspace
        evidence = proposal.get("evidence") or {}
        n_promotions = int(evidence.get("n_promotions") or 0)
        block_pct = float(evidence.get("would_block_pct") or 0.0)
        publish_to_workspace(
            source="goodhart_enforcing_proposer",
            content=(
                f"Goodhart hard-gate auto-proposal: Advisory → Enforcing "
                f"({n_promotions} promotions in window, "
                f"{block_pct:.1%} would-block rate). Awaiting operator "
                f"approval via /cp/settings."
            ),
            salience=0.75,                # high — substrate-governance event
            signal_type="disposition",    # gating regime change is dispositional
        )
    except Exception:
        logger.debug(
            "goodhart_enforcing_proposer: GW publish failed", exc_info=True,
        )
