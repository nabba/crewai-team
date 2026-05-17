"""Operator notifications + state-advance daemon for Tier-3 amendments.

The Tier-3 amendment protocol (``app.governance_amendment.protocol``)
is TIER_IMMUTABLE — its state machine + audit chain + eligibility
gate are part of the DGM safety core. This module sits OUTSIDE that
package and provides the mutable concerns the protocol deliberately
omits:

  1. **Signal alerts** on operator-relevant state transitions
     (PROPOSED, STAGED, ELIGIBILITY_FAILED, COOLDOWN_OK, APPLIED,
     STABLE, REVERTED). Without these the operator only sees
     proposals via the audit log; with them, agent-driven amendments
     surface as actionable Signal messages.

  2. **GW publish** to the SubIA Global Workspace on every
     transition so consciousness probes / scene / decentered
     reflection see governance-shaping events.

  3. **State-advance daemon** that polls STAGED proposals once a
     day and calls ``advance_cooldown`` after 7d, then APPLIED
     proposals and calls ``advance_monitoring`` after 30d. The
     protocol is operator-driven by design but the cadence advance
     is safe to automate (it's a TIME-based gate, not a decision
     gate — operator approval still happens between
     COOLDOWN_OK and APPLIED).

The daemon never approves / applies / reverts — those remain
operator actions. Cooldown and monitoring advance are pure
time-based transitions documented in the protocol's docstring as
"daemon / operator calls this once per day."

Master switch: ``TIER3_GOVERNANCE_NOTIFIER_ENABLED`` (default
``true``). The daemon also short-circuits if the protocol itself
is disabled — there's nothing to advance if no proposals exist.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────


_DAEMON_THREAD_NAME = "governance-notifier"
_WARMUP_S = 120
_POLL_INTERVAL_S = 6 * 3600   # 4× per day; cheap — just walks the proposals dir

# State-snapshot file: tracks the LAST observed state per proposal id
# so we can detect transitions between daemon passes. Lives at the
# workspace tier (mutable, cleanable).
_SNAPSHOT_FILENAME = "tier3_amendment_observed_states.json"


# ── Snapshot persistence ─────────────────────────────────────────────


def _snapshot_path() -> Path:
    base = Path(os.environ.get("CHANGE_REQUESTS_DIR")
                or "/app/workspace/change_requests")
    base.mkdir(parents=True, exist_ok=True)
    return base / _SNAPSHOT_FILENAME


def _load_snapshot() -> dict[str, str]:
    p = _snapshot_path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("governance_notifier: snapshot read failed", exc_info=True)
        return {}
    return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def _save_snapshot(snapshot: dict[str, str]) -> None:
    p = _snapshot_path()
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(snapshot, indent=2, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(p)
    except OSError:
        logger.debug("governance_notifier: snapshot write failed", exc_info=True)


# ── Notification helpers ─────────────────────────────────────────────


# URL builders live in app/dashboard_links so auto-deploy and other
# approval-style Signal messages can share them.
from app.dashboard_links import (  # noqa: E402
    signal_links_block as _render_links_block,
    url_iphone as _url_iphone,
    url_macbook as _url_macbook,
)


def _dashboard_url(path: str) -> str:
    """Backwards-compat alias — old name returned the iPhone URL."""
    return _url_iphone(path)


# Operator-relevant transitions. Each entry is (state-name, salience,
# message_template). Other states (PROPOSED, COOLDOWN_FAILED,
# REJECTED) get no Signal alert because either (a) the agent action
# already includes notification or (b) terminal failures don't need
# operator attention. {links} is the rendered block with both the
# iPhone (Funnel HTTPS) and Mac (Tailnet dev) URLs — Tier-3
# amendments stay React-only by design.
_NOTIFY_STATES: dict[str, tuple[float, str]] = {
    "staged": (
        0.5,
        "🏛️ Tier-3 amendment STAGED (id={id}): {target}\n"
        "Eligibility passed; 7-day cooldown started. Cooldown advances "
        "automatically — you'll be re-pinged when COOLDOWN_OK.\n"
        "Proposer: {proposer} · Citation: {citation}\n"
        "{links}",
    ),
    "eligibility_failed": (
        0.4,
        "🏛️ Tier-3 amendment ELIGIBILITY_FAILED (id={id}): {target}\n"
        "Failures: {failures}\n"
        "Proposal recorded for audit; no further action needed unless "
        "you want to investigate the eligibility metrics.\n"
        "{links}",
    ),
    "cooldown_ok": (
        0.7,
        "🏛️ Tier-3 amendment COOLDOWN_OK (id={id}): {target}\n"
        "7-day cooldown clean. Awaiting your approve/reject decision.\n"
        "Proposer: {proposer} · Citation: {citation}\n"
        "👉 Review + approve:\n{links}",
    ),
    "approved": (
        0.6,
        "🏛️ Tier-3 amendment APPROVED (id={id}): {target}\n"
        "Operator approved. Will be applied via host bridge next.\n"
        "{links}",
    ),
    "applied": (
        0.7,
        "🏛️ Tier-3 amendment APPLIED (id={id}): {target}\n"
        "Hot-applied + auto-PR opened. 30-day monitoring window "
        "started — auto-rollback if alignment / regression signal "
        "fires.\n"
        "{links}",
    ),
    "stable": (
        0.5,
        "🏛️ Tier-3 amendment STABLE (id={id}): {target}\n"
        "30-day monitoring clean. Amendment is now durable.\n"
        "{links}",
    ),
    "reverted": (
        0.8,
        "⚠️ Tier-3 amendment REVERTED (id={id}): {target}\n"
        "Monitoring detected a regression signal during the 30-day "
        "window. Hot-reverted automatically. See "
        "workspace/governance/tier3_amendments/audit.jsonl for "
        "details.\n"
        "{links}",
    ),
}


def _format_message(template: str, proposal: Any) -> str:
    failures = (
        ", ".join(getattr(proposal, "eligibility_failures", None) or [])
        or "(none)"
    )
    citation = (getattr(proposal, "citation", "") or "")[:200]
    pid = getattr(proposal, "id", "?")
    body = template.format(
        id=pid,
        target=getattr(proposal, "target_path", "?"),
        proposer=getattr(proposal, "proposer", "?"),
        failures=failures,
        citation=citation,
        links=_render_links_block(f"/cp/amendments/{pid}"),
    )
    # Q2 §39: append the path-keyed history that
    # ``request_tier3_amendment`` persisted into proposal.evidence.
    # The summary line is enough for a Signal message — full
    # detail is in the proposal record + dashboard.
    history_tail = _format_history_tail(proposal)
    if history_tail:
        body = f"{body}\n\n{history_tail}"
    return body


def _format_history_tail(proposal: Any) -> str:
    """One-line history tail extracted from proposal.evidence.

    The tool persisted the full ``relevant_history(target_path)``
    dict under ``evidence['relevant_history_90d']`` — we surface
    only the summary line in the Signal alert (operator can
    /cp/amendments for the full detail). Empty string when no
    history is recorded or the activity was nil.
    """
    evidence = getattr(proposal, "evidence", None) or {}
    history = evidence.get("relevant_history_90d") if isinstance(evidence, dict) else None
    if not isinstance(history, dict):
        return ""
    counts = history.get("counts") or {}
    total = int(counts.get("ledger", 0) or 0) + int(counts.get("cr_audit", 0) or 0)
    if total == 0:
        return ""
    summary = history.get("summary_line") or ""
    return f"📜 Recent activity: {summary}"


def _send_signal_alert(message: str) -> None:
    """Best-effort Signal alert — never raises."""
    try:
        from app.signal_client import send_message
        from app.config import get_settings
        recipient = (get_settings().signal_owner_number or "").strip()
        if not recipient:
            return
        send_message(recipient, message)
    except Exception:
        logger.debug("governance_notifier: signal alert failed", exc_info=True)


def _publish_to_workspace(*, proposal: Any, salience: float, summary: str) -> None:
    """Best-effort GW publish — never raises."""
    try:
        from app.workspace_publish import publish_to_workspace
        publish_to_workspace(
            source="tier3-amendment",
            content=summary,
            salience=salience,
            signal_type="disposition",
        )
    except Exception:
        logger.debug("governance_notifier: GW publish failed", exc_info=True)


def notify_proposal_created(proposal: Any) -> None:
    """Called by the agent tool right after a successful proposal.

    Sends an immediate Signal alert + GW publish for the proposal's
    state. The state is already either STAGED (eligibility passed) or
    ELIGIBILITY_FAILED — both are operator-relevant.
    """
    state = getattr(getattr(proposal, "state", None), "value", "")
    notice = _NOTIFY_STATES.get(state)
    if notice is None:
        return
    salience, template = notice
    message = _format_message(template, proposal)
    _send_signal_alert(message)
    _publish_to_workspace(
        proposal=proposal, salience=salience, summary=message[:200],
    )

    # Update the snapshot so the daemon doesn't double-alert.
    snap = _load_snapshot()
    snap[getattr(proposal, "id", "")] = state
    _save_snapshot(snap)


# ── State-advance + transition detection ─────────────────────────────


def _enabled() -> bool:
    return os.getenv("TIER3_GOVERNANCE_NOTIFIER_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _list_proposals() -> list[Any]:
    try:
        from app.governance_amendment import store as _store
    except Exception:
        return []
    try:
        return list(_store.list_all())
    except Exception:
        return []


def _try_advance_cooldown(proposal: Any) -> Optional[Any]:
    """For STAGED proposals: if 7d elapsed since staged_at, advance.

    Returns the updated proposal or None if no transition occurred.
    """
    try:
        from app.governance_amendment.protocol import advance_cooldown
    except Exception:
        return None
    try:
        advanced = advance_cooldown(proposal.id)
        return advanced
    except Exception:
        # Possible reasons: not yet 7d elapsed (no-op), state already
        # past STAGED. The protocol's docstring says these are
        # benign returns — we treat any exception as "didn't
        # advance" and try again next pass.
        return None


def _try_advance_monitoring(proposal: Any) -> Optional[Any]:
    """For APPLIED proposals: if 30d elapsed, advance to STABLE."""
    try:
        from app.governance_amendment.protocol import advance_monitoring
    except Exception:
        return None
    try:
        return advance_monitoring(proposal.id)
    except Exception:
        return None


def run_one_pass() -> dict[str, int]:
    """One notifier pass. Walks all proposals, detects transitions,
    advances time-based gates, sends alerts."""
    counters: dict[str, int] = {
        "seen": 0,
        "transitions_alerted": 0,
        "cooldowns_advanced": 0,
        "monitorings_advanced": 0,
    }
    if not _enabled():
        return counters

    proposals = _list_proposals()
    if not proposals:
        return counters

    snapshot = _load_snapshot()
    new_snapshot: dict[str, str] = {}

    for proposal in proposals:
        counters["seen"] += 1
        pid = getattr(proposal, "id", "")
        if not pid:
            continue
        state_obj = getattr(proposal, "state", None)
        state = getattr(state_obj, "value", "") if state_obj else ""

        # 1. Time-based advance: STAGED → COOLDOWN_OK after 7d.
        if state == "staged":
            advanced = _try_advance_cooldown(proposal)
            if advanced is not None:
                advanced_state = getattr(
                    getattr(advanced, "state", None), "value", "",
                )
                if advanced_state and advanced_state != "staged":
                    counters["cooldowns_advanced"] += 1
                    proposal = advanced
                    state = advanced_state

        # 2. Time-based advance: APPLIED → STABLE after 30d.
        elif state == "applied":
            advanced = _try_advance_monitoring(proposal)
            if advanced is not None:
                advanced_state = getattr(
                    getattr(advanced, "state", None), "value", "",
                )
                if advanced_state and advanced_state != "applied":
                    counters["monitorings_advanced"] += 1
                    proposal = advanced
                    state = advanced_state

        # 3. Transition detection: state differs from snapshot →
        #    operator-relevant alert.
        prior = snapshot.get(pid, "")
        if state != prior:
            notice = _NOTIFY_STATES.get(state)
            if notice is not None:
                salience, template = notice
                message = _format_message(template, proposal)
                _send_signal_alert(message)
                _publish_to_workspace(
                    proposal=proposal, salience=salience,
                    summary=message[:200],
                )
                counters["transitions_alerted"] += 1

        new_snapshot[pid] = state

    _save_snapshot(new_snapshot)
    return counters


# ── Daemon driver ────────────────────────────────────────────────────


_driver_lock = threading.Lock()
_driver_started = False
_stop_event = threading.Event()


def _is_running() -> bool:
    return any(
        t.name == _DAEMON_THREAD_NAME and t.is_alive()
        for t in threading.enumerate()
    )


def _driver() -> None:
    if _stop_event.wait(_WARMUP_S):
        return
    while not _stop_event.is_set():
        try:
            counters = run_one_pass()
            if any(counters.get(k) for k in (
                "transitions_alerted", "cooldowns_advanced",
                "monitorings_advanced",
            )):
                logger.info("governance_notifier: %s", counters)
        except Exception:
            logger.debug("governance_notifier: pass raised", exc_info=True)
        if _stop_event.wait(_POLL_INTERVAL_S):
            return


def start() -> None:
    """Idempotent daemon launch."""
    global _driver_started
    if not _enabled():
        logger.info(
            "governance_notifier: disabled via "
            "TIER3_GOVERNANCE_NOTIFIER_ENABLED",
        )
        return
    with _driver_lock:
        if _is_running():
            return
        if _driver_started:
            logger.warning(
                "governance_notifier: previous thread is dead, re-spawning",
            )
        _stop_event.clear()
        thread = threading.Thread(
            target=_driver, name=_DAEMON_THREAD_NAME, daemon=True,
        )
        thread.start()
        _driver_started = True
        logger.info(
            "governance_notifier: daemon started (warm-up=%ds, poll=%dh)",
            _WARMUP_S, _POLL_INTERVAL_S // 3600,
        )


def stop() -> None:
    _stop_event.set()


# Eager start at import — same pattern as healing/monitors and the
# proposal_bridge promoter.
start()
