"""Vacation-mode sweep — periodic scan + auto-approve of matching PENDING CRs.

Runs every ``SWEEP_INTERVAL_SECONDS`` (default 5 min). On each pass:

  1. Check vacation engagement; bail if not active or auto-expired.
  2. Check operator_anomaly recent critical alerts; pause for the
     anomaly dedup window if a ``new_sender`` event fired since
     engagement.
  3. List PENDING CRs from the store.
  4. For each, validate against ``validate_vacation_apply``.
  5. Approve via ``lifecycle.approve(..., source=VACATION_AUTO_APPLY)``
     + apply via ``apply.apply_change``.
  6. Emit a NON-arbitrated Signal alert (bypasses suppression).
  7. Append a row to ``workspace/vacation_mode/auto_apply_log.jsonl``.

Each step is failure-isolated. A broken CR cannot poison the whole
sweep; a poisoned sweep cannot bring down the daemon thread.

The daemon thread is started by the boot-anchor in ``app.healing``.
``start_daemon()`` is idempotent and safe to call from a watchdog
re-spawn.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.vacation_mode.allowlist import validate_vacation_apply
from app.vacation_mode.state import (
    current_state,
    disengage,
    is_active,
)

logger = logging.getLogger(__name__)


SWEEP_INTERVAL_SECONDS = 5 * 60
_DAEMON_THREAD_NAME = "vacation-mode-sweep"
_WARMUP_S = 60
_PAUSE_AFTER_NEW_SENDER_S = 24 * 3600  # 24h pause window post new-sender
_LOG_FILE_NAME = "auto_apply_log.jsonl"
_MAX_LOG_ROWS = 1000
_MAX_PER_SWEEP = 5  # never auto-apply more than 5 CRs in a single sweep

# Cross-sweep rate limits — mirror §38.3's contract but tighter.
# A buggy or compromised producer cannot flood vacation mode with
# many small additive CRs filed across many sweeps.
_RATE_LIMIT_PER_REQUESTOR_PER_DAY = 6   # vs §38.3 per-pattern 3/day
_RATE_LIMIT_GLOBAL_PER_DAY = 20         # vs §38.3 global 10/day
_RATE_LIMIT_FILE = "sweep_rate_limit.json"


_driver_lock = threading.Lock()
_stop_event = threading.Event()


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _log_path() -> Path:
    return _workspace() / "vacation_mode" / _LOG_FILE_NAME


def _rate_limit_path() -> Path:
    return _workspace() / "vacation_mode" / _RATE_LIMIT_FILE


def _today_key(ts: float) -> str:
    """UTC YYYY-MM-DD key for daily-bucket counters."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _read_rate_state() -> dict[str, Any]:
    p = _rate_limit_path()
    if not p.exists():
        return {"day": None, "global_count": 0, "per_requestor": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"day": None, "global_count": 0, "per_requestor": {}}


def _write_rate_state(state: dict[str, Any]) -> None:
    p = _rate_limit_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8",
        )
    except Exception:
        logger.debug(
            "vacation_mode.sweep: rate-state write failed", exc_info=True,
        )


def _rate_limit_ok(requestor: str, *, now: float) -> tuple[bool, Optional[str]]:
    """Check + increment the daily rate-limit counters. Returns
    ``(ok, reason)``. Increments only when ``ok=True``."""
    state = _read_rate_state()
    today = _today_key(now)
    # Reset counters at UTC day rollover.
    if state.get("day") != today:
        state = {"day": today, "global_count": 0, "per_requestor": {}}
    global_count = int(state.get("global_count", 0))
    if global_count >= _RATE_LIMIT_GLOBAL_PER_DAY:
        return False, (
            f"global daily cap reached "
            f"({global_count}/{_RATE_LIMIT_GLOBAL_PER_DAY})"
        )
    per_req = state.setdefault("per_requestor", {})
    if not isinstance(per_req, dict):
        per_req = {}
        state["per_requestor"] = per_req
    req_count = int(per_req.get(requestor, 0))
    if req_count >= _RATE_LIMIT_PER_REQUESTOR_PER_DAY:
        return False, (
            f"per-requestor daily cap reached for {requestor!r} "
            f"({req_count}/{_RATE_LIMIT_PER_REQUESTOR_PER_DAY})"
        )
    # Reserve the slot now (caller increments AFTER apply succeeds via
    # ``_commit_rate_limit``).
    return True, None


def _commit_rate_limit(requestor: str, *, now: float) -> None:
    """Atomically bump the daily counters after a successful
    auto-apply. Idempotent over crash-recovery: at worst we lose
    one slot if the gateway dies between apply + commit."""
    state = _read_rate_state()
    today = _today_key(now)
    if state.get("day") != today:
        state = {"day": today, "global_count": 0, "per_requestor": {}}
    state["global_count"] = int(state.get("global_count", 0)) + 1
    per_req = state.setdefault("per_requestor", {})
    if not isinstance(per_req, dict):
        per_req = {}
        state["per_requestor"] = per_req
    per_req[requestor] = int(per_req.get(requestor, 0)) + 1
    _write_rate_state(state)


def _append_log(row: dict[str, Any]) -> None:
    p = _log_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # Cap rows: read existing → drop oldest → append new → write.
        existing: list[str] = []
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    existing = [line for line in f if line.strip()]
            except OSError:
                existing = []
        if len(existing) >= _MAX_LOG_ROWS:
            existing = existing[-(_MAX_LOG_ROWS - 1):]
        existing.append(json.dumps(row, sort_keys=True) + "\n")
        with open(p, "w", encoding="utf-8") as f:
            f.writelines(existing)
    except Exception:
        logger.debug("vacation_mode.sweep: log write failed", exc_info=True)


def _anomaly_pause_active(now: float) -> bool:
    """True iff the operator_anomaly monitor has emitted a critical
    ``new_sender`` event since the current engagement began AND the
    ``_PAUSE_AFTER_NEW_SENDER_S`` window has not expired.

    Q16.1 Item 7: reads ``operator_anomaly.last_critical_alert_at``
    rather than touching the monitor's private state-file schema.
    Cross-module consumers must use the public function so the
    state-file format can evolve without breaking vacation_mode.

    Failure-isolated: returns False on any error (vacation mode does
    NOT pause-by-default if the check itself breaks; the loud Signal
    alerts on each auto-apply compensate)."""
    try:
        state = current_state()
        if not state.engaged or state.engagement is None:
            return False
        engagement_start = state.engagement.engaged_at
        from app.healing.monitors.operator_anomaly import (
            last_critical_alert_at,
        )
        new_sender_at = last_critical_alert_at("new_sender")
        if new_sender_at is None:
            return False
        if new_sender_at <= engagement_start:
            return False
        # Pause iff inside the 24h window after the new_sender alert.
        return (now - new_sender_at) < _PAUSE_AFTER_NEW_SENDER_S
    except Exception:
        logger.debug(
            "vacation_mode.sweep: anomaly-pause check failed",
            exc_info=True,
        )
        return False


def _list_pending_crs() -> list[Any]:
    """Read PENDING CRs from the store. Failure-isolated."""
    try:
        from app.change_requests import store, Status
        # Different store backends expose list_by_status differently;
        # try a few shapes.
        if hasattr(store, "list_by_status"):
            return list(store.list_by_status(Status.PENDING))
        if hasattr(store, "list"):
            return [
                cr for cr in store.list()
                if getattr(cr, "status", None) == Status.PENDING
            ]
    except Exception:
        logger.debug(
            "vacation_mode.sweep: pending listing failed", exc_info=True,
        )
    return []


def _approve_and_apply(cr: Any) -> dict[str, Any]:
    """Approve via the existing lifecycle.approve(...) + apply via
    apply_change. Returns a result row for the log."""
    started_at = time.time()
    out: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_id": getattr(cr, "request_id", None),
        "path": getattr(cr, "path", None),
        "requestor": getattr(cr, "requestor", None),
        "status_pre": getattr(cr.status, "value", str(cr.status))
            if hasattr(cr, "status") else "unknown",
        "elapsed_s": 0.0,
        "ok": False,
        "error": None,
        "status_post": None,
    }
    try:
        from app.change_requests import DecisionSource, lifecycle
        lifecycle.approve(
            cr.request_id,
            source=DecisionSource.VACATION_AUTO_APPLY,
            decision_reason=(
                "vacation_mode auto-approve under pre-staged allowlist"
            ),
        )
        # Apply.
        try:
            from app.change_requests.apply import apply_change
            apply_change(cr.request_id)
        except Exception as exc:
            out["error"] = f"apply: {type(exc).__name__}: {exc}"
            out["elapsed_s"] = round(time.time() - started_at, 3)
            return out
        # Re-fetch to capture post status.
        try:
            from app.change_requests import store as _store
            cr_post = _store.get(cr.request_id)
            if cr_post is not None and hasattr(cr_post, "status"):
                out["status_post"] = (
                    getattr(cr_post.status, "value", str(cr_post.status))
                )
        except Exception:
            pass
        out["ok"] = True
    except Exception as exc:
        out["error"] = f"approve: {type(exc).__name__}: {exc}"
    out["elapsed_s"] = round(time.time() - started_at, 3)
    return out


def _send_loud_alert(result: dict[str, Any]) -> None:
    """Non-arbitrated, critical Signal alert. Operator should ALWAYS
    see vacation auto-applies, even during arbiter suppression."""
    try:
        from app.notify import notify
        body = (
            f"⏸️ Vacation auto-apply landed on "
            f"`{result.get('path')}` (requestor=`{result.get('requestor')}`, "
            f"CR id=`{result.get('request_id')}`).\n\n"
            f"Status: {'✅ applied' if result.get('ok') else '⚠️ failed — ' + str(result.get('error'))}\n"
            f"60-min rollback window active. Disengage via "
            f"`/vacation disengage` if anything looks wrong."
        )
        notify(
            title="⏸️ Vacation auto-apply",
            body=body,
            url="/cp/changes",
            topic="vacation_mode:auto_apply",
            critical=True,        # bypass arbiter suppression
            arbitrate=False,       # no soft-route
        )
    except Exception:
        logger.debug(
            "vacation_mode.sweep: notify failed", exc_info=True,
        )


def sweep_pending(*, now: Optional[float] = None) -> dict[str, Any]:
    """One sweep pass. Returns a summary dict.

    Public entry point for tests."""
    summary: dict[str, Any] = {
        "ran": False,
        "active": False,
        "paused_on_anomaly": False,
        "n_pending": 0,
        "n_validated": 0,
        "n_auto_approved": 0,
        "n_failed": 0,
        "results": [],
    }
    cur = float(now) if now is not None else time.time()

    if not is_active():
        return summary
    summary["active"] = True

    if _anomaly_pause_active(cur):
        summary["paused_on_anomaly"] = True
        return summary

    pending = _list_pending_crs()
    summary["n_pending"] = len(pending)
    if not pending:
        return summary

    summary["ran"] = True

    auto_applied = 0
    rate_limited: list[dict[str, Any]] = []
    for cr in pending:
        if auto_applied >= _MAX_PER_SWEEP:
            break
        try:
            result = validate_vacation_apply(
                path=getattr(cr, "path", ""),
                new_content=getattr(cr, "new_content", "") or "",
                old_content=getattr(cr, "old_content", "") or "",
                requestor=getattr(cr, "requestor", ""),
            )
        except Exception:
            logger.debug(
                "vacation_mode.sweep: validator raised", exc_info=True,
            )
            continue
        if not result.ok:
            continue
        summary["n_validated"] += 1

        # Cross-sweep rate limit (Q16 follow-on, PROGRAM §51).
        requestor = getattr(cr, "requestor", "")
        ok, reason = _rate_limit_ok(requestor, now=cur)
        if not ok:
            rate_limited.append({
                "request_id": getattr(cr, "request_id", None),
                "requestor": requestor,
                "reason": reason,
            })
            continue

        row = _approve_and_apply(cr)
        summary["results"].append(row)
        if row.get("ok"):
            summary["n_auto_approved"] += 1
            auto_applied += 1
            _commit_rate_limit(requestor, now=cur)
        else:
            summary["n_failed"] += 1
        _append_log(row)
        _send_loud_alert(row)

    summary["rate_limited"] = rate_limited
    return summary


# ── Daemon ───────────────────────────────────────────────────────────────


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
            sweep_pending()
        except Exception:
            logger.debug(
                "vacation_mode.sweep: pass raised", exc_info=True,
            )
        if _stop_event.wait(SWEEP_INTERVAL_SECONDS):
            return


def start_daemon() -> None:
    """Idempotent daemon-thread start. Master switch is in
    runtime_settings; the sweep is bounded by ``is_active()`` so the
    daemon being alive is harmless when vacation mode is off."""
    if not _enabled():
        return
    with _driver_lock:
        if _is_running():
            return
        _stop_event.clear()
        thread = threading.Thread(
            target=_driver, name=_DAEMON_THREAD_NAME, daemon=True,
        )
        thread.start()
        logger.info(
            "vacation_mode.sweep: daemon started "
            "(warm-up=%ds, interval=%ds)",
            _WARMUP_S, SWEEP_INTERVAL_SECONDS,
        )


def stop() -> None:
    """Test hook to signal the driver to exit."""
    _stop_event.set()


def _enabled() -> bool:
    """Top-level vacation-mode kill switch. Independent of engagement —
    operator can disable vacation mode entirely via this flag even if
    state has a staged allowlist."""
    try:
        from app.runtime_settings import get_vacation_mode_enabled
        return get_vacation_mode_enabled()
    except Exception:
        return os.getenv(
            "VACATION_MODE_ENABLED", "true",
        ).lower() in ("true", "1", "yes", "on")
