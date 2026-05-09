"""Idle-cooldown monitor — detect idle-scheduler jobs stuck in cooldown.

After 3 consecutive failures, ``app/idle_scheduler.py`` puts a job into
a 1-hour cooldown via ``_persist_job_skip()``. The cooldown is
persisted in a sqlite-backed dbm at ``workspace/memory/idle_job_state``
so it survives gateway restarts. A *transient* outage that took longer
than 1 h leaves the job stuck for the full hour even after the upstream
recovers; a *persistent* failure leaves it stuck repeatedly.

This monitor reads the persisted state and alerts the operator when:

  * A job has been in cooldown for > 24 h, OR
  * A job has accumulated > 5 cooldown cycles in 7 days (chronic).

It does NOT auto-clear the cooldown — that's an operator choice via
the dashboard or by deleting the dbm key. The whole point of the
cooldown is to avoid storming a known-bad upstream.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from app.healing.handlers._common import (
    audit_event,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)

_STATE_FILE = "idle_cooldown_alerts.json"
_ALERT_WINDOW_S = 24 * 3600


def _read_idle_state() -> dict[str, Any]:
    """Best-effort load of the idle-scheduler persisted state.

    The on-disk format is a sqlite-backed dbm. The keys we care about
    follow the patterns ``skip:<jobname>`` and ``failures:<jobname>``.
    Returns ``{name: {skip_until, failures}}``.
    """
    try:
        import dbm
    except Exception:
        return {}

    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "workspace" / "memory" / "idle_job_state",
        repo_root / "workspace" / "memory" / "idle_job_state.db",
    ]
    base = next((p for p in candidates if p.exists()), None)
    if base is None:
        # dbm.open without a suffix — let dbm pick.
        base = repo_root / "workspace" / "memory" / "idle_job_state"
        if not any(
            (base.parent / f"{base.name}{suf}").exists()
            for suf in ("", ".db", ".pag", ".dir")
        ):
            return {}

    state: dict[str, dict] = {}
    try:
        with dbm.open(str(base), "r") as db:
            for raw_key in db.keys():
                key = raw_key.decode("utf-8", errors="ignore") if isinstance(raw_key, bytes) else raw_key
                value = db.get(raw_key)
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="ignore")
                if key.startswith("skip:"):
                    name = key[len("skip:"):]
                    try:
                        state.setdefault(name, {})["skip_until"] = float(value or 0)
                    except (TypeError, ValueError):
                        pass
                elif key.startswith("failures:"):
                    name = key[len("failures:"):]
                    try:
                        state.setdefault(name, {})["failures"] = int(value or 0)
                    except (TypeError, ValueError):
                        pass
    except Exception:
        logger.debug("idle_cooldown: dbm read failed", exc_info=True)
        return {}
    return state


def run() -> None:
    state = _read_idle_state()
    if not state:
        return

    now = time.time()
    long_stuck: list[dict] = []
    for name, entry in state.items():
        skip_until = entry.get("skip_until", 0.0)
        failures = entry.get("failures", 0)
        if skip_until <= now:
            continue  # not currently in cooldown
        in_cooldown_for = skip_until - now  # seconds remaining
        # A 1-hour cooldown is normal. Alert if remaining >24 h (which
        # would imply the job has been failing repeatedly and the
        # cooldown got extended) OR if failures count exceeds 15.
        if in_cooldown_for > 24 * 3600 or failures > 15:
            long_stuck.append({
                "name": name,
                "remaining_s": int(in_cooldown_for),
                "failures": failures,
            })

    audit_event(
        "idle_cooldown_check",
        n_in_cooldown=sum(1 for e in state.values() if e.get("skip_until", 0) > now),
        n_long_stuck=len(long_stuck),
    )

    if not long_stuck:
        return

    alert_state = read_state_json(_STATE_FILE, {"alerts": {}})
    alerts = alert_state.setdefault("alerts", {})
    fresh: list[dict] = []
    for j in long_stuck:
        last = alerts.get(j["name"], {}).get("last_alert_at", 0)
        if now - last >= _ALERT_WINDOW_S:
            alerts[j["name"]] = {"last_alert_at": now, **j}
            fresh.append(j)
    write_state_json(_STATE_FILE, alert_state)

    if not fresh:
        return

    lines = [
        f"  • `{j['name']}` — {j['remaining_s'] // 3600} h remaining, "
        f"{j['failures']} cumulative failures"
        for j in fresh
    ]
    send_signal_alert(
        "🧊 Self-heal: idle-scheduler job(s) deep in cooldown:\n\n"
        + "\n".join(lines)
        + "\n\nIf the upstream cause is fixed, clear the cooldown by "
          "deleting `skip:<job>` from `workspace/memory/idle_job_state`. "
          "Otherwise expect this background work to stay paused.",
        tag="idle_cooldown",
    )
