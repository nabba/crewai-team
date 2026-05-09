"""Runbook D — apscheduler_overrun_alert.

Triggers on the warning emitted by APScheduler when a job's run time
exceeds its trigger interval (``Run time of job "..." was missed by
N:NN:NN.NNN``). Per-job tracking; alerts when a single job overruns
≥3× in 24h.

Auto-demote (moving the job to a longer schedule) would require
mutating ``app/idle_scheduler.py`` (TIER_GATED) or ``app/main.py``
(TIER_IMMUTABLE) — out of scope for an opt-in runbook handler. The
right escalation is therefore Signal alerts that surface the offender
to the operator, plus the persisted tracking file the operator can
inspect.

The dispatcher matches against the SHA-1 of the normalized message,
where the job name has already been replaced with ``"<str>"`` so
all variants of the warning collapse into a single signature.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from app.healing.handlers._common import (
    audit_event,
    compute_signature,
    read_state_json,
    sample_contains,
    send_signal_alert,
    write_state_json,
)
from app.healing.runbooks import RunbookResult, register_runbook

logger = logging.getLogger(__name__)

_LOGGER = "apscheduler.executors.default"
# Concrete sample observed in the wild — the SHA-1 normalisation
# collapses all jobs into one signature thanks to the ``"<str>"``
# replacement of the quoted job name.
_MESSAGE = (
    'Run time of job "lifespan.<locals>._heartbeat_tick (trigger: '
    'interval[0:01:00], next run at: 2026-05-02 06:58:17 UTC)" was '
    'missed by 0:00:05.387825'
)
_SIGNATURE = compute_signature(_LOGGER, _MESSAGE)

_STATE_FILE = "scheduler_overruns.json"
_ALERT_THRESHOLD = 3  # overruns per job per 24h before alert
_ALERT_WINDOW_S = 24 * 3600

_JOB_NAME_RE = re.compile(r'job\s+"([^"]+)"')


def _extract_job_name(sample: str) -> str:
    m = _JOB_NAME_RE.search(sample or "")
    return m.group(1).split(" ", 1)[0] if m else "<unknown>"


def _handle(anomaly: dict[str, Any]) -> RunbookResult:
    if not sample_contains(anomaly, "run time of job"):
        return RunbookResult(
            name="apscheduler_overrun_alert",
            success=False,
            detail="sample mismatch",
            error="sample_mismatch",
        )

    job_name = _extract_job_name(anomaly.get("pattern_sample") or "")
    now = time.time()

    state = read_state_json(_STATE_FILE, {"jobs": {}})
    jobs = state.setdefault("jobs", {})
    entry = jobs.setdefault(job_name, {"events": [], "last_alert_at": 0})

    # Drop events older than the window.
    entry["events"] = [t for t in entry.get("events", []) if t >= now - _ALERT_WINDOW_S]
    entry["events"].append(now)
    entry["last_seen"] = now

    breach = len(entry["events"]) >= _ALERT_THRESHOLD
    cool = (now - entry.get("last_alert_at", 0)) >= _ALERT_WINDOW_S

    write_state_json(_STATE_FILE, state)

    audit_event(
        "apscheduler_overrun",
        job=job_name,
        events_in_window=len(entry["events"]),
        threshold=_ALERT_THRESHOLD,
        pattern_signature=anomaly.get("pattern_signature"),
    )

    if breach and cool:
        entry["last_alert_at"] = now
        write_state_json(_STATE_FILE, state)
        send_signal_alert(
            f"⏱️  Self-heal: APScheduler job `{job_name}` overran its window "
            f"{len(entry['events'])} times in the last 24 h. Consider moving "
            f"it to a longer cron or splitting work. Details in "
            f"`workspace/self_heal/scheduler_overruns.json`.",
            tag="apscheduler_overrun_alert",
        )
        return RunbookResult(
            name="apscheduler_overrun_alert",
            success=True,
            detail=f"alerted on {job_name} ({len(entry['events'])} overruns)",
        )

    return RunbookResult(
        name="apscheduler_overrun_alert",
        success=True,
        detail=f"tracked {job_name} ({len(entry['events'])}/{_ALERT_THRESHOLD})",
    )


def register() -> None:
    register_runbook("apscheduler_overrun_alert", _SIGNATURE, _handle)
