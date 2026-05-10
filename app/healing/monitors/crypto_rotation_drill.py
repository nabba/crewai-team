"""Crypto rotation-drill freshness monitor (§2.1).

Companion to :mod:`app.audit.algorithm_pinning`. Watches the
algorithm-pin manifest:

  * alerts when KNOWN_ARTIFACT_CLASSES has unpinned entries
    (operator hasn't yet recorded what hash algorithm is in use
    for that subsystem);
  * alerts when any pin is older than
    ``CRYPTO_ROTATION_REVIEW_INTERVAL_DAYS`` (default 730 — every
    2 years, the operator should re-confirm the choice still meets
    the threat model);
  * runs a small rotation drill at probe time
    (``run_rotation_drill``) to assert the runtime can compute
    hashes under both the current and a candidate target algorithm
    (default sha256 → sha3_256). A failed drill means the runtime
    isn't ready for the rotation when the operator decides to do it.

This is informational + insurance, not auto-rotation. Rotation
itself coordinates changes across multiple TIER_IMMUTABLE files;
the operator does that when the time comes. This monitor's job is
to surface the readiness signal.

Cadence: weekly probe. Master switch:
``CRYPTO_ROTATION_DRILL_MONITOR_ENABLED`` (default ON).
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from app.audit.algorithm_pinning import (
    KNOWN_ARTIFACT_CLASSES,
    list_pins,
    missing_artifact_classes,
    run_rotation_drill,
    stale_pins,
)
from app.healing.handlers._common import (
    audit_event,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)


_STATE_FILE = "crypto_rotation_drill_monitor.json"
_RUN_CADENCE_S = 7 * 24 * 3600
_DEDUP_WINDOW_S = 30 * 86400
_DEFAULT_REVIEW_INTERVAL_DAYS = 730


def _enabled() -> bool:
    return os.getenv("CRYPTO_ROTATION_DRILL_MONITOR_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _review_interval_days() -> int:
    raw = os.getenv(
        "CRYPTO_ROTATION_REVIEW_INTERVAL_DAYS",
        str(_DEFAULT_REVIEW_INTERVAL_DAYS),
    ).strip()
    try:
        return max(30, int(raw))  # floor at a month
    except ValueError:
        return _DEFAULT_REVIEW_INTERVAL_DAYS


def _target_algorithm() -> str:
    return os.getenv("CRYPTO_ROTATION_TARGET_ALGORITHM", "sha3_256")


def run(
    *,
    manifest_path: Path | str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Single-pass probe. Returns structured summary."""
    summary: dict[str, Any] = {
        "ran": False,
        "n_pins": 0,
        "missing": [],
        "stale": [],
        "drill_ok": None,
        "drill_error": "",
        "alert_fired": False,
        "alert_tag": None,
    }
    if not _enabled():
        return summary

    cur = float(now) if now is not None else time.time()
    state = read_state_json(_STATE_FILE, {
        "last_run_at": 0.0,
        "last_alert_at": {},
    })
    if cur - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return summary
    state["last_run_at"] = cur
    summary["ran"] = True

    pins = list_pins(path=manifest_path)
    summary["n_pins"] = len(pins)

    missing = missing_artifact_classes(path=manifest_path)
    summary["missing"] = missing

    interval = _review_interval_days()
    cur_dt = datetime.fromtimestamp(cur, tz=__import__("datetime").timezone.utc)
    stale = stale_pins(
        interval_days=interval, path=manifest_path, now=cur_dt,
    )
    summary["stale"] = [
        {
            "artifact_class": p.artifact_class,
            "algorithm": p.algorithm,
            "pinned_at": p.pinned_at,
        }
        for p in stale
    ]

    # Run a small rotation drill. The default sample entries inside
    # run_rotation_drill cover the runtime check.
    target = _target_algorithm()
    drill = run_rotation_drill(
        "monitor_runtime_probe",
        target_algorithm=target,
    )
    summary["drill_ok"] = drill.ok
    summary["drill_error"] = drill.error if not drill.ok else ""

    def _maybe_alert(tag: str, body: str) -> None:
        last = state.setdefault("last_alert_at", {})
        if not isinstance(last, dict):
            last = {}
            state["last_alert_at"] = last
        if cur - float(last.get(tag, 0)) < _DEDUP_WINDOW_S:
            return
        try:
            send_signal_alert(body, tag=tag)
        except Exception:
            logger.debug("crypto_rotation_drill: signal alert failed", exc_info=True)
        last[tag] = cur
        summary["alert_fired"] = True
        summary["alert_tag"] = tag

    if missing:
        _maybe_alert(
            "crypto_rotation:missing_pins",
            f"⚠️ Crypto rotation: {len(missing)} artifact class(es) "
            f"have NO algorithm pin in `workspace/audit/"
            f"algorithm_pinning.json`. Run pin_algorithm() for each "
            f"of: {', '.join(missing)}",
        )
    if stale:
        names = ", ".join(p.artifact_class for p in stale)
        _maybe_alert(
            "crypto_rotation:stale_pins",
            f"⚠️ Crypto rotation: {len(stale)} pin(s) are older than "
            f"{interval} days and due for review: {names}. Re-confirm "
            f"that the chosen algorithm still meets the threat model "
            f"or rotate.",
        )
    if not drill.ok:
        _maybe_alert(
            "crypto_rotation:drill_failed",
            f"❌ Crypto rotation drill FAILED. Target algorithm: "
            f"`{target}`. Error: {drill.error}\n\n"
            f"This means the runtime can't compute hashes under the "
            f"target. Investigate before scheduling any rotation.",
        )

    audit_event(
        "crypto_rotation_drill_pass",
        n_pins=len(pins),
        missing_count=len(missing),
        stale_count=len(stale),
        drill_ok=drill.ok,
        drill_error=drill.error,
        target_algorithm=target,
        alert_fired=summary["alert_fired"],
    )

    write_state_json(_STATE_FILE, state)
    return summary
