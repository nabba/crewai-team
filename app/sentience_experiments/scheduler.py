"""LIGHT idle-job entry points for the four Q5 modules.

PROGRAM §43.2 — Q5.2. Cadence-guarded internally; each module is its
own no-op when its master switch is OFF.

Cadences (informational — actual scheduling is in companion.loop):

  * ae2_causal_credit   → daily   (LIGHT — JSONL scan)
  * hot1_meta_affect    → daily   (LIGHT — small log + optional template)
  * hot4_metacog_monitor→ daily   (LIGHT — telemetry summary)
  * rpt1_self_calibration→ hourly (LIGHT — reconciler) + daily aggregator

Each entry returns a small summary dict that companion.loop logs.
Failure-isolated: any exception falls back to ``{"ok": False, ...}``.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def run_ae2() -> dict[str, Any]:
    """Daily idle entry for AE-2 rare-event causal credit assignment."""
    try:
        from app.sentience_experiments.ae2_causal_credit import run as _run
        return _run()
    except Exception:
        logger.debug("sentience.scheduler: ae2 failed", exc_info=True)
        return {"ok": False, "error": "ae2_exception"}


def run_hot1() -> dict[str, Any]:
    """Daily idle entry for HOT-1 meta-affect."""
    try:
        from app.sentience_experiments.hot1_meta_affect import run as _run
        return _run()
    except Exception:
        logger.debug("sentience.scheduler: hot1 failed", exc_info=True)
        return {"ok": False, "error": "hot1_exception"}


def run_hot4() -> dict[str, Any]:
    """Daily idle entry for HOT-4 metacognitive monitor."""
    try:
        from app.sentience_experiments.hot4_metacog_monitor import run as _run
        return _run()
    except Exception:
        logger.debug("sentience.scheduler: hot4 failed", exc_info=True)
        return {"ok": False, "error": "hot4_exception"}


def run_rpt1() -> dict[str, Any]:
    """Hourly idle entry for RPT-1 reconciler + calibration aggregator."""
    try:
        from app.sentience_experiments.rpt1_self_calibration import run as _run
        return _run()
    except Exception:
        logger.debug("sentience.scheduler: rpt1 failed", exc_info=True)
        return {"ok": False, "error": "rpt1_exception"}


def get_idle_jobs() -> list[tuple[str, callable, str]]:
    """Idle-job manifest for companion.loop.

    Returns: list of (job_name, callable, cadence_label) tuples.
    Cadence labels are informational — companion.loop runs all
    LIGHT jobs at its own cadence (typically every few minutes,
    skipping if the prior pass completed recently).
    """
    return [
        ("sentience-ae2", run_ae2, "LIGHT-daily"),
        ("sentience-hot1", run_hot1, "LIGHT-daily"),
        ("sentience-hot4", run_hot4, "LIGHT-daily"),
        ("sentience-rpt1", run_rpt1, "LIGHT-hourly"),
    ]
