"""feedback_loop_drift — concrete drift probe over the meta-agent loop.

PROGRAM §49 — Q14.2 (year-2+ resilience §10.2). The
:mod:`app.healing.influence_graph` package finds cycles in the
curated topology — observability. This monitor goes the next step:
it watches a specific named loop for the symptom the operator named
("optimize for the wrong thing"). The loop in scope:

    meta_agent.selector
       → agent_factory → crew_lifecycle
       → meta_agent.recorder → meta_agent.store
       → meta_agent.selector  (back to start)

Concrete probe — Gini coefficient of meta-agent recipe selection
rates over a rolling 30-day window. Gini = 0 means uniform
distribution (no recipe favoured); Gini = 1 means a single recipe
gets all selections. A *monotonic upward trend* in Gini over ≥30
days is evidence the closed loop is converging on a fixed point —
the system has settled on a strategy that may or may not be
globally optimal.

This is INFORMATIONAL, not corrective. The monitor never adjusts
recipe selection; the operator decides whether the convergence is
desirable. (Convergence on the BEST recipe = good; convergence on
a *locally* best recipe that the loop has reinforced through its
own feedback = bad.)

Algorithm:

  1. Read ``meta_agent_recipes`` recent ``uses`` counts.
  2. Compute Gini coefficient.
  3. Persist `(timestamp, gini)` to a history JSONL.
  4. Read the last 4 weekly samples; check for monotonic increase
     of magnitude ≥``_DELTA_THRESHOLD`` (default 0.10).
  5. Alert + emit ``feedback_loop_drift`` continuity-ledger event.

Cadence: weekly. Master switch: ``feedback_loop_drift_monitor_enabled``
(default ON).
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


NAME = "feedback_loop_drift"
CADENCE_SECONDS = 7 * 24 * 3600
MASTER_SWITCH_KEY = "feedback_loop_drift_monitor_enabled"

_DELTA_THRESHOLD = 0.10
_MIN_SAMPLES_FOR_TREND = 4
_DEDUP_WINDOW_S = 14 * 86400
_STATE_FILE_NAME = "feedback_loop_drift_state.json"
_HISTORY_FILE_NAME = "feedback_loop_drift_history.jsonl"


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_feedback_loop_drift_monitor_enabled
        return get_feedback_loop_drift_monitor_enabled()
    except Exception:
        return os.getenv(
            "FEEDBACK_LOOP_DRIFT_MONITOR_ENABLED", "true",
        ).lower() in ("true", "1", "yes", "on")


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _state_path() -> Path:
    return _workspace() / "healing" / _STATE_FILE_NAME


def _history_path() -> Path:
    return _workspace() / "healing" / _HISTORY_FILE_NAME


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"last_run_at": 0.0, "last_alert_at": 0.0}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"last_run_at": 0.0, "last_alert_at": 0.0}


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        logger.debug("feedback_loop_drift: state write failed", exc_info=True)


def gini(values: list[float]) -> float:
    """Standard Gini coefficient. Returns 0.0 if input is empty
    or sum is zero."""
    if not values:
        return 0.0
    vs = sorted(v for v in values if v >= 0)
    n = len(vs)
    if n == 0:
        return 0.0
    s = sum(vs)
    if s <= 0:
        return 0.0
    cum = 0.0
    for i, v in enumerate(vs, start=1):
        cum += i * v
    return (2 * cum) / (n * s) - (n + 1) / n


def _recipe_uses_distribution() -> list[float]:
    """Return list of ``uses`` counts for active meta-agent recipes."""
    try:
        from app.self_improvement.meta_agent.store import list_recipes
        recipes = list_recipes(limit=500, include_null=False)
        return [float(r.uses) for r in recipes if r.uses > 0]
    except Exception:
        logger.debug(
            "feedback_loop_drift: meta_agent store unavailable",
            exc_info=True,
        )
        return []


def _read_recent_history(n: int = 8) -> list[dict[str, Any]]:
    """Read the last N entries from the history JSONL."""
    p = _history_path()
    if not p.exists():
        return []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _append_history(row: dict[str, Any]) -> None:
    try:
        from app.utils.jsonl_retention import append_with_cap
        append_with_cap(
            _history_path(),
            json.dumps(row, sort_keys=True),
            500,
        )
    except Exception:
        logger.debug(
            "feedback_loop_drift: history append failed", exc_info=True,
        )


def _is_monotonic_increase(history: list[dict[str, Any]]) -> bool:
    """Are the last ``_MIN_SAMPLES_FOR_TREND`` ginis monotonically
    increasing AND has the total delta >= ``_DELTA_THRESHOLD``?"""
    if len(history) < _MIN_SAMPLES_FOR_TREND:
        return False
    recent = history[-_MIN_SAMPLES_FOR_TREND:]
    ginis = [float(r.get("gini", 0)) for r in recent]
    for a, b in zip(ginis, ginis[1:]):
        if b < a:
            return False
    delta = ginis[-1] - ginis[0]
    return delta >= _DELTA_THRESHOLD


def run(*, now: Optional[float] = None) -> dict[str, Any]:
    """One weekly pass. Returns summary dict.

    Failure-isolated: if meta_agent unavailable, skip with empty
    summary; no alerts on infrastructure failures."""
    summary: dict[str, Any] = {
        "ran": False,
        "gini": 0.0,
        "n_recipes": 0,
        "monotonic_increase": False,
        "alert_fired": False,
        "ledger_emitted": False,
    }
    if not _enabled():
        summary["skipped"] = True
        return summary

    cur = float(now) if now is not None else time.time()
    state = _read_state()
    last_run = float(state.get("last_run_at", 0))
    if last_run > 0 and cur - last_run < CADENCE_SECONDS:
        return summary
    state["last_run_at"] = cur
    summary["ran"] = True

    uses = _recipe_uses_distribution()
    summary["n_recipes"] = len(uses)
    if len(uses) < 3:
        # Too few recipes for Gini to mean anything.
        _write_state(state)
        return summary

    g = gini(uses)
    summary["gini"] = round(g, 4)
    _append_history({
        "ts": cur,
        "iso": datetime.now(timezone.utc).isoformat(),
        "gini": round(g, 4),
        "n_recipes": len(uses),
    })

    history = _read_recent_history(n=8)
    summary["monotonic_increase"] = _is_monotonic_increase(history)

    if summary["monotonic_increase"]:
        # Dedup alerts.
        if cur - float(state.get("last_alert_at", 0)) >= _DEDUP_WINDOW_S:
            try:
                from app.notify import notify
                ginis = [r["gini"] for r in history[-_MIN_SAMPLES_FOR_TREND:]]
                body = (
                    f"🔁 Feedback-loop drift: meta-agent recipe selection "
                    f"is converging on a fixed point.\n\n"
                    f"Gini coefficient over last "
                    f"{_MIN_SAMPLES_FOR_TREND} weekly samples: "
                    f"{ginis[0]:.2f} → {ginis[-1]:.2f} "
                    f"(monotonic increase, Δ={ginis[-1] - ginis[0]:+.2f}).\n\n"
                    f"This means the meta-agent → recipe → outcome → "
                    f"lessons KB → selection loop is concentrating on a "
                    f"shrinking subset of recipes. May be: (a) genuinely "
                    f"learning the best strategy, or (b) reinforcing a "
                    f"locally-optimal pattern that drifts from user "
                    f"value. Review at /cp/meta-agent."
                )
                notify(
                    title="🔁 Feedback-loop drift (meta-agent loop)",
                    body=body,
                    url="/cp/meta-agent",
                    topic="feedback_loop_drift",
                    critical=False,
                    arbitrate=True,
                )
                summary["alert_fired"] = True
                state["last_alert_at"] = cur
            except Exception:
                logger.debug(
                    "feedback_loop_drift: notify failed", exc_info=True,
                )

        # Continuity-ledger emission — landmark only.
        try:
            from app.identity.continuity_ledger import record_event
            record_event(
                kind="feedback_loop_drift",
                actor="feedback_loop_drift_monitor",
                summary=(
                    f"Meta-agent recipe-selection Gini increasing "
                    f"monotonically over {_MIN_SAMPLES_FOR_TREND} "
                    f"weekly samples (latest={g:.2f})."
                ),
                detail={
                    "loop": "meta_agent_selector_loop",
                    "gini_now": g,
                    "gini_history": [r["gini"] for r in history],
                    "n_recipes": len(uses),
                },
            )
            summary["ledger_emitted"] = True
        except Exception:
            logger.debug(
                "feedback_loop_drift: ledger emit failed", exc_info=True,
            )

    _write_state(state)
    return summary
