"""healing/runbooks.py — anomaly-triggered runbook dispatcher.

Listens (via ``maybe_run_runbook``) for anomaly rows produced by
``app/observability/error_monitor.py:_record_anomaly``. When a registered
runbook's pattern matches the anomaly's signature AND seven safety
gates pass, dispatches the runbook handler in a daemon thread.

Off by default — set ``ERROR_RUNBOOKS_ENABLED=true`` to opt in.

Composes with the rest of the healing stack:

  * ``app/recovery/loop.py``        — refusal-shaped final answers
  * ``app/tool_runtime/supervisor.py`` — mid-iteration tool exceptions
  * ``app/healing/error_diagnosis.py`` — per-exception remediation proposal
  * ``app/healing/health_remediator.py`` — aggregate-health alerts
  * (this module)                   — pattern-aggregate anomaly remediation

Audit trail: ``actor='self_heal_runbook'``, actions ``dispatch.started
| dispatch.finished | dispatch.skipped``. Query via
``/api/cp/audit?actor=self_heal_runbook``.

Operator constraint: runbook handlers are operator-authored and must
NOT modify any path in ``app/auto_deployer.TIER_IMMUTABLE``. The
dispatcher itself makes no LLM calls — pure Python — keeping
remediation cheap, deterministic, and auditable.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Pattern

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────────────

_RUNBOOK_DIR = Path(__file__).resolve().parents[2] / "workspace" / "self_heal"
_SETTINGS_PATH = _RUNBOOK_DIR / "runbook_settings.json"
_STATS_PATH = _RUNBOOK_DIR / "runbook_stats.json"

# Last N outcomes per runbook used for the success-rate gate.
_RECENT_HISTORY_CAP = 10

# Success-rate gate threshold: drop the runbook if recent rate < this.
_MIN_SUCCESS_RATE = 0.5

# Concurrency cap — at most one runbook dispatch in flight at a time.
_MAX_CONCURRENT_DISPATCHES = 1

# Recurrence window — anomaly's signature must have been seen at least
# `min_recurrence` times in the last 24h before we'll dispatch.
_RECURRENCE_WINDOW_HOURS = 24


# ── Master switch ──────────────────────────────────────────────────────────


def runbooks_enabled() -> bool:
    """Master switch. Default off, per the Recovery Loop precedent."""
    return os.getenv("ERROR_RUNBOOKS_ENABLED", "false").lower() in (
        "true", "1", "yes",
    )


# ── Registry ───────────────────────────────────────────────────────────────


RunbookHandler = Callable[[dict[str, Any]], "RunbookResult"]


@dataclass
class _RunbookEntry:
    name: str
    pattern: Pattern[str]
    handler: RunbookHandler


@dataclass
class RunbookResult:
    """Outcome of a single runbook dispatch."""
    name: str
    success: bool
    detail: str = ""
    duration_ms: int = 0
    error: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


# Insertion-ordered registry. First match wins on dispatch.
_REGISTERED_RUNBOOKS: dict[str, _RunbookEntry] = {}
_registry_lock = threading.Lock()

# Concurrency tracking.
_active_runbooks: set[str] = set()
_active_lock = threading.Lock()


def register_runbook(
    name: str, pattern: str | Pattern[str], handler: RunbookHandler,
) -> None:
    """Register a runbook handler under ``name``.

    ``pattern`` is matched against the anomaly's ``pattern_signature``.
    Accepts either a raw regex string or an already-compiled
    ``re.Pattern``. Re-registration of the same name replaces the
    previous entry (useful for tests and operator hot-edits).
    """
    if isinstance(pattern, str):
        compiled = re.compile(pattern)
    else:
        compiled = pattern
    with _registry_lock:
        _REGISTERED_RUNBOOKS[name] = _RunbookEntry(
            name=name, pattern=compiled, handler=handler,
        )


def unregister_runbook(name: str) -> None:
    """Remove a runbook from the registry. No-op if not present."""
    with _registry_lock:
        _REGISTERED_RUNBOOKS.pop(name, None)


def _match_runbook(pattern_signature: str) -> Optional[_RunbookEntry]:
    """Return the first registered runbook whose pattern matches.

    First-registered-wins on collision so operator boot order is the
    precedence tie-breaker.
    """
    with _registry_lock:
        entries = list(_REGISTERED_RUNBOOKS.values())
    for entry in entries:
        if entry.pattern.search(pattern_signature):
            return entry
    return None


# ── Settings ───────────────────────────────────────────────────────────────


def _load_runbook_settings() -> dict[str, Any]:
    """Load per-runbook ``enabled`` flag + ``min_recurrence``.

    Missing file / malformed JSON ⇒ empty mapping (every runbook
    defaults to disabled, matching the safe-by-default principle).
    """
    try:
        with open(_SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("runbooks", {}) if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        logger.debug(
            "self_heal_runbook: settings load failed", exc_info=True,
        )
        return {}


# ── Stats (success-rate gate) ──────────────────────────────────────────────


def _load_runbook_stats() -> dict[str, Any]:
    """Load per-runbook recent outcomes for the success-rate gate."""
    try:
        with open(_STATS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("runbooks", {}) if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        logger.debug(
            "self_heal_runbook: stats load failed", exc_info=True,
        )
        return {}


def _save_runbook_stats(stats: dict[str, Any]) -> None:
    try:
        _RUNBOOK_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _STATS_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"runbooks": stats}, f, indent=2)
        tmp.replace(_STATS_PATH)
    except Exception:
        logger.debug(
            "self_heal_runbook: stats persist failed", exc_info=True,
        )


def _record_runbook_outcome(name: str, success: bool) -> None:
    """Append an outcome to the runbook's recent-history list (cap N)."""
    stats = _load_runbook_stats()
    entry = stats.setdefault(name, {"recent": []})
    recent = entry.setdefault("recent", [])
    recent.append({"success": bool(success), "ts": time.time()})
    if len(recent) > _RECENT_HISTORY_CAP:
        del recent[: len(recent) - _RECENT_HISTORY_CAP]
    _save_runbook_stats(stats)


def _runbook_success_rate(name: str) -> float:
    """Recent success rate. Empty history ⇒ 1.0 (treated as passing)."""
    stats = _load_runbook_stats()
    recent = stats.get(name, {}).get("recent", [])
    if not recent:
        return 1.0
    successes = sum(1 for r in recent if r.get("success"))
    return successes / len(recent)


# ── Recurrence (gate 6) ────────────────────────────────────────────────────


def _signature_recurrence(signature: str) -> int:
    """Count occurrences of ``signature`` in the last 24h.

    Reads the error_monitor's in-memory rolling window (cheap; no DB
    round-trip). Falls back to 0 if the monitor is uninitialised.
    """
    try:
        from app.observability.error_monitor import _hourly_rate  # type: ignore
        return _hourly_rate(signature, hours=_RECURRENCE_WINDOW_HOURS)
    except Exception:
        return 0


# ── Audit ──────────────────────────────────────────────────────────────────


def _runbook_audit(action: str, **detail: Any) -> None:
    """Best-effort audit. Never raises."""
    try:
        from app.control_plane.audit import get_audit
        get_audit().log(actor="self_heal_runbook", action=action, detail=detail)
    except Exception:
        logger.debug(
            "self_heal_runbook: audit write failed", exc_info=True,
        )


# ── Dispatch ───────────────────────────────────────────────────────────────


def maybe_run_runbook(anomaly: dict[str, Any]) -> Optional[RunbookResult]:
    """Maybe dispatch a runbook for ``anomaly``. Returns immediately.

    The actual handler runs in a daemon thread so monitor scans are
    not blocked. The returned ``RunbookResult`` (when non-None)
    indicates the dispatch was *initiated*; the handler outcome is
    persisted via ``_record_runbook_outcome`` and emitted as
    ``dispatch.finished`` audit event.

    Seven gates, each emits an explicit ``dispatch.skipped`` audit
    event with a ``reason`` field for traceability:

      1. Env flag                      — ``ERROR_RUNBOOKS_ENABLED``
      2. Severity                      — ``severity_info``
      3. Pattern match                 — ``no_pattern_match``
      4. Per-runbook enabled flag      — ``runbook_disabled``
      5. Recurrence ≥ ``min_recurrence`` — ``below_recurrence_threshold``
      6. Recent success rate ≥ 50%     — ``recent_success_rate_low``
      7. Concurrency cap (max 1)       — ``concurrency_cap``
    """
    sig = anomaly.get("pattern_signature") or anomaly.get("signature") or ""
    severity = anomaly.get("severity", "info")

    # Gate 1 — env flag
    if not runbooks_enabled():
        return None

    # Gate 2 — severity
    if severity == "info":
        _runbook_audit(
            "dispatch.skipped",
            reason="severity_info", pattern_signature=sig, severity=severity,
        )
        return None

    # Gate 3 — pattern match
    entry = _match_runbook(sig)
    if entry is None:
        _runbook_audit(
            "dispatch.skipped",
            reason="no_pattern_match", pattern_signature=sig,
        )
        return None

    # Gate 4 — per-runbook enabled
    settings = _load_runbook_settings()
    rb_settings = settings.get(entry.name, {})
    if not rb_settings.get("enabled", False):
        _runbook_audit(
            "dispatch.skipped",
            reason="runbook_disabled",
            runbook_name=entry.name, pattern_signature=sig,
        )
        return None

    # Gate 5 — recurrence
    min_recurrence = int(rb_settings.get("min_recurrence", 1))
    recurrence = _signature_recurrence(sig)
    if recurrence < min_recurrence:
        _runbook_audit(
            "dispatch.skipped",
            reason="below_recurrence_threshold",
            runbook_name=entry.name,
            pattern_signature=sig,
            recurrence=recurrence,
            min_recurrence=min_recurrence,
        )
        return None

    # Gate 6 — recent success rate
    success_rate = _runbook_success_rate(entry.name)
    if success_rate < _MIN_SUCCESS_RATE:
        _runbook_audit(
            "dispatch.skipped",
            reason="recent_success_rate_low",
            runbook_name=entry.name,
            pattern_signature=sig,
            success_rate=round(success_rate, 3),
        )
        return None

    # Gate 7 — concurrency cap
    with _active_lock:
        if len(_active_runbooks) >= _MAX_CONCURRENT_DISPATCHES:
            _runbook_audit(
                "dispatch.skipped",
                reason="concurrency_cap",
                runbook_name=entry.name,
                pattern_signature=sig,
                in_flight=sorted(_active_runbooks),
            )
            return None
        _active_runbooks.add(entry.name)

    _runbook_audit(
        "dispatch.started",
        runbook_name=entry.name,
        pattern_signature=sig,
        anomaly_type=anomaly.get("anomaly_type"),
        severity=severity,
    )

    thread = threading.Thread(
        target=_run_handler,
        args=(entry, anomaly),
        name=f"runbook-{entry.name}",
        daemon=True,
    )
    thread.start()

    return RunbookResult(name=entry.name, success=True, detail="dispatched")


def _run_handler(entry: _RunbookEntry, anomaly: dict[str, Any]) -> None:
    """Invoke the handler with bounded scope and persist the outcome."""
    started = time.monotonic()
    try:
        result = entry.handler(anomaly)
        if not isinstance(result, RunbookResult):
            result = RunbookResult(
                name=entry.name,
                success=bool(result),
                detail="non-RunbookResult return coerced",
            )
        result.duration_ms = int((time.monotonic() - started) * 1000)
        _record_runbook_outcome(entry.name, result.success)
        _runbook_audit(
            "dispatch.finished",
            runbook_name=entry.name,
            success=result.success,
            duration_ms=result.duration_ms,
            detail=result.detail,
            error=result.error,
        )
    except Exception as exc:
        duration_ms = int((time.monotonic() - started) * 1000)
        _record_runbook_outcome(entry.name, False)
        _runbook_audit(
            "dispatch.finished",
            runbook_name=entry.name,
            success=False,
            duration_ms=duration_ms,
            error=f"{type(exc).__name__}: {str(exc)[:300]}",
        )
        logger.debug(
            "self_heal_runbook: handler raised", exc_info=True,
        )
    finally:
        with _active_lock:
            _active_runbooks.discard(entry.name)


# ── Reference handler ──────────────────────────────────────────────────────


def _runbook_log_only(anomaly: dict[str, Any]) -> RunbookResult:
    """Reference handler — logs, takes no system action.

    Auto-registered with the catch-all ``.*`` pattern. Verifies the
    dispatch wiring end-to-end without changing system state.
    Operators replace or narrow it once real runbooks are registered
    from boot code.
    """
    sig = anomaly.get("pattern_signature") or anomaly.get("signature") or ""
    sev = anomaly.get("severity", "info")
    logger.info(
        "self_heal_runbook[log_only]: triggered — sig=%s severity=%s type=%s",
        sig, sev, anomaly.get("anomaly_type"),
    )
    return RunbookResult(
        name="log_only",
        success=True,
        detail=f"logged sig={sig} severity={sev}",
    )


# Register the reference handler at import time.
register_runbook("log_only", r".*", _runbook_log_only)
