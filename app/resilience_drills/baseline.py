"""Drill baselines — operator-ratified expected observations.

PROGRAM §57 — Q18 resilience-drill v2. Closes the "drill threshold
mismatches deployment" failure mode from the 2026-05-16 incident.

The model
=========

A drill is an *observer*. Each run produces a structured
:class:`Observation` — a dict of measurements + a tagging
``summary``. Pass/fail in the legacy sense is no longer the drill's
output; it's a *comparison result* between the latest observation
and the operator-ratified :class:`Baseline`.

This is the same pattern already used by
:mod:`app.healing.monitors.embedding_drift` (anchor queries vs
baseline) and
:mod:`app.epistemic.calibration` (Brier drift vs baseline). Q18
generalises it to resilience drills, which were previously baking
posture opinions into hardcoded thresholds (``≥50% providers
ready``, ``≥2 fallbacks``, …) invisible to the operator.

Ratification flow
-----------------

1. Drill ships in ``WARMING_UP`` state. It runs on cadence and
   accumulates observations, but emits no alerts.
2. Operator visits ``/cp/drills/<name>``, reviews recent
   observations, clicks **Ratify baseline** — picks one observation
   as the expected/acceptable state plus optional tolerances.
3. Future observations compare against the baseline. Drift beyond
   tolerance → ``BASELINE_REGRESSION`` failure → standard scheduler
   escalation (WATCH → DEGRADED → backoff).

Tolerance grammar
-----------------

Per-key rules. Default rule is ``exact``. Supported:

* ``{"rule": "exact"}`` — values must match.
* ``{"rule": "min", "value": N}`` — observation key must be ≥ N.
* ``{"rule": "max", "value": N}`` — observation key must be ≤ N.
* ``{"rule": "range", "min": A, "max": B}`` — observation key ∈ [A, B].
* ``{"rule": "subset_of", "value": [...]}`` — observation list must
  be a subset of the given list (gained members are violations).
* ``{"rule": "superset_of", "value": [...]}`` — observation list must
  include every member (lost members are violations).

Storage
-------

One JSON document per drill at
``workspace/resilience/drill_baselines/<name>.json``. Per-drill matches
the per-state choice — easier to diff in git, easier to ratify one at
a time.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Data model ───────────────────────────────────────────────────────────


@dataclass
class Observation:
    """Structured measurements a drill emits — pass/fail-agnostic.

    The drill code is free to put any JSON-serializable values into
    ``measurements``. ``summary`` is an optional one-line label that
    surfaces in operator views ("1 fallback configured", "5/6
    providers reachable", etc.).
    """

    drill_name: str
    observed_at: str            # ISO-8601 UTC
    measurements: dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "drill_name": self.drill_name,
            "observed_at": self.observed_at,
            "measurements": dict(self.measurements or {}),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Observation":
        return cls(
            drill_name=str(data.get("drill_name") or ""),
            observed_at=str(data.get("observed_at") or ""),
            measurements=dict(data.get("measurements") or {}),
            summary=str(data.get("summary") or ""),
        )


@dataclass
class Baseline:
    """Operator-ratified expected observation + tolerances.

    Set by the operator (via REST or React) from one of the drill's
    accumulated observations. The drill is HEALTHY when every
    measurement matches the baseline subject to its tolerance rule.
    """

    drill_name: str
    ratified_at: str            # ISO-8601 UTC when operator ratified
    ratified_by: str            # "operator-react" | "operator-cli" | …
    measurements: dict[str, Any] = field(default_factory=dict)
    tolerances: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Baseline":
        return cls(
            drill_name=str(data.get("drill_name") or ""),
            ratified_at=str(data.get("ratified_at") or ""),
            ratified_by=str(data.get("ratified_by") or ""),
            measurements=dict(data.get("measurements") or {}),
            tolerances=dict(data.get("tolerances") or {}),
            notes=str(data.get("notes") or ""),
        )


@dataclass
class Regression:
    """One key's deviation from baseline."""

    key: str
    rule: str                   # the tolerance rule that fired
    expected: Any
    observed: Any
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonResult:
    """Outcome of comparing an Observation against a Baseline."""

    ok: bool
    regressions: list[Regression] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)   # baseline keys absent in observation

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "regressions": [r.to_dict() for r in self.regressions],
            "missing_keys": list(self.missing_keys),
        }


# ── Path resolution ──────────────────────────────────────────────────────


_lock = threading.RLock()


def _baseline_dir() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "resilience" / "drill_baselines"
    except Exception:
        return Path("/app/workspace/resilience/drill_baselines")


def _baseline_path(drill_name: str) -> Path:
    return _baseline_dir() / f"{drill_name}.json"


# ── Persistence ──────────────────────────────────────────────────────────


def load(drill_name: str) -> Baseline | None:
    """Read the ratified baseline for a drill, or None when not yet
    ratified (drill is still in WARMING_UP)."""
    path = _baseline_path(drill_name)
    if not path.exists():
        return None
    try:
        with _lock:
            data = json.loads(path.read_text(encoding="utf-8"))
        return Baseline.from_dict(data)
    except Exception:
        logger.warning("baseline: cannot read %s", path, exc_info=True)
        return None


def save(baseline: Baseline) -> bool:
    """Atomically persist a Baseline. Returns True on success."""
    path = _baseline_path(baseline.drill_name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            body = json.dumps(baseline.to_dict(), indent=2, sort_keys=True, default=str)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8",
                dir=str(path.parent),
                prefix=f".{baseline.drill_name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp.write(body)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_name = tmp.name
            os.replace(tmp_name, path)
        return True
    except OSError:
        logger.warning("baseline: persist failed for %s",
                       baseline.drill_name, exc_info=True)
        return False


def list_all_baselines() -> list[Baseline]:
    """Enumerate every ratified baseline."""
    out: list[Baseline] = []
    bd = _baseline_dir()
    if not bd.exists():
        return out
    for p in sorted(bd.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            out.append(Baseline.from_dict(data))
        except Exception:
            logger.warning("baseline: skipping unreadable %s", p, exc_info=True)
    return out


# ── Comparison ───────────────────────────────────────────────────────────


def compare(observation: Observation, baseline: Baseline) -> ComparisonResult:
    """Compare an Observation against a Baseline.

    Walks every key in ``baseline.measurements``; for each one,
    applies the tolerance rule (default ``exact``) from
    ``baseline.tolerances`` and records a Regression on mismatch.

    Keys present in the observation but absent in the baseline are
    NOT regressions — operators may add new measurement keys that
    didn't exist when the baseline was ratified, and silently
    failing on those would be hostile to evolution.

    Keys present in the baseline but absent in the observation ARE
    flagged in ``missing_keys`` — the drill code probably stopped
    emitting them, which is a regression of a different sort.
    """
    regressions: list[Regression] = []
    missing: list[str] = []
    for key, expected in (baseline.measurements or {}).items():
        if key not in (observation.measurements or {}):
            missing.append(key)
            continue
        observed = observation.measurements[key]
        rule_spec = (baseline.tolerances or {}).get(key) or {"rule": "exact"}
        rule = rule_spec.get("rule", "exact")
        regression = _check_rule(key, rule, rule_spec, expected, observed)
        if regression is not None:
            regressions.append(regression)
    return ComparisonResult(
        ok=(not regressions and not missing),
        regressions=regressions,
        missing_keys=missing,
    )


def _check_rule(key: str, rule: str, rule_spec: dict[str, Any],
                expected: Any, observed: Any) -> Regression | None:
    """Apply one tolerance rule. Returns a Regression on mismatch
    or None on pass."""
    try:
        if rule == "exact":
            if observed != expected:
                return Regression(
                    key=key, rule=rule,
                    expected=expected, observed=observed,
                    detail="exact-match required",
                )
        elif rule == "min":
            threshold = rule_spec.get("value", expected)
            if not isinstance(observed, (int, float)):
                return Regression(
                    key=key, rule=rule,
                    expected=threshold, observed=observed,
                    detail="non-numeric observed value",
                )
            if observed < threshold:
                return Regression(
                    key=key, rule=rule,
                    expected=threshold, observed=observed,
                    detail=f"observed {observed} < min {threshold}",
                )
        elif rule == "max":
            threshold = rule_spec.get("value", expected)
            if not isinstance(observed, (int, float)):
                return Regression(
                    key=key, rule=rule,
                    expected=threshold, observed=observed,
                    detail="non-numeric observed value",
                )
            if observed > threshold:
                return Regression(
                    key=key, rule=rule,
                    expected=threshold, observed=observed,
                    detail=f"observed {observed} > max {threshold}",
                )
        elif rule == "range":
            lo = rule_spec.get("min")
            hi = rule_spec.get("max")
            if lo is None or hi is None or not isinstance(observed, (int, float)):
                return Regression(
                    key=key, rule=rule,
                    expected={"min": lo, "max": hi}, observed=observed,
                    detail="range bounds missing or non-numeric value",
                )
            if observed < lo or observed > hi:
                return Regression(
                    key=key, rule=rule,
                    expected={"min": lo, "max": hi}, observed=observed,
                    detail=f"observed {observed} outside [{lo}, {hi}]",
                )
        elif rule == "subset_of":
            allowed = rule_spec.get("value", expected)
            if not isinstance(observed, list) or not isinstance(allowed, list):
                return Regression(
                    key=key, rule=rule,
                    expected=allowed, observed=observed,
                    detail="non-list observed or rule value",
                )
            extras = [v for v in observed if v not in allowed]
            if extras:
                return Regression(
                    key=key, rule=rule,
                    expected=allowed, observed=observed,
                    detail=f"gained members: {extras[:5]}",
                )
        elif rule == "superset_of":
            required = rule_spec.get("value", expected)
            if not isinstance(observed, list) or not isinstance(required, list):
                return Regression(
                    key=key, rule=rule,
                    expected=required, observed=observed,
                    detail="non-list observed or rule value",
                )
            missing = [v for v in required if v not in observed]
            if missing:
                return Regression(
                    key=key, rule=rule,
                    expected=required, observed=observed,
                    detail=f"lost members: {missing[:5]}",
                )
        else:
            # Unknown rule — fail loudly so it surfaces in audit.
            return Regression(
                key=key, rule=rule,
                expected=expected, observed=observed,
                detail=f"unknown tolerance rule {rule!r}",
            )
    except Exception as exc:
        logger.debug("baseline: rule %s raised on %s", rule, key, exc_info=True)
        return Regression(
            key=key, rule=rule,
            expected=expected, observed=observed,
            detail=f"comparison raised: {type(exc).__name__}: {exc}",
        )
    return None


# ── Ratification ─────────────────────────────────────────────────────────


def ratify_from_observation(observation: Observation, *, operator: str,
                            tolerances: dict[str, dict[str, Any]] | None = None,
                            notes: str = "") -> Baseline:
    """Operator-facing helper: turn an Observation into a Baseline.

    The observation's measurements become the baseline; the caller
    supplies optional per-key tolerance rules. Persists immediately.
    """
    baseline = Baseline(
        drill_name=observation.drill_name,
        ratified_at=datetime.now(timezone.utc).isoformat(),
        ratified_by=operator,
        measurements=dict(observation.measurements or {}),
        tolerances=dict(tolerances or {}),
        notes=notes[:500],
    )
    save(baseline)
    return baseline


def reset_for_tests() -> None:
    """Test-only — remove all baseline files."""
    bd = _baseline_dir()
    if not bd.exists():
        return
    for p in bd.glob("*.json"):
        try:
            p.unlink()
        except OSError:
            pass
    for p in bd.glob(".*.tmp"):
        try:
            p.unlink()
        except OSError:
            pass
