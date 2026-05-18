"""Drill protocol: DrillSpec / DrillResult / DrillRegistry.

PROGRAM §44.1 — Q6.1 foundation. Defines the contract every drill
implements. Each drill is registered once at import-time; the
registry is the single source of truth for "what drills exist."

Drill lifecycle
---------------

1. **Registration**: each drill module defines a ``SPEC`` and a
   ``run(*, dry_run: bool = True) -> DrillResult`` callable, then
   calls ``register(SPEC, run)`` at module load.
2. **Scheduling**: ``scheduler.py`` reads the registry, checks the
   audit log for last-run timestamps, emits "due" notifications.
3. **Execution**: ``DrillRegistry.execute(name, dry_run=...)``
   invokes the runner and writes the result to the audit log.
4. **Audit + emission**: result lands in ``drill_audit.jsonl``;
   landmark events (failures, first-ever runs) emit to the
   continuity ledger as ``resilience_drill`` events.

Risk levels
-----------

* ``LOW``    — safe to run any time, never disrupts production
* ``MEDIUM`` — runs use real resources (API calls, DB scans) but
                doesn't disrupt; should respect cadence
* ``HIGH``   — disruptive; requires operator typed-phrase confirmation
                AND explicit master-switch ON (kill_the_gateway)
"""
from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class DrillRisk(str, Enum):
    """Risk classification — see module docstring."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DrillStatus(str, Enum):
    """Outcome of a single drill run."""
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"        # master switch off, gate not satisfied, etc.
    ERROR = "error"            # exception during run; treat as worse than FAIL


class FailureClass(str, Enum):
    """Q18 (PROGRAM §57) — how a drill failed.

    The scheduler's response depends on the class:

    * ``CODE_ERROR``          — uncaught exception in drill code. Three
      consecutive code errors send the drill to ``QUARANTINED`` and the
      scheduler refuses to auto-run it until the operator unquarantines.
    * ``STRUCTURAL_FAIL``     — drill produced a structured FAIL with
      stable findings (e.g. ``cascade_diversity: only 1 fallback``). The
      finding will not change between rapid retries; apply standard
      WATCH/DEGRADED backoff. After the operator ratifies a baseline,
      these may become ``BASELINE_REGRESSION`` instead.
    * ``TRANSIENT_FAIL``      — failure looks timing/network-related and
      may recover quickly. Same escalation as STRUCTURAL_FAIL but the
      tighter WATCH backoff is more appropriate.
    * ``BASELINE_REGRESSION`` — observation drifted from the operator-
      ratified baseline. The actionable signal — alert the operator
      with the per-key regression diff.
    """
    CODE_ERROR = "code_error"
    STRUCTURAL_FAIL = "structural_fail"
    TRANSIENT_FAIL = "transient_fail"
    BASELINE_REGRESSION = "baseline_regression"


@dataclass(frozen=True)
class DrillSpec:
    """Static declaration of a drill. Registered once at import."""

    name: str                   # unique identifier, e.g. "backup_restore"
    cadence_days: int           # ideal interval; >cadence + grace → stale
    grace_days: int = 30        # grace period before "stale" alerts fire
    risk: DrillRisk = DrillRisk.LOW
    description: str = ""       # one-line human-readable
    requires_typed_phrase: str | None = None  # confirmation string for HIGH-risk
    requires_master_switch: str | None = None  # runtime_settings flag (e.g. drill_kill_the_gateway_enabled)
    # Q18 (PROGRAM §57) — new drills land in WARMING_UP for this many
    # days. During warmup the drill runs and emits observations but
    # doesn't fire alerts; the operator ratifies a baseline before
    # active monitoring kicks in. Existing drills may pass 0 to skip
    # warmup (back-compat for the §44 ones already in production).
    warmup_days: int = 7


@dataclass
class DrillResult:
    """Outcome of one drill run. Persisted to the audit JSONL.

    Q18 additions: ``failure_class`` (informs scheduler escalation
    policy) and ``observation`` (the structured measurements
    baseline-comparison reads). Both are optional so the §44 drills
    can be converted one at a time.
    """

    drill_name: str
    status: DrillStatus
    started_at: str             # ISO-8601
    completed_at: str           # ISO-8601
    duration_s: float
    dry_run: bool
    detail: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    # Q18 — set by the runner on FAIL/ERROR; the scheduler uses this
    # to pick state-machine transitions. None on PASS/SKIPPED.
    failure_class: FailureClass | None = None
    # Q18 — structured measurements (compared against baseline by the
    # scheduler). The drill's main side-channel for the operator —
    # what was actually observed during this run.
    observation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "drill_name": self.drill_name,
            "status": self.status.value if isinstance(self.status, DrillStatus) else str(self.status),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": float(self.duration_s),
            "dry_run": bool(self.dry_run),
            "detail": dict(self.detail or {}),
            "errors": list(self.errors or []),
        }
        if self.failure_class is not None:
            out["failure_class"] = (
                self.failure_class.value
                if isinstance(self.failure_class, FailureClass)
                else str(self.failure_class)
            )
        if self.observation is not None:
            out["observation"] = dict(self.observation)
        return out


# ── Registry ──────────────────────────────────────────────────────────────


# A drill runner takes ``dry_run: bool`` and returns a DrillResult.
DrillRunner = Callable[..., DrillResult]


class DrillRegistry:
    """Single source of truth for known drills.

    Modules register at import; the scheduler and the REST surfaces
    consult the registry to enumerate. The registry is a singleton
    accessible via ``get_registry()`` so isolated-loads in tests
    don't accidentally fragment state."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._specs: dict[str, DrillSpec] = {}
        self._runners: dict[str, DrillRunner] = {}

    def register(self, spec: DrillSpec, runner: DrillRunner) -> None:
        """Register a drill. Idempotent — re-registration updates the
        runner reference, useful for hot-reload during tests.

        Q6.5 P2#2 — warn when a different module re-registers the
        same name with a DIFFERENT runner object. Same-runner is the
        hot-reload case (legitimate); different-runner suggests an
        accidental name collision between two drill modules that
        should be investigated."""
        with self._lock:
            existing_runner = self._runners.get(spec.name)
            if (
                existing_runner is not None
                and existing_runner is not runner
                and getattr(existing_runner, "__module__", None)
                    != getattr(runner, "__module__", None)
            ):
                # Different module + different runner → likely a name
                # collision. Don't refuse (operator may have intent)
                # but log loudly so it surfaces in operator review.
                logger.warning(
                    "DrillRegistry: drill name %r being re-registered "
                    "from different module (was %r, now %r); silent "
                    "overwrite may indicate accidental name collision",
                    spec.name,
                    getattr(existing_runner, "__module__", "<unknown>"),
                    getattr(runner, "__module__", "<unknown>"),
                )
            self._specs[spec.name] = spec
            self._runners[spec.name] = runner

    def get(self, name: str) -> DrillSpec | None:
        with self._lock:
            return self._specs.get(name)

    def list_specs(self) -> list[DrillSpec]:
        with self._lock:
            return list(self._specs.values())

    def runner_for(self, name: str) -> DrillRunner | None:
        with self._lock:
            return self._runners.get(name)

    def clear_for_tests(self) -> None:
        """Test-only — drop all registrations."""
        with self._lock:
            self._specs.clear()
            self._runners.clear()


_registry: DrillRegistry | None = None


def get_registry() -> DrillRegistry:
    """Lazy-singleton accessor."""
    global _registry
    if _registry is None:
        _registry = DrillRegistry()
    return _registry


def register(spec: DrillSpec, runner: DrillRunner) -> None:
    """Module-level convenience for registration at import time."""
    get_registry().register(spec, runner)


# ── Master-switch readers ────────────────────────────────────────────────


def master_enabled() -> bool:
    """Top-level Q6 switch. When OFF, all drills are SKIPPED."""
    try:
        from app.runtime_settings import get_resilience_drills_enabled
        return get_resilience_drills_enabled()
    except Exception:
        return True


def drill_enabled(spec: DrillSpec) -> bool:
    """Combined check: master + per-drill flag (if specified)."""
    if not master_enabled():
        return False
    if not spec.requires_master_switch:
        return True
    try:
        from app import runtime_settings
        getter_name = f"get_{spec.requires_master_switch}"
        getter = getattr(runtime_settings, getter_name, None)
        if getter is None:
            return False
        return bool(getter())
    except Exception:
        return False
