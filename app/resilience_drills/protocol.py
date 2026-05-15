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


@dataclass
class DrillResult:
    """Outcome of one drill run. Persisted to the audit JSONL."""

    drill_name: str
    status: DrillStatus
    started_at: str             # ISO-8601
    completed_at: str           # ISO-8601
    duration_s: float
    dry_run: bool
    detail: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "drill_name": self.drill_name,
            "status": self.status.value if isinstance(self.status, DrillStatus) else str(self.status),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": float(self.duration_s),
            "dry_run": bool(self.dry_run),
            "detail": dict(self.detail or {}),
            "errors": list(self.errors or []),
        }


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
        runner reference, useful for hot-reload during tests."""
        with self._lock:
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
