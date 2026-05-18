"""Drill state machine — per-drill persistent state.

PROGRAM §57 — Q18 resilience-drill v2 foundation. Replaces the
"scan audit log every pass" model from §44 with an explicit state
machine. A drill that fails enters DEGRADED with exponential
backoff; a drill whose code raises uncaught (CODE_ERROR) enters
QUARANTINED after 3 such failures. The scheduler reads state and
respects ``next_attempt_after`` — the hot-loop pattern that
produced the 2026-05-16 incident (3580 drill rows in 48h) is
architecturally impossible under this model.

State machine
=============

::

    WARMING_UP ── (warmup_days elapse) ──► HEALTHY ◄──────► WATCH
                                              │              │
                                              ▼              ▼
                                          DEGRADED ◄────────┘
                                              │
                                              ▼
                                         QUARANTINED  (manual unlock)

    Any state ─► MUTED (operator-set; reversible)

Transition rules
----------------

* ``WARMING_UP``: newly-registered drill. Observations are recorded
  but failures don't fire alerts and don't auto-escalate. Operator
  ratifies a baseline before the drill moves to active monitoring.
* ``HEALTHY``: passed last run; on regular cadence. ``next_attempt
  _after = last_success_at + cadence_days``.
* ``WATCH``: failed once. Short retry window (15 min, exponential
  to 1 h). A pass returns it to HEALTHY; another fail demotes to
  DEGRADED.
* ``DEGRADED``: 2+ consecutive structural failures. Backoff schedule
  1h → 2h → 4h → 8h ... capped at ``cadence_days``. A pass returns
  it to HEALTHY.
* ``QUARANTINED``: 3+ consecutive ``CODE_ERROR`` outcomes (uncaught
  exception suggests drill code bug). Scheduler will NOT auto-run a
  quarantined drill regardless of master switch — operator must
  explicitly call ``unquarantine()`` after fixing.
* ``MUTED``: operator-silenced. No auto-runs, no alerts. Reversible
  via ``unmute()``.

Persistence
-----------

Per-drill JSON at ``workspace/resilience/drill_state/<name>.json``.
Per-drill matches the per-drill baseline storage choice — easier to
diff in git, easier to ratify one at a time. Atomic writes via
``tempfile + replace``.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── State enum ───────────────────────────────────────────────────────────


class DrillState(str, Enum):
    WARMING_UP = "warming_up"
    HEALTHY = "healthy"
    WATCH = "watch"
    DEGRADED = "degraded"
    QUARANTINED = "quarantined"
    MUTED = "muted"


# Code-error threshold for QUARANTINED transition. 3 consecutive
# uncaught exceptions strongly suggests a drill code bug rather
# than a real environmental issue. Quarantined drills require
# operator action to re-enter scheduling.
QUARANTINE_THRESHOLD = 3

# Backoff schedule for DEGRADED state. Each consecutive failure
# doubles the wait, capped at ``cadence_days``. The first failure
# (WATCH state) waits ``WATCH_BACKOFF_SECONDS``; subsequent
# failures use exponential from there.
WATCH_BACKOFF_SECONDS = 15 * 60       # 15 min
DEGRADED_BASE_SECONDS = 60 * 60       # 1h
MAX_TRANSITION_HISTORY = 12           # last N kept per drill

# Default warmup period for newly-registered drills (overridable
# via ``DrillSpec.warmup_days``). During warmup the drill runs and
# emits observations but doesn't fire alerts or escalate state.
DEFAULT_WARMUP_DAYS = 7


# ── State record ─────────────────────────────────────────────────────────


@dataclass
class StateTransition:
    """One state transition. Kept in a bounded ring for operator audit."""

    at: str                     # ISO-8601
    from_state: str
    to_state: str
    reason: str
    triggered_by: str = "scheduler"  # "scheduler" | "operator" | "boot"


@dataclass
class DrillStateRecord:
    """Persistent state for one drill. Single record per drill name."""

    drill_name: str
    state: DrillState = DrillState.WARMING_UP

    # Counters reset on a successful pass (returns to HEALTHY).
    consecutive_failures: int = 0
    consecutive_code_errors: int = 0

    # Outcome timestamps (ISO-8601). last_run_at updates on every
    # invocation regardless of outcome; the two specific timestamps
    # only update on their respective outcomes.
    last_run_at: str | None = None
    last_success_at: str | None = None
    last_failure_at: str | None = None
    last_failure_summary: str = ""
    last_failure_class: str = ""
    last_traceback: str = ""

    # Computed: when the scheduler may next attempt this drill. None
    # means "any time" (newly-registered before first run). For
    # QUARANTINED + MUTED states this is None — the scheduler
    # checks the state enum, not the timestamp.
    next_attempt_after: str | None = None

    # WARMING_UP: when the warmup period ends. Set on first
    # registration; cleared on transition out of WARMING_UP.
    warming_up_until: str | None = None

    # QUARANTINED: when the drill entered quarantine + reason.
    quarantined_at: str | None = None
    quarantined_reason: str = ""

    # MUTED: when operator muted + optional auto-unmute time.
    muted_at: str | None = None
    muted_by: str = ""
    muted_until: str | None = None  # None = indefinite

    # Last 12 transitions for operator audit.
    transitions: list[StateTransition] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Serialize enum as string value
        d["state"] = self.state.value if isinstance(self.state, DrillState) else str(self.state)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DrillStateRecord":
        # Backward-compat: older records lack newer fields → use defaults.
        kwargs = {k: data.get(k) for k in (
            "drill_name", "consecutive_failures", "consecutive_code_errors",
            "last_run_at", "last_success_at", "last_failure_at",
            "last_failure_summary", "last_failure_class", "last_traceback",
            "next_attempt_after", "warming_up_until", "quarantined_at",
            "quarantined_reason", "muted_at", "muted_by", "muted_until",
        )}
        # state with default
        state_raw = data.get("state") or DrillState.WARMING_UP.value
        kwargs["state"] = DrillState(state_raw)
        kwargs["consecutive_failures"] = int(kwargs.get("consecutive_failures") or 0)
        kwargs["consecutive_code_errors"] = int(kwargs.get("consecutive_code_errors") or 0)
        kwargs["last_failure_summary"] = str(kwargs.get("last_failure_summary") or "")
        kwargs["last_failure_class"] = str(kwargs.get("last_failure_class") or "")
        kwargs["last_traceback"] = str(kwargs.get("last_traceback") or "")
        kwargs["quarantined_reason"] = str(kwargs.get("quarantined_reason") or "")
        kwargs["muted_by"] = str(kwargs.get("muted_by") or "")
        rec = cls(**kwargs)
        for t in (data.get("transitions") or [])[-MAX_TRANSITION_HISTORY:]:
            rec.transitions.append(StateTransition(
                at=str(t.get("at") or ""),
                from_state=str(t.get("from_state") or ""),
                to_state=str(t.get("to_state") or ""),
                reason=str(t.get("reason") or ""),
                triggered_by=str(t.get("triggered_by") or "scheduler"),
            ))
        return rec


# ── Path resolution ──────────────────────────────────────────────────────


_lock = threading.RLock()


def _state_dir() -> Path:
    """Return the directory for per-drill state files. Honors
    ``WORKSPACE_ROOT``."""
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "resilience" / "drill_state"
    except Exception:
        return Path("/app/workspace/resilience/drill_state")


def _state_path(drill_name: str) -> Path:
    return _state_dir() / f"{drill_name}.json"


# ── Persistence ──────────────────────────────────────────────────────────


def load(drill_name: str) -> DrillStateRecord | None:
    """Read a drill's persisted state. Returns None when not yet
    written (first-ever boot or freshly-registered drill)."""
    path = _state_path(drill_name)
    if not path.exists():
        return None
    try:
        with _lock:
            data = json.loads(path.read_text(encoding="utf-8"))
        return DrillStateRecord.from_dict(data)
    except Exception:
        logger.warning("drill_state: cannot read %s", path, exc_info=True)
        return None


def save(record: DrillStateRecord) -> bool:
    """Atomically persist a drill's state. Returns True on success."""
    path = _state_path(record.drill_name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            # Trim transition history to last MAX_TRANSITION_HISTORY
            if len(record.transitions) > MAX_TRANSITION_HISTORY:
                record.transitions = record.transitions[-MAX_TRANSITION_HISTORY:]
            body = json.dumps(record.to_dict(), indent=2, sort_keys=True, default=str)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8",
                dir=str(path.parent),
                prefix=f".{record.drill_name}.",
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
        logger.warning("drill_state: persist failed for %s", record.drill_name, exc_info=True)
        return False


def load_or_initialize(drill_name: str, *, warmup_days: int = DEFAULT_WARMUP_DAYS) -> DrillStateRecord:
    """Read the state record, or initialize a WARMING_UP record on
    first access. The newly-initialized record IS persisted so
    subsequent reads return the same record (with the same
    warming_up_until)."""
    existing = load(drill_name)
    if existing is not None:
        return existing
    now = datetime.now(timezone.utc)
    warming_until = (now.timestamp() + warmup_days * 86400.0)
    warming_until_iso = datetime.fromtimestamp(warming_until, tz=timezone.utc).isoformat()
    record = DrillStateRecord(
        drill_name=drill_name,
        state=DrillState.WARMING_UP,
        warming_up_until=warming_until_iso,
        transitions=[StateTransition(
            at=now.isoformat(),
            from_state="(none)",
            to_state=DrillState.WARMING_UP.value,
            reason=f"drill registered; {warmup_days}d warmup",
            triggered_by="boot",
        )],
    )
    save(record)
    return record


def list_all_state_records() -> list[DrillStateRecord]:
    """Enumerate every persisted state record. Used by the REST
    surface + operator views."""
    out: list[DrillStateRecord] = []
    sd = _state_dir()
    if not sd.exists():
        return out
    for p in sorted(sd.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            out.append(DrillStateRecord.from_dict(data))
        except Exception:
            logger.warning("drill_state: skipping unreadable %s", p, exc_info=True)
    return out


# ── Transition logic ─────────────────────────────────────────────────────


def _record_transition(record: DrillStateRecord, to_state: DrillState, reason: str,
                       triggered_by: str = "scheduler") -> None:
    """Append a transition to the audit ring. Does NOT persist —
    caller calls ``save()``."""
    record.transitions.append(StateTransition(
        at=datetime.now(timezone.utc).isoformat(),
        from_state=record.state.value,
        to_state=to_state.value,
        reason=reason[:200],
        triggered_by=triggered_by,
    ))
    record.state = to_state


def is_warming_up(record: DrillStateRecord, *, now: datetime | None = None) -> bool:
    """True iff the drill is still inside its warmup window. Used by
    the scheduler to suppress alerts during warmup."""
    if record.state != DrillState.WARMING_UP:
        return False
    if not record.warming_up_until:
        return False
    try:
        until = datetime.fromisoformat(record.warming_up_until.replace("Z", "+00:00"))
    except ValueError:
        return False
    now_dt = now or datetime.now(timezone.utc)
    return now_dt < until


def is_runnable_now(record: DrillStateRecord, *, now: datetime | None = None) -> tuple[bool, str]:
    """Whether the scheduler may invoke this drill right now.

    Returns (ok, reason). ``reason`` is a short human-readable
    explanation used for skip-logging.

    Decision rules (in order):
      * MUTED → False (operator silenced)
      * QUARANTINED → False (operator must unquarantine)
      * WARMING_UP within warming_up_until → True (drill should run
        and gather observations; just no alerts)
      * next_attempt_after in the future → False (backoff active)
      * else → True
    """
    now_dt = now or datetime.now(timezone.utc)

    if record.state == DrillState.MUTED:
        # Auto-unmute check
        if record.muted_until:
            try:
                muted_until = datetime.fromisoformat(record.muted_until.replace("Z", "+00:00"))
                if now_dt >= muted_until:
                    return True, "auto-unmute window expired"
            except ValueError:
                pass
        return False, "muted by operator"

    if record.state == DrillState.QUARANTINED:
        return False, "quarantined (operator action required)"

    # next_attempt_after is the canonical backoff gate.
    if record.next_attempt_after:
        try:
            naa = datetime.fromisoformat(record.next_attempt_after.replace("Z", "+00:00"))
            if now_dt < naa:
                wait_s = (naa - now_dt).total_seconds()
                return False, f"backoff active ({wait_s:.0f}s remaining)"
        except ValueError:
            # Malformed → fail-open so the drill can still run
            pass

    return True, "ok"


def _compute_backoff_seconds(record: DrillStateRecord, cadence_days: int) -> int:
    """Backoff seconds based on consecutive_failures count.

    Schedule: WATCH (1 fail) = 15 min; DEGRADED (2+) = 1h, 2h, 4h, 8h
    ... capped at cadence_days. The exponential keeps a flapping
    drill from re-running every idle pass while a chronically-broken
    one trends toward its cadence (= least frequent allowed)."""
    if record.consecutive_failures <= 1:
        return WATCH_BACKOFF_SECONDS
    # DEGRADED — exponential from DEGRADED_BASE_SECONDS
    # consecutive_failures=2 → 1h, 3 → 2h, 4 → 4h, ...
    exponent = max(0, record.consecutive_failures - 2)
    seconds = DEGRADED_BASE_SECONDS * (2 ** exponent)
    cap = max(cadence_days, 1) * 86400
    return int(min(seconds, cap))


def record_pass(record: DrillStateRecord, *, cadence_days: int) -> None:
    """Update the record after a successful drill outcome.

    Transitions: any state (except MUTED/QUARANTINED) → HEALTHY.
    Resets consecutive_* counters; sets next_attempt_after to
    cadence_days from now.
    """
    now = datetime.now(timezone.utc)
    record.last_run_at = now.isoformat()
    record.last_success_at = now.isoformat()
    record.consecutive_failures = 0
    record.consecutive_code_errors = 0
    record.last_failure_summary = ""
    record.last_failure_class = ""
    record.last_traceback = ""
    naa = datetime.fromtimestamp(now.timestamp() + cadence_days * 86400.0, tz=timezone.utc)
    record.next_attempt_after = naa.isoformat()

    target_state: DrillState
    if record.state in (DrillState.MUTED, DrillState.QUARANTINED):
        # Don't auto-exit muted/quarantined on a pass — operator
        # must take action explicitly. We DO update timestamps
        # though, in case a manual run produced a pass.
        target_state = record.state
    elif record.state == DrillState.WARMING_UP:
        # Warmup ended OR drill passed during warmup — promote to
        # HEALTHY if the warmup window has elapsed; else stay in
        # WARMING_UP (no alert noise during warmup).
        if not is_warming_up(record):
            _record_transition(record, DrillState.HEALTHY, "warmup completed with pass")
            record.warming_up_until = None
            target_state = DrillState.HEALTHY
        else:
            target_state = DrillState.WARMING_UP
    else:
        if record.state != DrillState.HEALTHY:
            _record_transition(record, DrillState.HEALTHY, "pass after failure")
        target_state = DrillState.HEALTHY
    record.state = target_state


def record_failure(record: DrillStateRecord, *,
                   cadence_days: int,
                   failure_class: str,
                   summary: str = "",
                   traceback_text: str = "") -> None:
    """Update the record after a failed drill outcome.

    Transitions:
      * CODE_ERROR ≥ QUARANTINE_THRESHOLD → QUARANTINED
      * other → WATCH (first) or DEGRADED (subsequent)
      * WARMING_UP failures stay WARMING_UP (observational only)

    Resets next_attempt_after to backoff.
    """
    now = datetime.now(timezone.utc)
    record.last_run_at = now.isoformat()
    record.last_failure_at = now.isoformat()
    record.last_failure_class = failure_class
    record.last_failure_summary = summary[:500] if summary else ""
    record.last_traceback = traceback_text[:4000] if traceback_text else ""

    is_code_error = failure_class == "code_error"
    record.consecutive_failures += 1
    if is_code_error:
        record.consecutive_code_errors += 1
    else:
        # Non-code-error resets the code-error counter — one drill
        # producing both kinds shouldn't accumulate two parallel paths.
        record.consecutive_code_errors = 0

    # Don't escalate state during warmup; just record observation.
    if record.state == DrillState.WARMING_UP and is_warming_up(record):
        backoff = _compute_backoff_seconds(record, cadence_days)
        naa = datetime.fromtimestamp(now.timestamp() + backoff, tz=timezone.utc)
        record.next_attempt_after = naa.isoformat()
        return

    if record.state in (DrillState.MUTED, DrillState.QUARANTINED):
        # Already in terminal-ish state; don't re-transition. Just
        # update backoff in case operator un-mutes.
        backoff = _compute_backoff_seconds(record, cadence_days)
        naa = datetime.fromtimestamp(now.timestamp() + backoff, tz=timezone.utc)
        record.next_attempt_after = naa.isoformat()
        return

    # QUARANTINE check FIRST — a CODE_ERROR run that pushes us over
    # the threshold goes straight to QUARANTINED regardless of prior
    # state.
    if record.consecutive_code_errors >= QUARANTINE_THRESHOLD:
        _record_transition(
            record, DrillState.QUARANTINED,
            f"{record.consecutive_code_errors} consecutive code errors",
        )
        record.quarantined_at = now.isoformat()
        record.quarantined_reason = (
            f"{record.consecutive_code_errors} consecutive code errors. "
            f"Last error: {summary[:200]}"
        )
        # When QUARANTINED, next_attempt_after is moot — scheduler
        # checks state enum and refuses to run. Clear it to make
        # the JSON unambiguous.
        record.next_attempt_after = None
        return

    # Normal escalation: HEALTHY/WARMING_UP/WATCH → WATCH; WATCH → DEGRADED.
    if record.consecutive_failures == 1:
        if record.state != DrillState.WATCH:
            _record_transition(record, DrillState.WATCH,
                                f"first failure ({failure_class})")
    else:
        if record.state != DrillState.DEGRADED:
            _record_transition(
                record, DrillState.DEGRADED,
                f"{record.consecutive_failures} consecutive failures",
            )

    backoff = _compute_backoff_seconds(record, cadence_days)
    naa = datetime.fromtimestamp(now.timestamp() + backoff, tz=timezone.utc)
    record.next_attempt_after = naa.isoformat()


# ── Operator actions ─────────────────────────────────────────────────────


def unquarantine(drill_name: str, *, operator: str, reason: str = "") -> DrillStateRecord | None:
    """Clear a drill's QUARANTINED state. Returns the updated record
    or None if no state exists. Sets state to WATCH so the drill
    re-runs once and either passes (→ HEALTHY) or fails (→ DEGRADED
    with full backoff)."""
    record = load(drill_name)
    if record is None:
        return None
    if record.state != DrillState.QUARANTINED:
        # No-op if not actually quarantined.
        return record
    _record_transition(
        record, DrillState.WATCH,
        f"unquarantined by {operator}: {reason}"[:200],
        triggered_by="operator",
    )
    record.quarantined_at = None
    record.quarantined_reason = ""
    record.consecutive_code_errors = 0
    # Allow immediate re-run.
    record.next_attempt_after = None
    save(record)
    return record


def mute(drill_name: str, *, operator: str, reason: str = "",
         until_iso: str | None = None) -> DrillStateRecord | None:
    """Silence a drill. Optional auto-unmute timestamp."""
    record = load_or_initialize(drill_name)
    _record_transition(
        record, DrillState.MUTED,
        f"muted by {operator}: {reason}"[:200],
        triggered_by="operator",
    )
    record.muted_at = datetime.now(timezone.utc).isoformat()
    record.muted_by = operator
    record.muted_until = until_iso
    save(record)
    return record


def unmute(drill_name: str, *, operator: str) -> DrillStateRecord | None:
    """Lift operator mute. Returns to HEALTHY (clean slate)."""
    record = load(drill_name)
    if record is None:
        return None
    if record.state != DrillState.MUTED:
        return record
    _record_transition(
        record, DrillState.HEALTHY,
        f"unmuted by {operator}",
        triggered_by="operator",
    )
    record.muted_at = None
    record.muted_by = ""
    record.muted_until = None
    record.next_attempt_after = None
    save(record)
    return record


def reset_for_tests() -> None:
    """Test-only — remove all state files in the configured dir."""
    sd = _state_dir()
    if not sd.exists():
        return
    for p in sd.glob("*.json"):
        try:
            p.unlink()
        except OSError:
            pass
    for p in sd.glob(".*.tmp"):
        try:
            p.unlink()
        except OSError:
            pass
