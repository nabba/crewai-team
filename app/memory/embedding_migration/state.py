"""Migration state machine over runtime_settings.

PROGRAM §40 (2026-05-10) — Q3 Item 12.

States::

    IDLE         — no plan loaded
    PLANNED      — plan saved; nothing fired yet
    DUAL_WRITE   — every new write goes to both source + shadow
    BACKFILLING  — historical rows being copied source → shadow
    SHADOW_READ  — sampled live queries hit shadow; NDCG measured
    READY        — convergence threshold met; cutover unblocked
    CUTOVER      — Tier-3 amendment in flight
    APPLIED      — _EMBED_DIM rewritten; source archived (within stand-down)
    STANDDOWN_COMPLETE  — source collections deleted; migration retired
    ABORTED      — operator cancelled; shadow collections deleted

Transitions are STRICTLY MONOTONIC except:
  * any state → ABORTED (operator panic button)
  * IDLE  → PLANNED (load plan)
  * PLANNED → IDLE  (cancel before any phase fires)

Storage:
  Phase + counters live in ``runtime_settings``  under
  ``embedding_migration_state``. Three counters live in the same blob:
  ``shadow_writes``, ``backfill_rows``, ``shadow_query_count``.

Master switches (default OFF, all gated through React `/cp/settings`):
  * ``embedding_migration_dual_write_enabled``
  * ``embedding_migration_shadow_read_enabled``
  * ``embedding_migration_cutover_enabled``     (rejected unless state == READY)

The cutover_enabled toggle is purely a request signal — actually
applying the cutover always also requires a Tier-3 amendment proposal
that passes the eligibility gate.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


PHASE_IDLE = "IDLE"
PHASE_PLANNED = "PLANNED"
PHASE_DUAL_WRITE = "DUAL_WRITE"
PHASE_BACKFILLING = "BACKFILLING"
PHASE_SHADOW_READ = "SHADOW_READ"
PHASE_READY = "READY"
PHASE_CUTOVER = "CUTOVER"
PHASE_APPLIED = "APPLIED"
PHASE_STANDDOWN_COMPLETE = "STANDDOWN_COMPLETE"
PHASE_ABORTED = "ABORTED"


_VALID_TRANSITIONS: dict[str, set[str]] = {
    PHASE_IDLE: {PHASE_PLANNED},
    PHASE_PLANNED: {PHASE_IDLE, PHASE_DUAL_WRITE, PHASE_ABORTED},
    PHASE_DUAL_WRITE: {PHASE_BACKFILLING, PHASE_ABORTED},
    PHASE_BACKFILLING: {PHASE_SHADOW_READ, PHASE_ABORTED},
    PHASE_SHADOW_READ: {PHASE_READY, PHASE_ABORTED},
    PHASE_READY: {PHASE_CUTOVER, PHASE_ABORTED, PHASE_SHADOW_READ},
    PHASE_CUTOVER: {PHASE_APPLIED, PHASE_ABORTED, PHASE_READY},
    PHASE_APPLIED: {PHASE_STANDDOWN_COMPLETE, PHASE_ABORTED},
    PHASE_STANDDOWN_COMPLETE: set(),
    PHASE_ABORTED: set(),
}


class MigrationStateError(RuntimeError):
    """Raised on invalid transition / corrupt state."""


@dataclass
class MigrationCounters:
    shadow_writes: int = 0
    backfill_rows: int = 0
    shadow_query_count: int = 0
    last_ndcg_at_10: float | None = None
    last_ndcg_window_size: int = 0
    cutover_requested_at: str | None = None
    applied_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "MigrationCounters":
        if not d:
            return cls()
        return cls(
            shadow_writes=int(d.get("shadow_writes") or 0),
            backfill_rows=int(d.get("backfill_rows") or 0),
            shadow_query_count=int(d.get("shadow_query_count") or 0),
            last_ndcg_at_10=(
                float(d["last_ndcg_at_10"])
                if d.get("last_ndcg_at_10") is not None else None
            ),
            last_ndcg_window_size=int(d.get("last_ndcg_window_size") or 0),
            cutover_requested_at=d.get("cutover_requested_at"),
            applied_at=d.get("applied_at"),
        )


@dataclass
class MigrationState:
    phase: str = PHASE_IDLE
    plan_id: str | None = None
    counters: MigrationCounters = field(default_factory=MigrationCounters)
    last_advance_at: float = 0.0
    last_advance_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "plan_id": self.plan_id,
            "counters": self.counters.to_dict(),
            "last_advance_at": self.last_advance_at,
            "last_advance_reason": self.last_advance_reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "MigrationState":
        if not d:
            return cls()
        return cls(
            phase=str(d.get("phase") or PHASE_IDLE),
            plan_id=d.get("plan_id"),
            counters=MigrationCounters.from_dict(d.get("counters")),
            last_advance_at=float(d.get("last_advance_at") or 0.0),
            last_advance_reason=str(d.get("last_advance_reason") or ""),
        )


_RUNTIME_KEY = "embedding_migration_state"
_state_lock = threading.Lock()


# ── Read / write through runtime_settings ────────────────────────────────


def _read_raw() -> dict[str, Any]:
    try:
        from app.runtime_settings import get_embedding_migration_state
        return get_embedding_migration_state()
    except Exception:
        logger.debug(
            "embedding_migration.state: runtime_settings unavailable",
            exc_info=True,
        )
    return {}


def _write_raw(blob: dict[str, Any]) -> None:
    try:
        from app.runtime_settings import set_embedding_migration_state
        set_embedding_migration_state(blob)
    except Exception:
        logger.debug(
            "embedding_migration.state: runtime_settings write failed",
            exc_info=True,
        )


def get_state() -> MigrationState:
    return MigrationState.from_dict(_read_raw())


def set_state(state: MigrationState) -> None:
    with _state_lock:
        _write_raw(state.to_dict())


# ── Transitions ─────────────────────────────────────────────────────────


def transition(new_phase: str, reason: str = "") -> MigrationState:
    with _state_lock:
        cur = MigrationState.from_dict(_read_raw())
        allowed = _VALID_TRANSITIONS.get(cur.phase, set())
        if new_phase not in allowed:
            raise MigrationStateError(
                f"invalid transition {cur.phase} → {new_phase}. "
                f"allowed: {sorted(allowed)}"
            )
        cur.phase = new_phase
        cur.last_advance_at = time.time()
        cur.last_advance_reason = reason
        _write_raw(cur.to_dict())
        logger.info(
            "embedding_migration: transition → %s (reason=%s)",
            new_phase, reason or "n/a",
        )
        return cur


def adopt_plan(plan_id: str) -> MigrationState:
    """Move IDLE → PLANNED with the given plan_id."""
    with _state_lock:
        cur = MigrationState.from_dict(_read_raw())
        if cur.phase != PHASE_IDLE:
            raise MigrationStateError(
                f"cannot adopt plan in phase {cur.phase}; abort first"
            )
        cur.phase = PHASE_PLANNED
        cur.plan_id = plan_id
        cur.last_advance_at = time.time()
        cur.last_advance_reason = "adopt_plan"
        _write_raw(cur.to_dict())
        return cur


def abort(reason: str = "operator-cancel") -> MigrationState:
    """Hard abort. Allowed from any non-terminal phase."""
    with _state_lock:
        cur = MigrationState.from_dict(_read_raw())
        if cur.phase in (PHASE_STANDDOWN_COMPLETE, PHASE_ABORTED):
            return cur
        cur.phase = PHASE_ABORTED
        cur.last_advance_at = time.time()
        cur.last_advance_reason = reason
        _write_raw(cur.to_dict())
        logger.warning("embedding_migration: ABORTED (%s)", reason)
        return cur


# ── Counter mutators ─────────────────────────────────────────────────────


def increment_shadow_write(n: int = 1) -> None:
    if n <= 0:
        return
    with _state_lock:
        blob = _read_raw()
        counters = MigrationCounters.from_dict(blob.get("counters"))
        counters.shadow_writes += n
        blob["counters"] = counters.to_dict()
        _write_raw(blob)


def increment_backfill(n: int = 1) -> None:
    if n <= 0:
        return
    with _state_lock:
        blob = _read_raw()
        counters = MigrationCounters.from_dict(blob.get("counters"))
        counters.backfill_rows += n
        blob["counters"] = counters.to_dict()
        _write_raw(blob)


def record_shadow_query(ndcg_at_10: float, window_size: int) -> None:
    """Record one shadow-read measurement. ``window_size`` is the
    rolling sample size used by the verifier."""
    with _state_lock:
        blob = _read_raw()
        counters = MigrationCounters.from_dict(blob.get("counters"))
        counters.shadow_query_count += 1
        counters.last_ndcg_at_10 = float(ndcg_at_10)
        counters.last_ndcg_window_size = int(window_size)
        blob["counters"] = counters.to_dict()
        _write_raw(blob)


# ── Master switches (read-only convenience) ──────────────────────────────


def dual_write_enabled() -> bool:
    """Source-of-truth for whether the chromadb_manager dual-writes."""
    try:
        from app.runtime_settings import (
            get_embedding_migration_dual_write_enabled,
        )
        if not get_embedding_migration_dual_write_enabled():
            return False
    except Exception:
        return False
    cur = get_state()
    return cur.phase in (
        PHASE_DUAL_WRITE, PHASE_BACKFILLING, PHASE_SHADOW_READ, PHASE_READY,
    )


def shadow_read_enabled() -> bool:
    try:
        from app.runtime_settings import (
            get_embedding_migration_shadow_read_enabled,
        )
        if not get_embedding_migration_shadow_read_enabled():
            return False
    except Exception:
        return False
    cur = get_state()
    return cur.phase in (PHASE_SHADOW_READ, PHASE_READY)


def cutover_enabled() -> bool:
    """The cutover toggle ALSO requires phase==READY. The Tier-3
    amendment is a separate gate on top of this."""
    try:
        from app.runtime_settings import (
            get_embedding_migration_cutover_enabled,
        )
        if not get_embedding_migration_cutover_enabled():
            return False
    except Exception:
        return False
    return get_state().phase == PHASE_READY
