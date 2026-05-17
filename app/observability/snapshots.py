"""
snapshots.py — Typed, store-agnostic observability snapshots.

A *snapshot* is a stable record of system state at a point in time —
"what is the gateway doing right now?", "what does the SubIA kernel
look like?", "what are the latest anomalies?" — serialised to a dict
payload and tagged with a short ``kind`` string.

Prior to this module, publishers (then in ``app.firebase.publishers``,
now ``app.observability.publishers``) bundled three concerns into a
single function: compute the data, choose where to store it
(Firestore), and implicitly define how it's served (via the Firebase
SDK on the dashboard side).  Snapshots pull those apart:

    ┌─ compute ─┐   ┌─ storage ─┐   ┌─ serving ─┐
    │ publisher │ → │   sink    │ → │  HTTP API │
    │   fn()    │   │ (store)   │   │  endpoint │
    └───────────┘   └───────────┘   └───────────┘
        pure            pluggable        queries

Compute lives in publisher functions that just return a ``Snapshot``.
Storage lives behind a ``SnapshotStore`` protocol; ``PostgresSnapshotStore``
writes to a single ``observability_snapshots`` table in the existing
control-plane Postgres.  Serving comes from HTTP endpoints under
``/api/cp/observability/snapshots/...`` which read through the same
store abstraction.

The protocol is deliberately minimal (put / latest / recent) so a
future Firestore-backed store, an in-memory store for tests, or a
``TeeStore`` that mirrors writes to both Firestore and Postgres during
a migration, can all slot in without changing callers.

Schema
------
``observability_snapshots``::

    id           bigserial primary key
    ts           timestamptz default now()   -- when the snapshot was taken
    kind         text not null               -- "heartbeat", "subia_state", ...
    payload      jsonb not null              -- the snapshot body

One index on ``(kind, ts desc)`` gives O(log n) retrieval for the two
workloads that matter: ``latest`` (kind = X order by ts desc limit 1)
and ``recent`` (kind = X order by ts desc limit N).

Retention is intentionally not handled here — data-age policy is a
separate concern that belongs wherever the app decides how long to
keep observability data (e.g. a cron that runs
``DELETE FROM observability_snapshots WHERE ts < now() - interval '30 days'``).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ── Snapshot value object ────────────────────────────────────────────


@dataclass
class Snapshot:
    """A single point-in-time observation.

    ``kind`` is a short, stable string that identifies the shape of
    ``payload``.  Consumers (HTTP endpoints, tests, dashboards) query
    by ``kind``; storage is opaque over payload contents.

    ``payload`` is any JSON-serialisable dict.  Deeply nested structures
    are fine — jsonb handles arbitrary depth — but keep it under ~64 KB
    per snapshot for reasonable query performance.

    ``ts`` defaults to "now (UTC)" at construction so callers don't
    have to pass a timestamp unless they're replaying historical data.
    """
    kind: str
    payload: dict
    ts: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ── Store protocol ───────────────────────────────────────────────────


class SnapshotStore(Protocol):
    """Pluggable storage for snapshots.

    Intentionally minimal: three operations that cover every known
    consumer today (latest for status displays, recent for history
    panels, put for publishers).  Anything more elaborate —
    aggregation, filtering, time-range queries — composes on top.
    """

    def put(self, snap: Snapshot) -> None: ...
    def latest(self, kind: str) -> Snapshot | None: ...
    def recent(
        self, kind: str, *, limit: int = 50,
    ) -> list[Snapshot]: ...


# ── Postgres implementation ──────────────────────────────────────────


class PostgresSnapshotStore:
    """Stores snapshots in the control-plane Postgres.

    The table is created on first use (idempotent) so deployment
    doesn't require a migration step.  Every write is wrapped in a
    try/except so a storage failure (connection drop, disk full) can't
    propagate back into the publisher; observability is best-effort.
    """

    _ensured: bool = False

    def _ensure_table(self) -> None:
        # PR 1 fix (2026-05-16): the previous implementation called
        # ``execute()`` (optional semantics — swallows exceptions and
        # returns None). If the CREATE TABLE silently failed (pool
        # unavailable, permission denied, SQL error against a
        # half-migrated schema), ``_ensured`` was still set to True and
        # every subsequent put/latest/recent call also silently failed
        # against a non-existent table. Switching to ``execute_required``
        # means a CREATE failure now raises — we catch and stay
        # ``_ensured=False`` so the next ``put()`` retries.
        if self.__class__._ensured:
            return
        try:
            from app.control_plane.db import execute_required
            execute_required(
                """
                CREATE TABLE IF NOT EXISTS observability_snapshots (
                    id         BIGSERIAL PRIMARY KEY,
                    ts         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    kind       TEXT NOT NULL,
                    payload    JSONB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS
                    idx_obs_snapshots_kind_ts
                    ON observability_snapshots (kind, ts DESC);
                """
            )
            self.__class__._ensured = True
        except Exception:
            logger.debug(
                "observability.snapshots: could not ensure table "
                "(will retry on next write)",
                exc_info=True,
            )

    def put(self, snap: Snapshot) -> None:
        self._ensure_table()
        try:
            from app.control_plane.db import execute
            execute(
                """
                INSERT INTO observability_snapshots (ts, kind, payload)
                VALUES (%s, %s, %s)
                """,
                (snap.ts, snap.kind, json.dumps(snap.payload)),
            )
        except Exception:
            logger.debug(
                "observability.snapshots: write failed for kind=%s "
                "(non-fatal)", snap.kind, exc_info=True,
            )

    def latest(self, kind: str) -> Snapshot | None:
        self._ensure_table()
        try:
            from app.control_plane.db import execute
            rows = execute(
                """
                SELECT ts, kind, payload
                  FROM observability_snapshots
                 WHERE kind = %s
              ORDER BY ts DESC
                 LIMIT 1
                """,
                (kind,),
                fetch=True,
            )
            if not rows:
                return None
            r = rows[0] if isinstance(rows[0], dict) else {}
            payload = r.get("payload")
            if isinstance(payload, str):
                payload = json.loads(payload)
            return Snapshot(
                kind=r.get("kind", kind),
                payload=payload or {},
                ts=r.get("ts"),
            )
        except Exception:
            logger.debug(
                "observability.snapshots: latest(%s) failed",
                kind, exc_info=True,
            )
            return None

    def recent(
        self, kind: str, *, limit: int = 50,
    ) -> list[Snapshot]:
        self._ensure_table()
        limit = max(1, min(limit, 500))  # clamp
        try:
            from app.control_plane.db import execute
            rows = execute(
                """
                SELECT ts, kind, payload
                  FROM observability_snapshots
                 WHERE kind = %s
              ORDER BY ts DESC
                 LIMIT %s
                """,
                (kind, limit),
                fetch=True,
            )
            out: list[Snapshot] = []
            for r in rows or []:
                r = r if isinstance(r, dict) else {}
                payload = r.get("payload")
                if isinstance(payload, str):
                    payload = json.loads(payload)
                out.append(Snapshot(
                    kind=r.get("kind", kind),
                    payload=payload or {},
                    ts=r.get("ts"),
                ))
            return out
        except Exception:
            logger.debug(
                "observability.snapshots: recent(%s) failed",
                kind, exc_info=True,
            )
            return []


# ── Default store singleton ─────────────────────────────────────────
#
# One store instance per process is enough — SnapshotStore has no
# mutable state beyond a boolean "did we ensure the table yet".  The
# module-level singleton keeps callers from having to thread a store
# parameter through every layer.
#
# Tests can override this by calling ``set_default_store(...)`` with
# an in-memory stub.

_default_store: SnapshotStore = PostgresSnapshotStore()


def get_default_store() -> SnapshotStore:
    return _default_store


def set_default_store(store: SnapshotStore) -> None:
    """Replace the default store.  Intended for tests + the eventual
    ``TeeStore`` used during migration off Firestore."""
    global _default_store
    _default_store = store


# ── Convenience shortcuts ───────────────────────────────────────────


def put(kind: str, payload: dict) -> None:
    """Record a snapshot with the current time.  Shortcut for callers
    that don't want to build a ``Snapshot`` object by hand."""
    get_default_store().put(Snapshot(kind=kind, payload=payload or {}))


def latest(kind: str) -> Snapshot | None:
    return get_default_store().latest(kind)


def recent(kind: str, *, limit: int = 50) -> list[Snapshot]:
    return get_default_store().recent(kind, limit=limit)
