"""Idle reconciler — kills expired and idle coding sessions.

Runs as an idle job (same pattern as ``belief-outbox-neo4j`` and
``dlq-drain``). Every ~5 minutes, scans ACTIVE sessions, and:

  * If ``now > expires_at`` → expire with reason ``"ttl"``
  * If ``now - last_activity_at > idle_seconds`` → expire with
    reason ``"idle"``

After marking expired, attempts to tear down the worktree via the
manager's backend. Failed teardowns are logged but don't prevent the
expire from sticking — the session record reaches a terminal state
even if the on-disk worktree lingers (a separate disk-pressure scan
can clean those up later).

The reconciler is **idempotent**: re-running over the same set of
ACTIVE sessions in the same minute is fine. ``manager.expire`` is a
no-op on already-EXPIRED sessions, so racing reconciler invocations
don't double-act.

Exposed entry point: ``run_once(*, manager) -> ReconcileReport``.
The ``ReconcileReport`` is for the operator's idle-job log; nothing
in the runtime depends on it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.coding_session import store
from app.coding_session.manager import IllegalTransition, Manager
from app.coding_session.models import CodingSession, Status

logger = logging.getLogger(__name__)


@dataclass
class ReconcileReport:
    """One row per session inspected; one summary line at the top."""

    scanned: int = 0
    expired_ttl: int = 0
    expired_idle: int = 0
    teardowns_ok: int = 0
    teardowns_failed: int = 0
    details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scanned": self.scanned,
            "expired_ttl": self.expired_ttl,
            "expired_idle": self.expired_idle,
            "teardowns_ok": self.teardowns_ok,
            "teardowns_failed": self.teardowns_failed,
            "details": list(self.details),
        }


def run_once(*, manager: Manager, now: datetime | None = None) -> ReconcileReport:
    """Scan ACTIVE sessions; expire those past TTL or idle."""
    now = now or datetime.now(timezone.utc)
    report = ReconcileReport()

    sessions = store.list_all(status=Status.ACTIVE, limit=1000)
    report.scanned = len(sessions)

    for cs in sessions:
        outcome = _classify(cs, now=now, idle_seconds=manager.config.idle_seconds)
        if outcome is None:
            continue

        reason, kind = outcome
        try:
            manager.expire(cs.id, reason=reason)
        except IllegalTransition:
            # Race with another reconciler / agent — already terminal.
            logger.debug(
                "reconciler: session %s no longer ACTIVE; skipping",
                cs.id,
            )
            continue

        if kind == "ttl":
            report.expired_ttl += 1
        else:
            report.expired_idle += 1

        # Best-effort worktree teardown
        cs_after = manager.get(cs.id) or cs
        ok, err = manager.remove_worktree(cs_after)
        if ok:
            report.teardowns_ok += 1
        else:
            report.teardowns_failed += 1

        report.details.append({
            "session_id": cs.id,
            "agent_id": cs.agent_id,
            "kind": kind,
            "reason": reason,
            "teardown_ok": ok,
            "teardown_error": err,
        })

    if report.scanned:
        logger.info(
            "coding_session.reconciler: scanned=%d ttl=%d idle=%d "
            "teardown_ok=%d teardown_failed=%d",
            report.scanned, report.expired_ttl, report.expired_idle,
            report.teardowns_ok, report.teardowns_failed,
        )
    return report


def _classify(
    cs: CodingSession,
    *,
    now: datetime,
    idle_seconds: int,
) -> tuple[str, str] | None:
    """Return (reason, kind) if the session should expire, else None."""
    expires_at = _parse_iso(cs.expires_at)
    last_activity = _parse_iso(cs.last_activity_at)

    if expires_at is not None and now >= expires_at:
        return f"ttl: created_at + ttl < now (expires_at={cs.expires_at})", "ttl"

    if last_activity is not None:
        idle = (now - last_activity).total_seconds()
        if idle >= idle_seconds:
            return (
                f"idle: no activity for {int(idle)}s "
                f"(>= {idle_seconds}s)",
                "idle",
            )

    return None


def _parse_iso(s: str) -> datetime | None:
    """Parse an ISO-8601 timestamp; return None if unparseable."""
    if not s:
        return None
    try:
        # Python's fromisoformat handles +00:00 and naive forms; explicit
        # tz fallback for the rare case where the timestamp came in naive
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        logger.warning("reconciler: unparseable timestamp %r", s)
        return None
