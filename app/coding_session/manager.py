"""Coding-session lifecycle orchestration.

Public entry points (one per state transition + read accessors):

  * ``start(*, agent_id, base, purpose)`` — create a worktree from
    the given base; persist the session as ACTIVE.
  * ``get(session_id)`` — read.
  * ``list_active(*, agent_id=None)`` — read.
  * ``touch(session_id)`` — refresh ``last_activity_at`` (called from
    every tool that operates on the session, so idle-timeout works).
  * ``record_write(session_id, path, content_size)`` — track files +
    bytes for quota enforcement.
  * ``record_run(session_id)`` — increment the run counter.
  * ``submit(session_id, *, results)`` — ACTIVE → SUBMITTED.
  * ``discard(session_id, *, reason)`` — ACTIVE → DISCARDED.
  * ``expire(session_id, *, reason)`` — ACTIVE → EXPIRED (reconciler).
  * ``fail(session_id, *, reason)`` — ACTIVE → FAILED (corruption).

Worktree creation goes through an injectable ``WorktreeBackend``
protocol — production wires this to ``BridgeWorktreeBackend`` (host
``git worktree add`` via the bridge), tests use an in-process
``LocalWorktreeBackend`` against a real local repo. The lifecycle code
is identical either way.

A session in any non-ACTIVE state is read-only — every transition
function rejects with a clear error if the session is already
terminal. This is the same idempotency shape as the change-request
lifecycle: callers can retry safely.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

from app.coding_session import store
from app.coding_session.models import CodingSession, Status, SubmitResult
from app.coding_session.quotas import (
    DEFAULTS,
    QuotaConfig,
    QuotaResult,
    can_start_session,
)

logger = logging.getLogger(__name__)


# ── Worktree backend ────────────────────────────────────────────────


class WorktreeBackend(Protocol):
    """Abstracts the git operations needed by the manager + submit.
    Two concrete implementations:

      * ``LocalWorktreeBackend`` — runs ``subprocess.run`` directly
        (used in tests against a real local repo fixture).
      * ``BridgeWorktreeBackend`` — sends commands over the host
        bridge so the operations run on the host (used in production
        — the gateway container can't run git directly against the
        host repo).

    Three groups of operations:

      * **Lifecycle** (``resolve_ref``, ``create_worktree``,
        ``remove_worktree``) — used by the manager.
      * **Read** (``list_changed_paths``, ``read_worktree_file``,
        ``read_base_file``) — used by the submit module to compute
        per-file diffs.

    The split is intentional: the manager doesn't need read access;
    the submit module doesn't need lifecycle; both share the same
    backend instance so ops use the same git context.
    """

    # ── Lifecycle ─────────────────────────────────────────────────

    def resolve_ref(self, ref: str) -> str:
        """Resolve a branch/tag/sha to a commit sha. Raises
        ``ValueError`` if the ref doesn't exist."""

    def create_worktree(self, *, worktree_path: str, base_sha: str) -> None:
        """Create a fresh worktree at ``worktree_path`` checked out
        at ``base_sha``. Raises on failure."""

    def remove_worktree(self, *, worktree_path: str, force: bool = True) -> None:
        """Tear down a worktree. ``force=True`` is the default —
        the manager always wants the cleanup to succeed regardless of
        the worktree's dirty state."""

    # ── Read (used by submit) ─────────────────────────────────────

    def list_changed_paths(
        self, *, worktree_path: str,
    ) -> list[tuple[str, str]]:
        """Return the list of paths modified inside the worktree
        relative to its base sha, as ``(path, kind)`` tuples where
        ``kind ∈ {"M", "A", "D", "R"}`` (modified, added, deleted,
        renamed — same letters as ``git status --porcelain``).

        Used by ``submit_session`` to discover what to file change
        requests for. Empty list means a clean worktree (submit is
        a no-op).
        """

    def read_worktree_file(
        self, *, worktree_path: str, path: str,
    ) -> str:
        """Read the current content of ``worktree_path/path``.
        Raises ``FileNotFoundError`` if the file doesn't exist
        (e.g. it's a deletion case — submit handles that by reading
        the base content and recording a delete)."""

    def read_base_file(self, *, base_sha: str, path: str) -> str:
        """Read the content of ``path`` at ``base_sha``. Used to
        produce ``old_content`` for the change request — captures
        the "before" the agent's edits.

        Raises ``FileNotFoundError`` if ``path`` did not exist at
        ``base_sha`` (i.e. the agent added a new file). The submit
        module catches this and uses ``""`` for the new-file case.
        """


# ── Manager ─────────────────────────────────────────────────────────


@dataclass
class Manager:
    """Stateless orchestrator. Holds the backend + quota config; reads
    + writes go through the store module's globals.

    Manager instances are cheap to construct — there's typically one
    per Manager-using subsystem (the tools layer, the reconciler, the
    control-plane API), and they share state via the store module.
    """

    backend: WorktreeBackend
    config: QuotaConfig = DEFAULTS

    # ── Read accessors ────────────────────────────────────────────

    def get(self, session_id: str) -> CodingSession | None:
        return store.get(session_id)

    def list_active(self, *, agent_id: str | None = None) -> list[CodingSession]:
        return store.list_all(status=Status.ACTIVE, agent_id=agent_id)

    # ── Lifecycle: start ──────────────────────────────────────────

    def start(
        self,
        *,
        agent_id: str,
        base: str,
        purpose: str,
        worktree_root: str | Path,
    ) -> CodingSession:
        """Create a fresh ACTIVE session.

        Raises ``QuotaExceeded`` if the per-agent or system quota is
        already at the cap.

        Raises ``ValueError`` for malformed input (empty agent_id,
        empty purpose, or an unresolvable base ref).
        """
        if not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if not purpose or not purpose.strip():
            raise ValueError("purpose must be a non-empty string")
        if not base:
            raise ValueError("base must be a non-empty branch/ref")

        # Quota check
        check = can_start_session(
            config=self.config,
            agent_active_count=store.count_active(agent_id=agent_id),
            system_active_count=store.count_active(),
        )
        if not check.ok:
            raise QuotaExceeded(check.reason or "quota check failed")

        # Resolve base → sha (locks the session to a fixed commit, so the
        # diff at submit time is computed against this sha even if the
        # base branch has moved in the meantime)
        try:
            base_sha = self.backend.resolve_ref(base)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"cannot resolve base ref {base!r}: {exc}"
            ) from exc

        # Build the session record before creating the worktree so the
        # id is available for the path
        session_id = str(uuid.uuid4())
        worktree_path = str(Path(worktree_root) / session_id)

        now = _now_iso()
        expires_at = _now_plus(self.config.ttl_seconds)
        session = CodingSession(
            id=session_id,
            agent_id=agent_id,
            purpose=purpose.strip(),
            created_at=now,
            base=base,
            base_sha=base_sha,
            worktree_path=worktree_path,
            expires_at=expires_at,
            last_activity_at=now,
            status=Status.ACTIVE,
        )

        # Actually create the worktree on disk. If this fails, we don't
        # persist the session — leaving no orphan record.
        try:
            self.backend.create_worktree(
                worktree_path=worktree_path, base_sha=base_sha,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "coding_session.start: create_worktree failed for %s at %s: %s",
                session_id, worktree_path, exc,
            )
            raise

        # Persist + audit
        store.save(session, audit_event="started")
        return session

    # ── Lifecycle: ACTIVE-only mutations ──────────────────────────

    def touch(self, session_id: str) -> None:
        """Refresh ``last_activity_at``. Caller is any tool that
        successfully operated on the session — keeps idle-timeout
        accurate.

        No-op (debug log only) if the session is terminal: the caller
        likely raced with the reconciler; tools layer surfaces a clear
        error to the agent through a separate path.
        """
        cs = store.get(session_id)
        if cs is None:
            logger.debug("touch: unknown session %s", session_id)
            return
        if cs.is_terminal:
            logger.debug(
                "touch: session %s is %s; skipping",
                session_id, cs.status.value,
            )
            return
        cs.last_activity_at = _now_iso()
        store.save(cs)  # no audit — touches are too frequent to log

    def record_write(
        self,
        session_id: str,
        path: str,
        content_size: int,
    ) -> CodingSession:
        """Add ``path`` to ``files_touched`` (deduped) and add
        ``content_size`` to ``bytes_written``. Refreshes activity.

        Raises ``IllegalTransition`` if the session isn't ACTIVE.
        """
        cs = self._require_active(session_id, op="record_write")
        if path not in cs.files_touched:
            cs.files_touched = [*cs.files_touched, path]
        cs.bytes_written += int(content_size)
        cs.last_activity_at = _now_iso()
        store.save(cs)  # no audit — too frequent
        return cs

    def record_run(self, session_id: str) -> CodingSession:
        """Increment the run counter + refresh activity. Raises
        ``IllegalTransition`` if the session isn't ACTIVE."""
        cs = self._require_active(session_id, op="record_run")
        cs.run_count += 1
        cs.last_activity_at = _now_iso()
        store.save(cs)
        return cs

    # ── Lifecycle: transitions out of ACTIVE ──────────────────────

    def submit(
        self,
        session_id: str,
        *,
        results: list[SubmitResult],
    ) -> CodingSession:
        """ACTIVE → SUBMITTED. Stores per-file submit results and
        marks the session terminal. Worktree teardown is the caller's
        responsibility (submit module handles bridge interaction)."""
        cs = self._require_active(session_id, op="submit")
        cs.status = Status.SUBMITTED
        cs.terminated_at = _now_iso()
        cs.terminated_reason = "submitted"
        cs.submit_results = list(results)
        store.save(cs, audit_event="submitted")
        return cs

    def discard(self, session_id: str, *, reason: str) -> CodingSession:
        """ACTIVE → DISCARDED. Agent gave up; record reason for
        postmortem.

        Idempotent: if already DISCARDED, returns the existing record.
        Raises ``IllegalTransition`` for any other terminal status —
        the operator should see "already submitted" as an error, not
        as a successful discard.
        """
        cs = store.get(session_id)
        if cs is None:
            raise IllegalTransition(f"session {session_id!r} not found")
        if cs.status is Status.DISCARDED:
            return cs
        if cs.status is not Status.ACTIVE:
            raise IllegalTransition(
                f"cannot discard in status {cs.status.value}; "
                f"only ACTIVE sessions can be discarded."
            )
        cs.status = Status.DISCARDED
        cs.terminated_at = _now_iso()
        cs.terminated_reason = reason or "discarded"
        store.save(cs, audit_event="discarded")
        return cs

    def expire(self, session_id: str, *, reason: str) -> CodingSession:
        """ACTIVE → EXPIRED. Reconciler-driven; never called by the
        agent. Idempotent — expiring an already-EXPIRED session is a
        no-op (handles reconciler retry races)."""
        cs = store.get(session_id)
        if cs is None:
            raise IllegalTransition(f"session {session_id!r} not found")
        if cs.status is Status.EXPIRED:
            return cs
        if cs.status is not Status.ACTIVE:
            raise IllegalTransition(
                f"cannot expire in status {cs.status.value}; "
                f"only ACTIVE sessions can be expired."
            )
        cs.status = Status.EXPIRED
        cs.terminated_at = _now_iso()
        cs.terminated_reason = reason or "expired"
        store.save(cs, audit_event="expired")
        return cs

    def fail(self, session_id: str, *, reason: str) -> CodingSession:
        """ACTIVE → FAILED. Worktree corruption or other
        infrastructure failure; the worktree is retained for forensics
        (the manager does NOT remove it on FAILED). Agent gets a clear
        error; operator can investigate."""
        cs = store.get(session_id)
        if cs is None:
            raise IllegalTransition(f"session {session_id!r} not found")
        if cs.status is Status.FAILED:
            return cs
        if cs.status is not Status.ACTIVE:
            raise IllegalTransition(
                f"cannot fail in status {cs.status.value}",
            )
        cs.status = Status.FAILED
        cs.terminated_at = _now_iso()
        cs.terminated_reason = reason or "failed"
        store.save(cs, audit_event="failed")
        return cs

    # ── Worktree teardown helper ──────────────────────────────────

    def remove_worktree(self, session: CodingSession) -> tuple[bool, str | None]:
        """Best-effort worktree removal. Used by submit / discard
        flows AFTER the session record has transitioned. Returns
        (ok, error). FAILED sessions are NOT torn down — the
        worktree stays for postmortem.
        """
        if session.status is Status.FAILED:
            return True, "FAILED session; worktree retained for postmortem"
        try:
            self.backend.remove_worktree(
                worktree_path=session.worktree_path, force=True,
            )
            return True, None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "coding_session.remove_worktree: %s for session %s: %s",
                exc, session.id, session.worktree_path,
            )
            return False, str(exc)

    # ── Internals ─────────────────────────────────────────────────

    def _require_active(self, session_id: str, *, op: str) -> CodingSession:
        cs = store.get(session_id)
        if cs is None:
            raise IllegalTransition(
                f"{op}: session {session_id!r} not found",
            )
        if cs.status is not Status.ACTIVE:
            raise IllegalTransition(
                f"{op}: session is {cs.status.value} (not ACTIVE); "
                f"no further mutations allowed.",
            )
        return cs


# ── Exceptions ──────────────────────────────────────────────────────


class QuotaExceeded(RuntimeError):
    """Raised when start() is rejected by the quota policy."""


class IllegalTransition(RuntimeError):
    """Raised when a state-machine transition is not allowed."""


# ── Time helpers ────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _now_plus(seconds: int) -> str:
    return (_now() + timedelta(seconds=seconds)).isoformat()
