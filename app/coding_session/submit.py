"""Bundle a coding-session diff into change requests.

The ``submit_session`` function is the single escape hatch from a
worktree to production. It runs:

  1. Validates the session is ACTIVE
  2. Discovers changed paths via the backend's
     ``list_changed_paths(worktree_path)``
  3. For each path:
       a. Reads ``new_content`` from the worktree (or empty for D)
       b. Reads ``old_content`` from the base sha (or empty for A)
       c. Calls the change-request port's ``create_request(...)``
       d. If the resulting CR is PENDING, calls ``send_ask(cr.id)``
       e. Builds a SubmitResult row
  4. Calls ``manager.submit(session_id, results=...)`` to mark the
     session SUBMITTED and store the results
  5. Calls ``manager.remove_worktree(session)`` to clean up

The change-request port is injectable: tests pass a fake; production
uses the default that lazy-imports ``app.change_requests``. This
keeps Phase 5.4-c's unit tests independent of #54 — the integration
runs end-to-end once both branches land.

What submit handles correctly:

  * **Per-file split** — each touched file becomes its own change
    request. Operator sees one Signal ASK per file.
  * **TIER_IMMUTABLE refusal** — the change-request validator
    rejects the file at request time; we record a SubmitResult with
    no change_request_id and the validator's reason. Other files in
    the batch still submit normally.
  * **Validator failure** — same shape as TIER_IMMUTABLE refusal;
    the SubmitResult carries the validator's reason.
  * **New files** — base read raises FileNotFoundError; we use ""
    for old_content.
  * **Deleted files** (kind 'D') — Phase 5.4-c does NOT handle
    deletes (the change-request system has no delete primitive).
    We record a SubmitResult with refusal_reason "delete-not-supported";
    the agent must use a different workflow if it wants to remove a
    file. Tracked as a follow-up.
  * **Rename** (kind 'R') — treated as an add of the new path. The
    old path's content disappears from the resulting branch; the
    change request records the new file. (Conservative; full rename
    semantics come later if needed.)

What submit does NOT handle:

  * Re-opening a SUBMITTED session — re-iteration is a fresh session.
    The manager's ``submit()`` is the gatekeeper — it raises
    IllegalTransition on already-terminal sessions.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

from app.coding_session.manager import (
    IllegalTransition,
    Manager,
)
from app.coding_session.models import CodingSession, SubmitResult

logger = logging.getLogger(__name__)


# ── Change-request port ─────────────────────────────────────────────


class ChangeRequestPort(Protocol):
    """Seam between coding_session and the change-request system.

    Production wires ``DefaultChangeRequestPort`` which lazy-imports
    ``app.change_requests``. Tests pass a fake.
    """

    def create_request(
        self,
        *,
        requestor: str,
        path: str,
        new_content: str,
        old_content: str,
        reason: str,
    ) -> Any:
        """Create a ChangeRequest. Returns an object with ``.id`` and
        ``.status`` (a string-valued enum)."""

    def send_ask(self, request_id: str) -> int | None:
        """Send the Signal ASK; returns the message ts or None on
        failure. The submit module logs but doesn't fail the whole
        batch on send_ask errors."""


class DefaultChangeRequestPort:
    """Production-default port. Lazy-imports the real change-request
    module so unit tests of submit_session don't pull in the whole
    app.change_requests dependency tree."""

    def create_request(
        self,
        *,
        requestor: str,
        path: str,
        new_content: str,
        old_content: str,
        reason: str,
    ) -> Any:
        from app.change_requests import create_request

        return create_request(
            requestor=requestor,
            path=path,
            new_content=new_content,
            old_content=old_content,
            reason=reason,
        )

    def send_ask(self, request_id: str) -> int | None:
        from app.change_requests import send_ask

        return send_ask(request_id)


# ── Submit ──────────────────────────────────────────────────────────


def submit_session(
    session_id: str,
    *,
    submit_reason: str,
    manager: Manager,
    port: ChangeRequestPort | None = None,
    cleanup_worktree: bool = True,
) -> tuple[CodingSession, list[SubmitResult]]:
    """Discover the worktree's changes, file change requests, and
    finalize the session.

    Args:
        session_id: the session to submit. Must be ACTIVE.
        submit_reason: operator-facing explanation; appended to each
            change request's reason after the session's purpose.
        manager: the lifecycle manager (provides backend access).
        port: change-request seam; defaults to
            :class:`DefaultChangeRequestPort` (lazy-imports the real
            module).
        cleanup_worktree: if True (default), tear down the worktree
            after submit. Tests pass False to inspect the worktree
            after submit.

    Returns:
        ``(updated_session, [SubmitResult, ...])`` — the session in
        SUBMITTED status with ``submit_results`` populated, plus the
        same list returned for the caller's convenience (typically
        the tools layer).

    Raises:
        :class:`IllegalTransition` — session not ACTIVE or not found.
    """
    cs = manager.get(session_id)
    if cs is None:
        raise IllegalTransition(f"submit: session {session_id!r} not found")
    if not cs.is_active:
        raise IllegalTransition(
            f"submit: session is {cs.status.value} (not ACTIVE)"
        )

    port = port or DefaultChangeRequestPort()
    backend = manager.backend  # WorktreeBackend has the read methods

    # 1. Discover changed paths
    try:
        changes = backend.list_changed_paths(worktree_path=cs.worktree_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "submit: list_changed_paths failed for session %s: %s",
            session_id, exc,
        )
        manager.fail(session_id, reason=f"list_changed_paths failed: {exc}")
        # Re-raise so the tool surface returns a clean error
        raise

    results: list[SubmitResult] = []

    # 2. Per-file: build content + reason; call port; record result
    for path, kind in changes:
        try:
            result = _submit_one_file(
                cs=cs,
                path=path,
                kind=kind,
                submit_reason=submit_reason,
                port=port,
                backend=backend,
            )
        except Exception as exc:  # noqa: BLE001
            # An unexpected error per-file shouldn't kill the batch.
            # Record it as a refusal with the exception text and move
            # on. The session still terminates cleanly with the rest
            # of the results captured.
            logger.warning(
                "submit: file %s in session %s raised: %s",
                path, session_id, exc, exc_info=True,
            )
            result = SubmitResult(
                path=path,
                change_request_id=None,
                status="error",
                refusal_reason=f"{type(exc).__name__}: {exc}",
            )
        results.append(result)

    # 3. Mark session SUBMITTED + store the results
    updated = manager.submit(session_id, results=results)

    # 4. Tear down worktree (best-effort; failure non-fatal)
    if cleanup_worktree:
        ok, err = manager.remove_worktree(updated)
        if not ok:
            logger.warning(
                "submit: worktree teardown failed for session %s: %s",
                session_id, err,
            )

    return updated, results


# ── Per-file path ───────────────────────────────────────────────────


def _submit_one_file(
    *,
    cs: CodingSession,
    path: str,
    kind: str,
    submit_reason: str,
    port: ChangeRequestPort,
    backend: Any,
) -> SubmitResult:
    """Build the change-request payload for one file and dispatch it."""
    if kind == "D":
        # Deletes are out of scope for v1 — the change-request system
        # only writes content, not removes files. Operator can do it
        # manually if needed.
        return SubmitResult(
            path=path,
            change_request_id=None,
            status="refused",
            refusal_reason=(
                "delete-not-supported: the change-request system has "
                "no delete primitive in v1. To remove a file, the "
                "operator must do it manually via PR."
            ),
        )

    if kind == "?":
        return SubmitResult(
            path=path,
            change_request_id=None,
            status="refused",
            refusal_reason=f"unknown change kind for path {path!r}",
        )

    # Read the new content (worktree state)
    try:
        new_content = backend.read_worktree_file(
            worktree_path=cs.worktree_path, path=path,
        )
    except FileNotFoundError:
        # Race: file was modified-then-deleted; treat as delete refusal
        return SubmitResult(
            path=path,
            change_request_id=None,
            status="refused",
            refusal_reason=(
                f"file {path!r} disappeared from worktree during submit; "
                "treat as delete (not supported)."
            ),
        )

    # Read the old content (base sha state)
    try:
        old_content = backend.read_base_file(base_sha=cs.base_sha, path=path)
    except FileNotFoundError:
        # Added file: no base content
        old_content = ""

    # Build the per-CR reason. The operator sees this in the React UI
    # and the Signal ASK; tying it to the session id makes audit
    # forensics easier.
    full_reason = (
        f"{cs.purpose}\n\n"
        f"{submit_reason}\n\n"
        f"[from coding session {cs.id}, change kind {kind}]"
    )

    # Dispatch
    cr = port.create_request(
        requestor=cs.agent_id,
        path=path,
        new_content=new_content,
        old_content=old_content,
        reason=full_reason,
    )

    cr_status = _status_value(cr)

    # If the request landed PENDING, fire the Signal ASK. Failures
    # there are non-fatal (the request is still visible in React);
    # we log but don't change the SubmitResult shape.
    if cr_status == "pending":
        try:
            ts = port.send_ask(_id_value(cr))
            logger.debug(
                "submit: send_ask for %s returned ts=%r",
                _id_value(cr), ts,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "submit: send_ask raised for %s: %s",
                _id_value(cr), exc,
            )

    refusal = None
    if cr_status in {"tier_immutable_refused", "rejected"}:
        # Validator rejection — surface the validator's reason
        refusal = _decision_reason(cr) or "rejected at validation"

    return SubmitResult(
        path=path,
        change_request_id=_id_value(cr),
        status=cr_status,
        refusal_reason=refusal,
    )


# ── Duck-type accessors ─────────────────────────────────────────────


def _id_value(cr: Any) -> str:
    """Pull ``cr.id`` defensively (for fakes that might use a string)."""
    val = getattr(cr, "id", None)
    if val is None:
        raise AttributeError(
            f"change request {cr!r} has no .id attribute"
        )
    return str(val)


def _status_value(cr: Any) -> str:
    """Pull the change-request status as a lowercase string. The real
    ``Status`` enum is ``str``-valued so ``.value`` works; tests can
    pass plain strings too."""
    s = getattr(cr, "status", None)
    if s is None:
        raise AttributeError(
            f"change request {cr!r} has no .status attribute"
        )
    if hasattr(s, "value"):
        return str(s.value).lower()
    return str(s).lower()


def _decision_reason(cr: Any) -> str | None:
    return getattr(cr, "decision_reason", None)
