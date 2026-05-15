"""Control plane — read-only coding-session endpoints at
/api/cp/coding-sessions.

Phase 5.4-f. Operators see what coding sessions are active or
recently terminal. Lifecycle is owned by the agent + reconciler;
the operator does NOT have an approve/reject button — by design.
The decision is "did the session submit a sensible change?", and
that decision happens in the change-request UI (#55), not here.

Endpoints:

  GET  /api/cp/coding-sessions                 list (status + agent filter)
  GET  /api/cp/coding-sessions/{id}            detail

Auth: same ``require_gateway_auth`` dependency as the rest of /cp/.

Why no POST routes:
  * Worktree creation is the agent's responsibility (via
    ``coding_session_start``).
  * Submission is the agent's responsibility (via
    ``coding_session_submit``).
  * Termination is the agent's responsibility (via
    ``coding_session_discard``) or the reconciler's (TTL/idle).

If the operator wants to forcibly terminate a runaway session,
the path is "wait for the reconciler" or "kill via Postgres" —
keeping the surface read-only stops the operator from accidentally
discarding a session the agent is still actively iterating on.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.coding_session import Status, store as cs_store
from app.control_plane.auth_dep import require_gateway_auth

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/cp/coding-sessions",
    tags=["control-plane", "coding-sessions"],
    dependencies=[Depends(require_gateway_auth)],
)


# ── Helpers ─────────────────────────────────────────────────────────


def _serialize(cs) -> dict[str, Any]:
    """Serialize a CodingSession for the React UI. Uses ``to_dict``
    + adds derived predicates that are awkward to compute client-
    side."""
    if cs is None:
        return {}
    d = cs.to_dict()
    d["is_active"] = cs.is_active
    d["is_terminal"] = cs.is_terminal
    return d


# ── Routes ──────────────────────────────────────────────────────────


@router.get("")
def list_coding_sessions(
    status: str | None = Query(
        default=None,
        description=(
            "Filter by status (active, submitted, discarded, expired, "
            "failed)."
        ),
    ),
    agent_id: str | None = Query(
        default=None,
        description="Filter by agent_id (e.g. 'coder').",
    ),
    limit: int = Query(default=200, ge=1, le=500),
):
    """List coding sessions, newest first, optionally filtered."""
    status_enum: Status | None = None
    if status:
        try:
            status_enum = Status(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"invalid status {status!r}. Valid: "
                    f"{[s.value for s in Status]}"
                ),
            )
    items = cs_store.list_all(
        status=status_enum, agent_id=agent_id, limit=limit,
    )
    return {
        "count": len(items),
        "sessions": [_serialize(s) for s in items],
    }


@router.get("/{session_id}")
def get_coding_session(session_id: str):
    cs = cs_store.get(session_id)
    if cs is None:
        raise HTTPException(
            status_code=404,
            detail=f"coding session {session_id!r} not found",
        )
    return _serialize(cs)


# ── Q7.4 — Inline ShinkaEvolve runs per session ──────────────────────


@router.get("/{session_id}/evolution_runs")
def get_session_evolution_runs(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=200),
):
    """Return the per-session ShinkaEvolve audit trail.

    Newest-first list of evolution runs against this session (one row
    per ``coding_session_evolve_solution`` call). The audit lives
    outside the worktree at
    ``workspace/coding_sessions/<id>/evolution_audit.jsonl`` so it
    survives session cleanup.

    Returns 404 only if the SESSION doesn't exist. Sessions that have
    never invoked evolution return ``{"summary": {n_runs: 0,...},
    "runs": []}``.
    """
    cs = cs_store.get(session_id)
    if cs is None:
        raise HTTPException(
            status_code=404,
            detail=f"coding session {session_id!r} not found",
        )
    try:
        from app.coding_session.evolution_audit import (
            read_runs, session_summary,
        )
    except Exception:
        return {
            "session_id": session_id,
            "summary": {"n_runs": 0, "by_status": {}},
            "runs": [],
        }
    return {
        "session_id": session_id,
        "summary": session_summary(session_id),
        "runs": read_runs(session_id, limit=limit),
    }
