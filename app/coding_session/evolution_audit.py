"""Per-session ShinkaEvolve audit persistence.

PROGRAM §45.4 — Q7.4. The :mod:`evolution_bridge` writes ShinkaEvolve
intermediate state under ``<worktree>/.shinka_inline/<ts>/`` — but the
worktree is cleaned up when the session terminates. The audit trail
of WHICH evolutions ran, with WHAT result, would vanish.

This module persists a per-session JSONL **outside** the worktree at::

    workspace/coding_sessions/<session_id>/evolution_audit.jsonl

One row per ``evolve_in_session`` call. Rows are minimal — just the
provenance + counters + a ``diff_sha256`` hash of the unified diff
(not the diff itself; that's the agent's responsibility to apply via
``coding_session_write`` + ``coding_session_submit``). The full diff
is reconstructable from the post-submit CR audit log if and when the
agent submits it.

Design properties:
  * **Append-only** — never rewrites prior rows.
  * **Failure-isolated** — append errors swallowed so a broken FS
    layer doesn't fail the bridge.
  * **Worktree-independent** — the audit lives under ``workspace/``,
    NOT inside the worktree, so it survives session cleanup.
  * **Capped** — last 200 rows per session. Older rows get rotated
    out via ``app.utils.jsonl_retention.append_with_cap`` (already
    used by other audit ledgers in the codebase).

Reader entry point is ``read_runs(session_id, limit=50)`` returning a
list of dicts in newest-first order — that's what the REST endpoint
and React panel consume.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_MAX_ROWS_PER_SESSION = 200


def _workspace_root() -> Path:
    """Resolve the workspace root, honouring ``WORKSPACE_ROOT`` env override."""
    env = os.environ.get("WORKSPACE_ROOT")
    if env:
        return Path(env)
    return Path("/app/workspace")


def _audit_path(session_id: str) -> Path:
    """Path to the per-session evolution audit JSONL.

    Path-safe: rejects session ids that contain path separators (we
    use the id as a directory component).
    """
    if not session_id or "/" in session_id or ".." in session_id.split("/"):
        raise ValueError(f"unsafe session_id: {session_id!r}")
    return (
        _workspace_root()
        / "coding_sessions"
        / session_id
        / "evolution_audit.jsonl"
    )


def append_run(
    *,
    session_id: str,
    agent_id: str,
    initial_path: str,
    evaluate_path: str,
    num_generations: int,
    num_islands: int,
    max_cost_usd: float,
    result: Any,  # ``EvolutionResult`` — kept loose to avoid circular import
) -> bool:
    """Append one evolution-run row to the per-session audit JSONL.

    Returns True on success, False on any error (failure-isolated —
    the bridge keeps running).
    """
    try:
        path = _audit_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "agent_id": agent_id,
            "initial_path": initial_path,
            "evaluate_path": evaluate_path,
            "num_generations": num_generations,
            "num_islands": num_islands,
            "max_cost_usd": max_cost_usd,
            "status": getattr(result, "status", "unknown"),
            "baseline_score": float(getattr(result, "baseline_score", 0.0) or 0.0),
            "best_score": float(getattr(result, "best_score", 0.0) or 0.0),
            "delta": float(getattr(result, "delta", 0.0) or 0.0),
            "generations_run": int(getattr(result, "generations_run", 0) or 0),
            "variants_evaluated": int(getattr(result, "variants_evaluated", 0) or 0),
            "duration_seconds": float(getattr(result, "duration_seconds", 0.0) or 0.0),
            "diff_sha256": _hash_diff(getattr(result, "diff", "") or ""),
            "diff_length": len(getattr(result, "diff", "") or ""),
            "error": str(getattr(result, "error", "") or ""),
            "refusal_reason": str(getattr(result, "refusal_reason", "") or ""),
        }
        try:
            # Prefer the canonical capped-append helper if available.
            from app.utils.jsonl_retention import append_with_cap
            append_with_cap(path, row, max_lines=_MAX_ROWS_PER_SESSION)
        except Exception:
            # Fallback: naive append; OK for unit tests + degraded paths.
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        return True
    except Exception:  # noqa: BLE001
        logger.debug(
            "evolution_audit: append failed for session %s",
            session_id, exc_info=True,
        )
        return False


def read_runs(session_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent evolution runs for one session.

    Newest first. ``limit`` is clamped to ``[1, _MAX_ROWS_PER_SESSION]``.
    Returns an empty list when the file doesn't exist or is unreadable.
    """
    limit = max(1, min(limit, _MAX_ROWS_PER_SESSION))
    try:
        path = _audit_path(session_id)
    except ValueError:
        return []
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    # Walk newest first (tail of file).
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(rows) >= limit:
            break
    return rows


def session_summary(session_id: str) -> dict[str, Any]:
    """One-shot rollup for the React panel and REST endpoint.

    Computes counts (total / improved / no_improvement / refused /
    error), aggregate cost cap budgeted, best delta seen, last run
    timestamp. Returns zeros when no runs.
    """
    runs = read_runs(session_id, limit=_MAX_ROWS_PER_SESSION)
    if not runs:
        return {
            "n_runs": 0,
            "by_status": {},
            "best_delta": 0.0,
            "total_max_cost_usd": 0.0,
            "total_duration_seconds": 0.0,
            "last_run_at": None,
        }
    by_status: dict[str, int] = {}
    best_delta = 0.0
    total_max_cost = 0.0
    total_duration = 0.0
    for r in runs:
        st = str(r.get("status", "unknown"))
        by_status[st] = by_status.get(st, 0) + 1
        try:
            d = float(r.get("delta", 0.0) or 0.0)
        except (TypeError, ValueError):
            d = 0.0
        if d > best_delta:
            best_delta = d
        try:
            total_max_cost += float(r.get("max_cost_usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
        try:
            total_duration += float(r.get("duration_seconds", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
    return {
        "n_runs": len(runs),
        "by_status": by_status,
        "best_delta": best_delta,
        "total_max_cost_usd": total_max_cost,
        "total_duration_seconds": total_duration,
        # Runs are newest first → first row's ts is the most recent.
        "last_run_at": runs[0].get("ts"),
    }


def _hash_diff(diff: str) -> str:
    """SHA-256 hex digest of the unified diff. Empty diff → empty string."""
    if not diff:
        return ""
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()
