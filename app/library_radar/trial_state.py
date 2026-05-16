"""Per-discovery library-trial state ledger.

PROGRAM §46.13 (Q10.1). Sits alongside :mod:`app.proposal_bridge`'s
ProposalStore: where the proposal_bridge tracks "have we already
proposed this library?", trial_state tracks "have we already trialed
it, and did the smoke test pass?".

Schema (one row per discovery, append-only JSONL at
``workspace/library_radar/trial_state.jsonl``)::

    {
      "signature":       "<sha256[:12] from proposer>",
      "slug":            "<slug from discovery title>",
      "package_name":    "<candidate package picked for trial>",
      "candidates":      ["pkg_a", "pkg_b", ...],
      "status":          "pending|running|passed|failed|adoption_cr_filed",
      "last_run_at":     "2026-05-16T10:00:00+00:00",
      "session_id":      "cs_xxx",            # populated once a session starts
      "pytest_exit":     0,                    # populated once tested
      "trial_error":     "<short reason>",     # populated on failure
      "adoption_cr_id":  "cr_xxx",             # populated when adoption CR is filed
      "ts":              "<append timestamp>"
    }

Reads compact the JSONL into a dict-of-latest-by-signature. Writes
append a new row (audit trail preserved). The compaction is
deterministic — the on-disk file is the source of truth.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_VALID_STATUSES = frozenset({
    "pending",
    "running",
    "passed",
    "failed",
    "adoption_cr_filed",
    "adoption_cr_rejected",
})

_LOCK = threading.RLock()


@dataclass
class TrialState:
    signature: str
    slug: str
    package_name: str = ""
    candidates: list[str] = field(default_factory=list)
    status: str = "pending"
    last_run_at: str = ""
    session_id: str = ""
    pytest_exit: int | None = None
    trial_error: str = ""
    adoption_cr_id: str = ""
    ts: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Preserve insertion order for readability; serialize None as null.
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrialState":
        return cls(
            signature=str(data.get("signature", "")),
            slug=str(data.get("slug", "")),
            package_name=str(data.get("package_name", "") or ""),
            candidates=list(data.get("candidates", []) or []),
            status=str(data.get("status", "pending") or "pending"),
            last_run_at=str(data.get("last_run_at", "") or ""),
            session_id=str(data.get("session_id", "") or ""),
            pytest_exit=(
                int(data["pytest_exit"])
                if data.get("pytest_exit") is not None
                else None
            ),
            trial_error=str(data.get("trial_error", "") or ""),
            adoption_cr_id=str(data.get("adoption_cr_id", "") or ""),
            ts=str(data.get("ts", "") or ""),
        )


def _state_path() -> Path:
    base = Path(os.environ.get("WORKSPACE_ROOT", "/app/workspace"))
    d = base / "library_radar"
    d.mkdir(parents=True, exist_ok=True)
    return d / "trial_state.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append(state: TrialState) -> None:
    """Append a row. Append-only; deduplicated by signature on read."""
    if state.status not in _VALID_STATUSES:
        raise ValueError(
            f"invalid status {state.status!r}; "
            f"must be one of {sorted(_VALID_STATUSES)}"
        )
    if not state.ts:
        state.ts = _now_iso()
    path = _state_path()
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(state.to_dict(), sort_keys=True) + "\n")


def read_latest() -> dict[str, TrialState]:
    """Return the latest row per signature (later writes win)."""
    path = _state_path()
    if not path.exists():
        return {}
    by_sig: dict[str, TrialState] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                try:
                    state = TrialState.from_dict(row)
                except Exception:
                    continue
                if not state.signature:
                    continue
                by_sig[state.signature] = state
    except OSError:
        return {}
    return by_sig


def get(signature: str) -> TrialState | None:
    return read_latest().get(signature)


def list_pending(*, limit: int = 50) -> list[TrialState]:
    """Return discoveries that need a trial run. Excludes any already
    in a terminal state (passed, failed, adoption_cr_filed,
    adoption_cr_rejected)."""
    rows = list(read_latest().values())
    pending = [
        s for s in rows
        if s.status in ("pending", "running")
    ]
    pending.sort(key=lambda s: s.ts)
    return pending[:limit]


def mark_pending(
    *, signature: str, slug: str, candidates: list[str],
) -> TrialState:
    """Idempotent: only writes a pending row when none exists for the
    signature. Returns the latest state row."""
    existing = get(signature)
    if existing is not None:
        return existing
    package = candidates[0] if candidates else ""
    state = TrialState(
        signature=signature,
        slug=slug,
        package_name=package,
        candidates=list(candidates),
        status="pending",
        ts=_now_iso(),
    )
    append(state)
    logger.info(
        "library_radar.trial: marked %s pending (slug=%s, pkg=%s)",
        signature, slug, package,
    )
    return state


def mark_running(signature: str, *, session_id: str) -> None:
    existing = get(signature)
    if existing is None:
        return
    existing.status = "running"
    existing.session_id = session_id
    existing.last_run_at = _now_iso()
    existing.ts = existing.last_run_at
    append(existing)


def mark_passed(
    signature: str, *, pytest_exit: int = 0,
) -> None:
    existing = get(signature)
    if existing is None:
        return
    existing.status = "passed"
    existing.pytest_exit = pytest_exit
    existing.last_run_at = _now_iso()
    existing.ts = existing.last_run_at
    existing.trial_error = ""
    append(existing)


def mark_failed(signature: str, *, error: str, pytest_exit: int | None = None) -> None:
    existing = get(signature)
    if existing is None:
        return
    existing.status = "failed"
    existing.pytest_exit = pytest_exit
    existing.last_run_at = _now_iso()
    existing.ts = existing.last_run_at
    existing.trial_error = (error or "")[:500]
    append(existing)


def mark_adoption_filed(signature: str, *, cr_id: str) -> None:
    existing = get(signature)
    if existing is None:
        return
    existing.status = "adoption_cr_filed"
    existing.adoption_cr_id = cr_id
    existing.last_run_at = _now_iso()
    existing.ts = existing.last_run_at
    append(existing)


def summarise() -> dict[str, int]:
    """Counts by status — operator visibility surface."""
    counts: dict[str, int] = {s: 0 for s in _VALID_STATUSES}
    for state in read_latest().values():
        counts[state.status] = counts.get(state.status, 0) + 1
    return counts


def reset_for_tests() -> None:
    """Test helper: wipe the on-disk ledger."""
    path = _state_path()
    with _LOCK:
        if path.exists():
            path.unlink()
