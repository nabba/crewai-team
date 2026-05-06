"""CodingSession data model.

A coding session is the unit of an agent's iterative code work: it owns
an ephemeral git worktree where the agent can read, write, and run
commands freely. The only escape hatch from the worktree is
``coding_session_submit``, which routes through the change-request
human gate.

State machine::

    ACTIVE ─┬─→ SUBMITTED   (agent called submit; worktree destroyed)
            ├─→ DISCARDED   (agent called discard; worktree destroyed)
            ├─→ EXPIRED     (TTL or idle timeout; reconciler killed it)
            └─→ FAILED      (worktree corruption; retained for forensics)

Transitions:
  ACTIVE → SUBMITTED:   submit() — bundles diff into change requests
  ACTIVE → DISCARDED:   discard() — agent gave up
  ACTIVE → EXPIRED:     reconciler — past TTL or idle for too long
  ACTIVE → FAILED:      fail() — git worktree corrupt; keep for postmortem

Every transition emits an audit-log entry (hash-chained, append-only,
persisted to ``workspace/coding_sessions/audit.jsonl``). Mirrors the
Forge / change-requests audit-log discipline.

A session in any non-ACTIVE state is read-only — no further tool
calls succeed against it. The agent cannot resume; a second iteration
is a fresh session.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Status(str, enum.Enum):
    ACTIVE = "active"
    SUBMITTED = "submitted"
    DISCARDED = "discarded"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class SubmitResult:
    """One row in CodingSession.submit_results — one per file the
    agent touched at submit time."""

    path: str
    change_request_id: str | None  # None if refused (TIER_IMMUTABLE / validator)
    status: str                    # change-request status string
    refusal_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "path": self.path,
            "change_request_id": self.change_request_id,
            "status": self.status,
        }
        if self.refusal_reason is not None:
            d["refusal_reason"] = self.refusal_reason
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubmitResult":
        return cls(
            path=data["path"],
            change_request_id=data.get("change_request_id"),
            status=data["status"],
            refusal_reason=data.get("refusal_reason"),
        )


@dataclass
class CodingSession:
    """One agent's iteration sandbox. Persisted as JSON; manipulated
    via the manager module."""

    # ── Identity + provenance ──────────────────────────────────────
    id: str
    agent_id: str
    purpose: str               # one-paragraph statement from the agent
    created_at: str            # ISO-8601 UTC

    # ── Base + worktree location ───────────────────────────────────
    base: str                  # branch/ref name, e.g. "main"
    base_sha: str              # locked commit sha at session start
    worktree_path: str         # absolute path on host (under /tmp/agent-sessions/)

    # ── Lifecycle ──────────────────────────────────────────────────
    expires_at: str            # created_at + TTL
    last_activity_at: str      # for idle-timeout detection
    status: Status = Status.ACTIVE

    # ── Tracking (counters set during ACTIVE phase) ────────────────
    files_touched: list[str] = field(default_factory=list)
    run_count: int = 0
    bytes_written: int = 0

    # ── Terminal metadata (set on transition out of ACTIVE) ────────
    terminated_at: str | None = None
    terminated_reason: str | None = None       # short label
    submit_results: list[SubmitResult] | None = None

    # ── Predicates ─────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self.status is Status.ACTIVE

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            Status.SUBMITTED,
            Status.DISCARDED,
            Status.EXPIRED,
            Status.FAILED,
        }

    # ── Serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "agent_id": self.agent_id,
            "purpose": self.purpose,
            "created_at": self.created_at,
            "base": self.base,
            "base_sha": self.base_sha,
            "worktree_path": self.worktree_path,
            "expires_at": self.expires_at,
            "last_activity_at": self.last_activity_at,
            "status": self.status.value,
            "files_touched": list(self.files_touched),
            "run_count": self.run_count,
            "bytes_written": self.bytes_written,
        }
        if self.terminated_at is not None:
            d["terminated_at"] = self.terminated_at
        if self.terminated_reason is not None:
            d["terminated_reason"] = self.terminated_reason
        if self.submit_results is not None:
            d["submit_results"] = [r.to_dict() for r in self.submit_results]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodingSession":
        sub_raw = data.get("submit_results")
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            purpose=data["purpose"],
            created_at=data["created_at"],
            base=data["base"],
            base_sha=data["base_sha"],
            worktree_path=data["worktree_path"],
            expires_at=data["expires_at"],
            last_activity_at=data["last_activity_at"],
            status=Status(data["status"]),
            files_touched=list(data.get("files_touched") or []),
            run_count=int(data.get("run_count") or 0),
            bytes_written=int(data.get("bytes_written") or 0),
            terminated_at=data.get("terminated_at"),
            terminated_reason=data.get("terminated_reason"),
            submit_results=(
                [SubmitResult.from_dict(r) for r in sub_raw]
                if sub_raw is not None
                else None
            ),
        )
