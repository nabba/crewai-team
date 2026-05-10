"""ChangeRequest data model.

A change request is the unit of agent-proposed code modification:
the agent identifies a needed write to a restricted path, and the
system runs it through a human gate (Signal 👍/👎 OR React
operator override) before applying.

Status state machine:

    PENDING ─┬─→ APPROVED ──→ APPLIED ──→ ROLLED_BACK
             │                      ╲
             ├─→ REJECTED            ╲─→ APPLY_FAILED
             ├─→ TIER_IMMUTABLE_REFUSED  (terminal at request time)
             └─→ TIMEOUT

Transitions:
  PENDING → APPROVED:    Signal 👍 OR React operator approve
  PENDING → REJECTED:    Signal 👎 OR React operator reject
  PENDING → TIMEOUT:     no decision within 10 min
  PENDING → TIER_IMMUTABLE_REFUSED:
                         path validation rejected the request
                         BEFORE going to Signal (no human can
                         override TIER_IMMUTABLE)
  APPROVED → APPLIED:    hot-apply + git auto-PR succeeded
  APPROVED → APPLY_FAILED: hot-apply or git operations failed
  APPLIED → ROLLED_BACK: operator clicked rollback in React;
                         revert commit pushed; hot-revert applied

Every transition emits an audit-log entry (hash-chained, append-
only, persisted to ``workspace/change_requests/audit.jsonl``). The
chain integrity check is part of the operator-side governance —
tampering shows up immediately.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Status(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    APPLY_FAILED = "apply_failed"
    ROLLED_BACK = "rolled_back"
    TIER_IMMUTABLE_REFUSED = "tier_immutable_refused"
    TIMEOUT = "timeout"


# Decisions that allow a transition to APPROVED. Operator React-side
# decisions are explicit overrides; Signal 👍 from the user is the
# primary path. ``SELF_HEAL_AUTO_APPLY`` denotes the auto-apply
# pathway: an allowlisted self-heal handler filed a CR with
# ``risk_class=AUTO_APPLY`` that passed the strict auto-apply
# validator and is being applied without operator approval.
class DecisionSource(str, enum.Enum):
    SIGNAL_THUMBS_UP = "signal-thumbs-up"
    SIGNAL_THUMBS_DOWN = "signal-thumbs-down"
    REACT_APPROVE = "react-approve"
    REACT_REJECT = "react-reject"
    TIMEOUT = "timeout"
    SELF_HEAL_AUTO_APPLY = "self-heal-auto-apply"


# Risk classification for a change request. ``STANDARD`` is the
# default — every CR routes through the operator gate. ``AUTO_APPLY``
# bypasses the gate after passing the strict auto-apply validator
# (``app.change_requests.validator.validate_auto_apply``):
#
#   * caller is in the auto-apply requestor allowlist
#   * patch is additive-only (no deleted lines)
#   * net line delta ≤ ``_AUTO_APPLY_LINE_CAP``
#   * target path is in the auto-apply path allowlist
#   * target path is NOT under any forbidden prefix
#     (memory/KB/migrations/souls/governance)
#
# Auto-applied CRs are loudly Signal-notified; the auto-revert
# watcher monitors the originating error pattern and rolls back if
# the pattern recurs within the watch window.
class RiskClass(str, enum.Enum):
    STANDARD = "standard"
    AUTO_APPLY = "auto_apply"


@dataclass
class ChangeRequest:
    """One agent-proposed code change. Persisted as JSON; manipulated
    via the lifecycle module."""

    # Identity + provenance
    id: str
    created_at: str        # ISO-8601 UTC
    requestor: str         # agent_id like "coder", "researcher"

    # The proposed change
    path: str              # repo-relative, e.g. "app/agents/pim_agent.py"
    new_content: str
    old_content: str       # captured at request time for diff + rollback
    reason: str            # one-paragraph explanation for the operator
    diff: str              # unified diff, computed once

    # Lifecycle
    status: Status = Status.PENDING
    decided_at: str | None = None
    decided_by: DecisionSource | None = None
    decision_reason: str | None = None  # optional rejection reason

    # Risk class (default STANDARD — operator-gated). AUTO_APPLY CRs
    # bypass the operator gate after passing the strict validator;
    # ``origin_pattern_signature`` is the error_monitor signature that
    # triggered this CR — used by the auto-revert watcher to decide
    # whether to roll back when the same pattern recurs.
    risk_class: RiskClass = RiskClass.STANDARD
    origin_pattern_signature: str | None = None

    # Application (set on APPROVED → APPLIED transition)
    git_branch: str | None = None
    git_commit_sha: str | None = None
    pr_url: str | None = None
    applied_at: str | None = None
    apply_error: str | None = None  # set on APPLY_FAILED

    # Rollback (set on APPLIED → ROLLED_BACK transition)
    rollback_commit_sha: str | None = None
    rolled_back_at: str | None = None
    rolled_back_by: str | None = None  # operator identifier
    rollback_pr_url: str | None = None

    # Signal correlation — message timestamp the ASK landed on
    signal_message_ts: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "created_at": self.created_at,
            "requestor": self.requestor,
            "path": self.path,
            "new_content": self.new_content,
            "old_content": self.old_content,
            "reason": self.reason,
            "diff": self.diff,
            "status": self.status.value,
            "risk_class": self.risk_class.value,
        }
        for opt in (
            "decided_at", "decision_reason",
            "origin_pattern_signature",
            "git_branch", "git_commit_sha", "pr_url",
            "applied_at", "apply_error",
            "rollback_commit_sha", "rolled_back_at", "rolled_back_by",
            "rollback_pr_url",
        ):
            v = getattr(self, opt)
            if v is not None:
                d[opt] = v
        if self.decided_by is not None:
            d["decided_by"] = self.decided_by.value
        if self.signal_message_ts is not None:
            d["signal_message_ts"] = self.signal_message_ts
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChangeRequest":
        decided_by_raw = data.get("decided_by")
        # ``risk_class`` defaults to STANDARD for back-compat with
        # records persisted before the field existed.
        risk_class_raw = data.get("risk_class") or RiskClass.STANDARD.value
        return cls(
            id=data["id"],
            created_at=data["created_at"],
            requestor=data["requestor"],
            path=data["path"],
            new_content=data["new_content"],
            old_content=data["old_content"],
            reason=data["reason"],
            diff=data["diff"],
            status=Status(data["status"]),
            decided_at=data.get("decided_at"),
            decided_by=DecisionSource(decided_by_raw) if decided_by_raw else None,
            decision_reason=data.get("decision_reason"),
            risk_class=RiskClass(risk_class_raw),
            origin_pattern_signature=data.get("origin_pattern_signature"),
            git_branch=data.get("git_branch"),
            git_commit_sha=data.get("git_commit_sha"),
            pr_url=data.get("pr_url"),
            applied_at=data.get("applied_at"),
            apply_error=data.get("apply_error"),
            rollback_commit_sha=data.get("rollback_commit_sha"),
            rolled_back_at=data.get("rolled_back_at"),
            rolled_back_by=data.get("rolled_back_by"),
            rollback_pr_url=data.get("rollback_pr_url"),
            signal_message_ts=data.get("signal_message_ts"),
        )

    @property
    def is_terminal(self) -> bool:
        """True if the request is in a final state (no further
        transitions possible). Operators can roll back APPLIED but
        not the other terminals."""
        return self.status in {
            Status.REJECTED,
            Status.APPLY_FAILED,
            Status.ROLLED_BACK,
            Status.TIER_IMMUTABLE_REFUSED,
            Status.TIMEOUT,
        }

    @property
    def is_rollbackable(self) -> bool:
        """Only APPLIED requests can be rolled back."""
        return self.status == Status.APPLIED
