"""ActionRequest data model.

A unit of agent-proposed *external action*. The agent names an
action_type (email_draft / calendar_invite / ...) and a
type-specific data payload; the operator approves; the type's
handler applies.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class ActionType(str, enum.Enum):
    """Supported action types. Add new entries when handlers ship."""

    EMAIL_DRAFT = "email_draft"
    # Q9.5 (PROGRAM §46.8, 2026-05-16) — two new outbound surfaces:
    CALENDAR_INVITE = "calendar_invite"
    SIGNAL_SEND = "signal_send"
    # Future: SLACK_MESSAGE, DISCORD_DM, FILE_UPLOAD, ...


class ActionStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    APPLIED = "applied"
    APPLY_FAILED = "apply_failed"
    REJECTED = "rejected"
    INVALID = "invalid"  # validator rejected before reaching the gate
    TIMEOUT = "timeout"


class DecisionSource(str, enum.Enum):
    SIGNAL_THUMBS_UP = "signal-thumbs-up"
    SIGNAL_THUMBS_DOWN = "signal-thumbs-down"
    REACT_APPROVE = "react-approve"
    REACT_REJECT = "react-reject"
    TIMEOUT = "timeout"


@dataclass
class ActionRequest:
    """One agent-proposed external action."""

    id: str
    created_at: str
    requestor: str

    action_type: ActionType
    summary: str           # one-line description shown in Signal + React
    data: dict[str, Any]   # type-specific payload (e.g. email {to,subject,body})
    reason: str            # operator-facing motivation

    status: ActionStatus = ActionStatus.PENDING
    decided_at: str | None = None
    decided_by: DecisionSource | None = None
    decision_reason: str | None = None

    # Apply step
    applied_at: str | None = None
    apply_error: str | None = None
    apply_artifact: dict[str, Any] = field(default_factory=dict)
    # ^ handler-specific receipt (e.g. {message_id: ...} for email)

    # Validation failure (terminal pre-gate)
    invalid_reason: str | None = None

    # Signal correlation
    signal_message_ts: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "created_at": self.created_at,
            "requestor": self.requestor,
            "action_type": self.action_type.value,
            "summary": self.summary,
            "data": dict(self.data),
            "reason": self.reason,
            "status": self.status.value,
            "apply_artifact": dict(self.apply_artifact),
        }
        for opt in (
            "decided_at", "decision_reason",
            "applied_at", "apply_error",
            "invalid_reason",
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
    def from_dict(cls, data: dict[str, Any]) -> "ActionRequest":
        decided_by_raw = data.get("decided_by")
        return cls(
            id=data["id"],
            created_at=data["created_at"],
            requestor=data["requestor"],
            action_type=ActionType(data["action_type"]),
            summary=data["summary"],
            data=dict(data.get("data", {})),
            reason=data["reason"],
            status=ActionStatus(data["status"]),
            decided_at=data.get("decided_at"),
            decided_by=DecisionSource(decided_by_raw) if decided_by_raw else None,
            decision_reason=data.get("decision_reason"),
            applied_at=data.get("applied_at"),
            apply_error=data.get("apply_error"),
            apply_artifact=dict(data.get("apply_artifact", {})),
            invalid_reason=data.get("invalid_reason"),
            signal_message_ts=data.get("signal_message_ts"),
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            ActionStatus.APPLIED,
            ActionStatus.APPLY_FAILED,
            ActionStatus.REJECTED,
            ActionStatus.INVALID,
            ActionStatus.TIMEOUT,
        }

    @property
    def is_decided(self) -> bool:
        return self.status is not ActionStatus.PENDING
