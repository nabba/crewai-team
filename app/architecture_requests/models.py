"""ArchitectureRequest data model.

A unit of agent-proposed *subsystem* addition. The agent names a
package path, a list of files with their purposes, a list of
integration points (idle-job registrations, env switches, tool
registrations, signal commands), and a test plan. The operator
approves the *design*; implementation then flows through the
existing :mod:`app.change_requests` gate per file.

Status state machine::

    PROPOSED ─┬─→ APPROVED ─→ SCAFFOLDED ─→ IMPLEMENTING ─→ COMPLETED
              │                       ╲              ╲
              ├─→ REJECTED             ╲              ╲─→ ABANDONED
              ├─→ TIER_IMMUTABLE_REFUSED
              └─→ TIMEOUT

Transitions:
  PROPOSED   → APPROVED:    operator 👍 OR React approve
  PROPOSED   → REJECTED:    operator 👎 OR React reject
  PROPOSED   → TIMEOUT:     no decision within decision window
  PROPOSED   → TIER_IMMUTABLE_REFUSED:
                            validator rejected before reaching the
                            human gate; no override possible
  APPROVED   → SCAFFOLDED:  scaffolder wrote stubs to staging dir
  SCAFFOLDED → IMPLEMENTING: at least one child change-request filed
  IMPLEMENTING → COMPLETED:  all child change-requests are APPLIED
  SCAFFOLDED  → ABANDONED:   timeout or explicit abandon
  IMPLEMENTING→ ABANDONED:   same
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


VALID_INTEGRATION_KINDS = frozenset({
    "idle_job_registration",
    "tool_registration",
    "signal_command",
})


class ArchStatus(str, enum.Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    SCAFFOLDED = "scaffolded"
    IMPLEMENTING = "implementing"
    COMPLETED = "completed"
    REJECTED = "rejected"
    TIER_IMMUTABLE_REFUSED = "tier_immutable_refused"
    TIMEOUT = "timeout"
    ABANDONED = "abandoned"


class DecisionSource(str, enum.Enum):
    SIGNAL_THUMBS_UP = "signal-thumbs-up"
    SIGNAL_THUMBS_DOWN = "signal-thumbs-down"
    REACT_APPROVE = "react-approve"
    REACT_REJECT = "react-reject"
    TIMEOUT = "timeout"


@dataclass
class FileSpec:
    """One file the proposed package should contain."""

    path: str
    purpose: str
    initial_stub: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "purpose": self.purpose,
            "initial_stub": self.initial_stub,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileSpec":
        return cls(
            path=data["path"],
            purpose=data["purpose"],
            initial_stub=data.get("initial_stub", ""),
        )


@dataclass
class IntegrationPoint:
    """An external module the proposed package integrates with.

    ``kind`` must be one of :data:`VALID_INTEGRATION_KINDS`. The
    ``target_module`` field names the file the integration would
    write to — actual writes go through change_requests.
    """

    kind: str
    target_module: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target_module": self.target_module,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntegrationPoint":
        return cls(
            kind=data["kind"],
            target_module=data["target_module"],
            detail=data.get("detail", {}),
        )


@dataclass
class ArchitectureRequest:
    """One agent-proposed subsystem addition."""

    id: str
    created_at: str
    requestor: str

    intent: str
    motivation: str
    package_path: str
    file_layout: list[FileSpec]
    integration_points: list[IntegrationPoint]
    env_switches: dict[str, str]
    test_plan: str

    status: ArchStatus = ArchStatus.PROPOSED
    decided_at: str | None = None
    decided_by: DecisionSource | None = None
    decision_reason: str | None = None

    scaffolded_at: str | None = None
    scaffold_dir: str | None = None

    child_change_request_ids: list[str] = field(default_factory=list)
    completed_at: str | None = None
    abandoned_at: str | None = None
    abandon_reason: str | None = None

    signal_message_ts: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "created_at": self.created_at,
            "requestor": self.requestor,
            "intent": self.intent,
            "motivation": self.motivation,
            "package_path": self.package_path,
            "file_layout": [fs.to_dict() for fs in self.file_layout],
            "integration_points": [ip.to_dict() for ip in self.integration_points],
            "env_switches": dict(self.env_switches),
            "test_plan": self.test_plan,
            "status": self.status.value,
            "child_change_request_ids": list(self.child_change_request_ids),
        }
        for opt in (
            "decided_at", "decision_reason",
            "scaffolded_at", "scaffold_dir",
            "completed_at", "abandoned_at", "abandon_reason",
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
    def from_dict(cls, data: dict[str, Any]) -> "ArchitectureRequest":
        decided_by_raw = data.get("decided_by")
        return cls(
            id=data["id"],
            created_at=data["created_at"],
            requestor=data["requestor"],
            intent=data["intent"],
            motivation=data["motivation"],
            package_path=data["package_path"],
            file_layout=[FileSpec.from_dict(d) for d in data.get("file_layout", [])],
            integration_points=[
                IntegrationPoint.from_dict(d)
                for d in data.get("integration_points", [])
            ],
            env_switches=dict(data.get("env_switches", {})),
            test_plan=data["test_plan"],
            status=ArchStatus(data["status"]),
            decided_at=data.get("decided_at"),
            decided_by=DecisionSource(decided_by_raw) if decided_by_raw else None,
            decision_reason=data.get("decision_reason"),
            scaffolded_at=data.get("scaffolded_at"),
            scaffold_dir=data.get("scaffold_dir"),
            child_change_request_ids=list(data.get("child_change_request_ids", [])),
            completed_at=data.get("completed_at"),
            abandoned_at=data.get("abandoned_at"),
            abandon_reason=data.get("abandon_reason"),
            signal_message_ts=data.get("signal_message_ts"),
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            ArchStatus.REJECTED,
            ArchStatus.TIER_IMMUTABLE_REFUSED,
            ArchStatus.TIMEOUT,
            ArchStatus.COMPLETED,
            ArchStatus.ABANDONED,
        }

    @property
    def is_decided(self) -> bool:
        return self.status is not ArchStatus.PROPOSED
