"""Thread + SubQuestion data model.

A Thread is the persistent context for one line of inquiry that spans
multiple crew runs / days. It carries:

  - the question(s) being investigated,
  - what's resolved + what's blocking,
  - cross-references to related crew tasks and inquiry essays,
  - free-form notes the operator (or the recovery loop) can read.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class ThreadStatus(str, enum.Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


class InvalidThreadTransition(RuntimeError):
    """Raised when a state transition would violate the state machine."""


@dataclass
class SubQuestion:
    id: str
    text: str
    resolved: bool = False
    resolution: str = ""
    resolved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "resolved": self.resolved,
            "resolution": self.resolution,
            "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubQuestion":
        return cls(
            id=data["id"],
            text=data["text"],
            resolved=bool(data.get("resolved", False)),
            resolution=data.get("resolution", "") or "",
            resolved_at=data.get("resolved_at"),
        )


@dataclass
class Thread:
    id: str
    created_at: str
    title: str
    description: str

    status: ThreadStatus = ThreadStatus.OPEN
    sub_questions: list[SubQuestion] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    related_crew_task_ids: list[str] = field(default_factory=list)
    related_inquiry_slugs: list[str] = field(default_factory=list)

    last_touched_at: str = ""
    resolved_at: str | None = None
    abandoned_at: str | None = None
    abandon_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "created_at": self.created_at,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "sub_questions": [sq.to_dict() for sq in self.sub_questions],
            "blockers": list(self.blockers),
            "notes": list(self.notes),
            "related_crew_task_ids": list(self.related_crew_task_ids),
            "related_inquiry_slugs": list(self.related_inquiry_slugs),
            "last_touched_at": self.last_touched_at,
        }
        for opt in ("resolved_at", "abandoned_at", "abandon_reason"):
            v = getattr(self, opt)
            if v is not None:
                d[opt] = v
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Thread":
        return cls(
            id=data["id"],
            created_at=data["created_at"],
            title=data["title"],
            description=data["description"],
            status=ThreadStatus(data["status"]),
            sub_questions=[
                SubQuestion.from_dict(sq) for sq in data.get("sub_questions", [])
            ],
            blockers=list(data.get("blockers", [])),
            notes=list(data.get("notes", [])),
            related_crew_task_ids=list(data.get("related_crew_task_ids", [])),
            related_inquiry_slugs=list(data.get("related_inquiry_slugs", [])),
            last_touched_at=data.get("last_touched_at", ""),
            resolved_at=data.get("resolved_at"),
            abandoned_at=data.get("abandoned_at"),
            abandon_reason=data.get("abandon_reason"),
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in {ThreadStatus.RESOLVED, ThreadStatus.ABANDONED}

    @property
    def open_subquestions(self) -> list[SubQuestion]:
        return [sq for sq in self.sub_questions if not sq.resolved]

    @property
    def resolved_subquestions(self) -> list[SubQuestion]:
        return [sq for sq in self.sub_questions if sq.resolved]
