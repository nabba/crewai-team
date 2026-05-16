"""Brainstorm session — dataclass + serialization."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from app.brainstorm.techniques.base import TechniqueState

SessionStatus = Literal["active", "paused", "complete", "cancelled"]
SessionMode = Literal["solo", "team"]


@dataclass
class BrainstormSession:
    """One brainstorming session for one user.

    ``sender`` is the originating identifier — a Signal phone number, "cli",
    or any opaque caller ID. ``technique`` is the technique short name (e.g.
    ``"scamper"``). ``technique_state`` carries the per-technique state
    machine. ``transcript`` is a flat append-only log of every user/system
    turn for audit and report generation.

    Multi-agent (team) mode adds:
      - ``participants``: agent role names participating (in addition to user)
      - ``mode``: "solo" or "team"
      - ``agent_rounds``: list of {step_id, phase, responses[]} entries — one
        per gathered seed/react round, for replay and report generation.
    """

    session_id: str
    sender: str
    topic: str
    technique: str
    technique_state: TechniqueState = field(default_factory=TechniqueState)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    status: SessionStatus = "active"
    mode: SessionMode = "solo"
    participants: list[str] = field(default_factory=list)
    agent_rounds: list[dict[str, Any]] = field(default_factory=list)
    # Q11.1 (PROGRAM §46.18) — cross-domain analogues surfaced from
    # the analogy_index at session start. Empty when the index is
    # empty / disabled / no match crossed the similarity threshold.
    # Each entry: {signature, description, examples: [...]}.
    analogues: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    final_report_path: str | None = None
    final_report: str | None = None

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]

    def append_turn(
        self,
        role: str,
        content: str,
        *,
        participant: str | None = None,
        phase: str | None = None,
    ) -> None:
        """Add one turn to the transcript and bump ``updated_at``.

        ``participant`` is the agent role for ``role="agent"`` turns;
        ``phase`` is "seed" or "react" for those turns.
        """
        turn: dict[str, Any] = {"role": role, "content": content, "ts": time.time()}
        if participant is not None:
            turn["participant"] = participant
        if phase is not None:
            turn["phase"] = phase
        self.transcript.append(turn)
        self.updated_at = time.time()

    def record_agent_round(
        self, *, step_id: str, phase: str, responses: list[dict[str, Any]]
    ) -> None:
        """Persist one batch of agent responses for later report generation."""
        self.agent_rounds.append(
            {
                "step_id": step_id,
                "phase": phase,
                "ts": time.time(),
                "responses": list(responses),
            }
        )
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "sender": self.sender,
            "topic": self.topic,
            "technique": self.technique,
            "technique_state": self.technique_state.to_dict(),
            "transcript": list(self.transcript),
            "status": self.status,
            "mode": self.mode,
            "participants": list(self.participants),
            "agent_rounds": list(self.agent_rounds),
            "analogues": list(self.analogues),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "final_report_path": self.final_report_path,
            "final_report": self.final_report,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrainstormSession":
        ts_raw = data.get("technique_state") or {}
        return cls(
            session_id=str(data["session_id"]),
            sender=str(data["sender"]),
            topic=str(data.get("topic", "")),
            technique=str(data.get("technique", "")),
            technique_state=TechniqueState.from_dict(ts_raw),
            transcript=list(data.get("transcript", [])),
            status=data.get("status", "active"),
            mode=data.get("mode", "solo"),
            participants=list(data.get("participants", [])),
            agent_rounds=list(data.get("agent_rounds", [])),
            # Q11.1 — backward-compat: pre-§46.18 sessions have no
            # ``analogues`` key, default to empty list
            analogues=list(data.get("analogues", []) or []),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            final_report_path=data.get("final_report_path"),
            final_report=data.get("final_report"),
        )
