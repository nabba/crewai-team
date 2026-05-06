"""Technique base — abstract state machine for one brainstorming method.

A technique drives a sequence of prompts (steps). For each step the
facilitator shows ``next_prompt(state, topic)`` to the user, captures the
reply via ``record_response(state, reply)``, and repeats until
``is_complete(state)``. Then ``summarize(state, topic)`` returns a structured
payload for the Writer agent.

Most techniques are linear — see :class:`LinearTechnique`. Techniques with
branching can subclass :class:`Technique` directly.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class Step:
    """One prompt in a technique. ``prompt`` may use ``{topic}`` template."""

    step_id: str
    prompt: str
    expected_output: str = ""


@dataclass
class TechniqueState:
    """Mutable state for a running technique."""

    step_index: int = 0
    responses: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "responses": list(self.responses),
            "extras": dict(self.extras),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TechniqueState":
        return cls(
            step_index=int(data.get("step_index", 0)),
            responses=list(data.get("responses", [])),
            extras=dict(data.get("extras", {})),
        )


class Technique(ABC):
    """Abstract base. Concrete techniques set ``name``, ``title``,
    ``description`` and override the abstract methods."""

    name: ClassVar[str]
    title: ClassVar[str]
    description: ClassVar[str]

    def initial_state(self) -> TechniqueState:
        return TechniqueState()

    @abstractmethod
    def next_prompt(self, state: TechniqueState, topic: str) -> str | None:
        """Return the next prompt to present, or None when done."""

    @abstractmethod
    def is_complete(self, state: TechniqueState) -> bool:
        ...

    def record_response(
        self,
        state: TechniqueState,
        response: str,
        *,
        prompt: str | None = None,
        step_id: str | None = None,
    ) -> TechniqueState:
        """Record a user response and advance the state machine.

        Subclasses can override to add branching logic. Default behaviour:
        append to ``responses`` and bump ``step_index``.
        """
        state.responses.append(
            {
                "step_id": step_id or f"step_{state.step_index}",
                "prompt": prompt or "",
                "response": response,
                "ts": time.time(),
            }
        )
        state.step_index += 1
        return state

    @abstractmethod
    def summarize(self, state: TechniqueState, topic: str) -> dict[str, Any]:
        """Structured summary of the session for the Writer agent."""

    def total_steps(self) -> int | None:
        """Optional: total expected steps, for progress display. None = unknown."""
        return None


class LinearTechnique(Technique):
    """Common case: a fixed sequence of steps walked in order."""

    steps: ClassVar[list[Step]] = []

    def next_prompt(self, state: TechniqueState, topic: str) -> str | None:
        if state.step_index >= len(self.steps):
            return None
        return self.steps[state.step_index].prompt.format(topic=topic)

    def is_complete(self, state: TechniqueState) -> bool:
        return state.step_index >= len(self.steps)

    def record_response(
        self,
        state: TechniqueState,
        response: str,
        *,
        prompt: str | None = None,
        step_id: str | None = None,
    ) -> TechniqueState:
        idx = state.step_index
        step = self.steps[idx] if idx < len(self.steps) else None
        return super().record_response(
            state,
            response,
            prompt=prompt,
            step_id=step_id or (step.step_id if step else None),
        )

    def summarize(self, state: TechniqueState, topic: str) -> dict[str, Any]:
        return {
            "technique": self.name,
            "title": self.title,
            "topic": topic,
            "steps": [
                {
                    "step_id": r.get("step_id", ""),
                    "prompt": r.get("prompt", ""),
                    "response": r.get("response", ""),
                }
                for r in state.responses
            ],
        }

    def total_steps(self) -> int:
        return len(self.steps)
