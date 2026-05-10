"""Abstract base class + registry for action handlers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ApplyResult:
    """Outcome of executing a handler's apply step."""

    ok: bool
    artifact: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class ActionHandler(ABC):
    """Per-action-type contract.

    Subclasses provide validate / apply / render_summary. The handler
    registry dispatches by ``action_type``.
    """

    @property
    @abstractmethod
    def action_type(self):
        """The ``ActionType`` enum value this handler implements."""

    @abstractmethod
    def validate(self, data: dict[str, Any]) -> tuple[bool, str | None]:
        """Return (is_valid, reason). Reason is None on success."""

    @abstractmethod
    def apply(self, data: dict[str, Any]) -> ApplyResult:
        """Execute the action. ApplyResult.ok=False signals failure."""

    @abstractmethod
    def render_summary(self, data: dict[str, Any]) -> str:
        """Short human-readable summary for Signal + React surfaces."""


class HandlerRegistry:
    """Map action_type → handler. In-process singleton in
    :mod:`app.action_requests.handlers`; tests get a fresh registry."""

    def __init__(self) -> None:
        self._handlers: dict = {}

    def register(self, handler: ActionHandler) -> None:
        self._handlers[handler.action_type] = handler

    def unregister(self, action_type) -> None:
        self._handlers.pop(action_type, None)

    def get(self, action_type) -> ActionHandler | None:
        return self._handlers.get(action_type)

    def list_types(self) -> list:
        return sorted(self._handlers.keys(), key=lambda t: getattr(t, "value", str(t)))
