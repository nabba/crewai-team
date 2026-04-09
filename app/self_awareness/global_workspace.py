"""
global_workspace.py — Global Workspace Theory (GWT) broadcast for AndrusAI.

Implements "global broadcast" — when important information arises,
it becomes available to ALL agents simultaneously (not just stored
in shared memory for optional pull).

Research: Butlin, Long, Chalmers (2025) — GWT identifies global broadcast
as a key consciousness indicator.

Architecture:
  - In-memory ring buffer (max 50 messages) for fast access
  - Critical messages persisted to PostgreSQL for durability
  - Agents check for unread broadcasts before each task
  - Broadcasts can only ADD caution (safety property)

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BroadcastMessage:
    """A single broadcast to the global workspace."""
    content: str
    importance: str = "normal"       # normal | high | critical
    source_agent: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    read_by: set[str] = field(default_factory=set)
    broadcast_id: int = 0

    def to_dict(self) -> dict:
        return {
            "content": self.content[:500],
            "importance": self.importance,
            "source_agent": self.source_agent,
            "timestamp": self.timestamp,
            "broadcast_id": self.broadcast_id,
        }


class GlobalWorkspace:
    """GWT-inspired broadcast mechanism for cross-agent coordination."""

    _instance: Optional["GlobalWorkspace"] = None
    _lock = threading.Lock()

    def __init__(self, max_messages: int = 50):
        self._messages: deque[BroadcastMessage] = deque(maxlen=max_messages)
        self._counter = 0
        self._msg_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "GlobalWorkspace":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def broadcast(
        self,
        content: str,
        importance: str = "normal",
        source_agent: str = "",
    ) -> BroadcastMessage:
        """Broadcast a message to the global workspace.

        All agents will see this in their next context injection.
        Critical messages are also persisted to PostgreSQL.
        """
        with self._msg_lock:
            self._counter += 1
            msg = BroadcastMessage(
                content=content,
                importance=importance,
                source_agent=source_agent,
                broadcast_id=self._counter,
            )
            self._messages.append(msg)

        # Persist critical messages
        if importance == "critical":
            try:
                from app.control_plane.db import execute
                execute(
                    """
                    INSERT INTO internal_states (
                        agent_id, decision_context, meta_strategy_assessment,
                        action_disposition, risk_tier, full_state
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        f"broadcast:{source_agent}",
                        content[:2000],
                        "critical_broadcast",
                        "escalate",
                        4,
                        '{"type": "broadcast", "importance": "critical"}',
                    ),
                )
            except Exception:
                pass

        logger.info(f"GWT broadcast [{importance}] from {source_agent}: {content[:80]}")
        return msg

    def check_broadcasts(
        self,
        agent_id: str,
        importance_filter: str = "high",
    ) -> list[BroadcastMessage]:
        """Return unread broadcasts for this agent at or above importance level.

        Marks returned messages as read by this agent.
        """
        importance_rank = {"normal": 0, "high": 1, "critical": 2}
        min_rank = importance_rank.get(importance_filter, 1)

        unread = []
        with self._msg_lock:
            for msg in self._messages:
                if agent_id not in msg.read_by:
                    if importance_rank.get(msg.importance, 0) >= min_rank:
                        unread.append(msg)
                        msg.read_by.add(agent_id)

        return unread

    def format_broadcasts(self, agent_id: str) -> str:
        """Format unread broadcasts for context injection."""
        unread = self.check_broadcasts(agent_id, importance_filter="high")
        if not unread:
            return ""

        lines = ["[Global Workspace Broadcasts]"]
        for msg in unread[-3:]:  # Max 3 broadcasts in context
            icon = "🔴" if msg.importance == "critical" else "🟡"
            lines.append(f"  {icon} [{msg.source_agent}] {msg.content[:200]}")
        return "\n".join(lines) + "\n"

    def get_recent(self, n: int = 10) -> list[dict]:
        """Return recent broadcasts for dashboard display."""
        with self._msg_lock:
            return [msg.to_dict() for msg in list(self._messages)[-n:]]


def get_workspace() -> GlobalWorkspace:
    return GlobalWorkspace.get_instance()


def broadcast(content: str, importance: str = "normal", source_agent: str = "") -> None:
    """Module-level convenience function for broadcasting."""
    get_workspace().broadcast(content, importance, source_agent)
