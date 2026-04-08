"""
events.py — Typed system events (ARCHITECTURAL REFERENCE).

NOTE: This file is documentation-as-code. It is not imported at runtime
by any module. It defines the canonical event types for reference by
developers and for future typed event bus implementation.

Defines the events that flow between subsystems. Subsystems produce events;
other subsystems consume them. This replaces ad-hoc dict passing with
typed contracts that can be validated at boundaries.

Usage:
    from app.contracts.events import TaskStarted, TaskCompleted, FeedbackReceived

    event = TaskStarted(crew="research", task="Find market data", difficulty=5)
    # Pass to health monitor, feedback pipeline, journal, etc.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class TaskStarted:
    """Emitted when Commander dispatches a task to a crew."""
    crew: str
    task: str
    difficulty: int = 3
    sender_id: str = ""
    task_id: str = ""
    model: str = ""
    parent_task_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class TaskCompleted:
    """Emitted when a crew finishes a task successfully."""
    crew: str
    task_id: str
    result_preview: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0
    model: str = ""
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class TaskFailed:
    """Emitted when a crew fails a task."""
    crew: str
    task_id: str
    error: str = ""
    error_type: str = ""
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class FeedbackReceived:
    """Emitted when user reacts to a bot response."""
    sender_id: str
    feedback_type: str  # explicit_positive, explicit_negative, explicit_correction, implicit_*
    emoji: str = ""
    target_timestamp: int = 0
    is_remove: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class PromptModified:
    """Emitted when a prompt version is promoted or rolled back."""
    role: str
    old_version: int = 0
    new_version: int = 0
    reason: str = ""
    modification_type: str = ""  # promotion, rollback, tier1_auto, tier2_approved
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class HealthAlert:
    """Emitted when health monitor detects a threshold violation."""
    severity: str  # warning, critical, emergency
    dimension: str  # error_rate, latency, hallucination, etc.
    current_value: float = 0.0
    threshold: float = 0.0
    auto_remediate: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class EvolutionResult:
    """Emitted after an evolution experiment completes."""
    strategy: str  # autoresearch, island, parallel, map_elites
    target_role: str = ""
    fitness_before: float = 0.0
    fitness_after: float = 0.0
    promoted: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
