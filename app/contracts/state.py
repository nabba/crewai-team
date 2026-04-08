"""
state.py — Typed state documents (ARCHITECTURAL REFERENCE).

NOTE: This file is documentation-as-code. It is not imported at runtime
by any module. It defines the canonical state shapes for reference by
developers and for future typed state management implementation.

Defines the shape of state that flows between subsystems via Firestore
or internal APIs. These are the "data contracts" that the dashboard,
scheduler, feedback loop, self-heal pipeline, and knowledge queues
depend on.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CrewStatus:
    """State of a single crew, written to Firestore crews/{name}."""
    name: str
    state: str = "idle"  # idle, active
    current_task: str = ""
    task_id: str = ""
    started_at: str = ""
    eta: str = ""
    model: str = ""
    last_updated: str = ""


@dataclass
class SystemHealth:
    """Aggregated system health, written to Firestore status/system."""
    status: str = "online"  # online, offline
    health_score: float = 1.0
    uptime_seconds: int = 0
    active_crews: int = 0
    tasks_today: int = 0
    cost_today: float = 0.0
    last_heartbeat: str = ""


@dataclass
class SubsystemStatus:
    """Status of a single subsystem in the System Architecture Monitor."""
    status: str = "ok"  # ok, error, offline
    label: str = ""
    modules: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass
class ProjectContext:
    """Active project/venture context."""
    name: str = ""
    display_name: str = ""
    mem0_namespace: str = ""
    chroma_collection: str = ""


@dataclass
class PromptVersion:
    """State of a versioned prompt in the registry."""
    role: str
    active_version: int = 1
    total_versions: int = 1
    last_modified: str = ""
    last_promoted_by: str = ""


@dataclass
class EvolutionState:
    """Current state of the evolution subsystem."""
    ensemble_phase: str = "exploration"
    exploration_rate: float = 0.3
    epoch: int = 0
    archive_size: int = 0
    island_populations: list[int] = field(default_factory=list)
