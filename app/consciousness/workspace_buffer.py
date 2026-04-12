"""
workspace_buffer.py — GWT-2: Competitive workspace with capacity constraint.

Implements the Butlin et al. (2025) GWT-2 indicator: a limited-capacity workspace
where information competes for access. Only the most salient items are admitted;
displaced items go to a peripheral queue. This is the bottleneck that gives
broadcast information its significance.

The workspace is NOT a message queue — it's a competition arena. The capacity
constraint forces prioritization, which is the defining property of GWT.

DGM Safety: Capacity, salience weights, and gating logic are infrastructure-level.
Agents cannot modify them.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

@dataclass
class WorkspaceItem:
    """An item competing for workspace access."""
    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    content_embedding: list[float] = field(default_factory=list)
    source_agent: str = ""
    source_channel: str = ""          # user_input, researcher_output, rag_retrieval, etc.
    salience_score: float = 0.0       # Composite from SalienceScorer
    goal_relevance: float = 0.0
    novelty_score: float = 0.0
    agent_urgency: float = 0.0
    surprise_signal: float = 0.0      # From PP-1 (0.0 if PP-1 not active)
    decay_rate: float = 0.05
    entered_at: float = field(default_factory=time.monotonic)
    cycles_in_workspace: int = 0
    consumed: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "content": self.content[:300],
            "source_agent": self.source_agent,
            "source_channel": self.source_channel,
            "salience_score": round(self.salience_score, 3),
            "goal_relevance": round(self.goal_relevance, 3),
            "novelty_score": round(self.novelty_score, 3),
            "cycles_in_workspace": self.cycles_in_workspace,
        }

@dataclass
class GateResult:
    """Result of competitive gating evaluation."""
    admitted: bool
    displaced_item: WorkspaceItem | None = None
    rejection_reason: str | None = None
    transition_type: str = "admitted"   # admitted, displaced, rejected, novelty_floor

def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors, normalized to [0, 1]."""
    if not a or not b or len(a) != len(b):
        return 0.5
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.5
    return (dot / (na * nb) + 1.0) / 2.0

class SalienceScorer:
    """Compute composite salience score for a workspace candidate."""

    def __init__(self, w_goal=0.35, w_novelty=0.25, w_urgency=0.15, w_surprise=0.25):
        self.w_goal = w_goal
        self.w_novelty = w_novelty
        self.w_urgency = w_urgency
        self.w_surprise = w_surprise

    def score(self, item: WorkspaceItem, goal_embeddings: list[list[float]],
              recent_items: list[WorkspaceItem]) -> float:
        """Compute composite salience. Pure arithmetic + embeddings."""
        # Goal alignment: max cosine sim to any current goal
        if goal_embeddings and item.content_embedding:
            item.goal_relevance = max(
                _cosine_sim(item.content_embedding, g) for g in goal_embeddings
            )
        else:
            item.goal_relevance = 0.5  # Default when no goals

        # Novelty: inverse max similarity to recent workspace items
        if recent_items and item.content_embedding:
            max_sim = max(
                (_cosine_sim(item.content_embedding, r.content_embedding)
                 for r in recent_items if r.content_embedding),
                default=0.5,
            )
            item.novelty_score = 1.0 - max_sim
        else:
            item.novelty_score = 0.8  # High novelty for first items

        # Recency decay for items already in workspace
        decay = (1.0 - item.decay_rate) ** item.cycles_in_workspace

        # Composite
        raw = (self.w_goal * item.goal_relevance
               + self.w_novelty * item.novelty_score
               + self.w_urgency * item.agent_urgency
               + self.w_surprise * item.surprise_signal)
        item.salience_score = round(raw * decay, 4)
        return item.salience_score

class CompetitiveGate:
    """Capacity-constrained competitive workspace gate."""

    def __init__(self, capacity: int = 5, novelty_floor_pct: float = 0.20,
                 consumption_decay: float = 0.50):
        self.capacity = capacity
        self.novelty_floor_pct = novelty_floor_pct
        self.consumption_decay = consumption_decay
        self._active: list[WorkspaceItem] = []
        self._peripheral: deque[WorkspaceItem] = deque(maxlen=100)
        self._cycle: int = 0
        self._lock = threading.Lock()
        self._novelty_admitted_this_cycle = False

    @property
    def active_items(self) -> list[WorkspaceItem]:
        with self._lock:
            return list(self._active)

    @property
    def peripheral_items(self) -> list[WorkspaceItem]:
        return list(self._peripheral)

    def set_dynamic_capacity(self, capacity: int,
                              novelty_floor_pct: float | None = None,
                              consumption_decay: float | None = None) -> None:
        """Set personality-driven workspace parameters for this cycle.

        Called before evaluate() when PDS integration is active.
        Capacity bounded to [2, 9] for safety.
        """
        self.capacity = max(2, min(9, capacity))
        if novelty_floor_pct is not None:
            self.novelty_floor_pct = max(0.05, min(0.50, novelty_floor_pct))
        if consumption_decay is not None:
            self.consumption_decay = max(0.10, min(0.90, consumption_decay))

    def advance_cycle(self) -> None:
        """Advance workspace cycle. Apply decay to all active items."""
        with self._lock:
            self._cycle += 1
            self._novelty_admitted_this_cycle = False
            for item in self._active:
                item.cycles_in_workspace += 1
                # Consumption decay: already-acted-on items lose salience faster
                if item.consumed:
                    item.salience_score *= self.consumption_decay

    def evaluate(self, candidate: WorkspaceItem) -> GateResult:
        """Evaluate a candidate for workspace admission."""
        with self._lock:
            # Below capacity: admit unconditionally
            if len(self._active) < self.capacity:
                self._active.append(candidate)
                return GateResult(admitted=True, transition_type="admitted")

            # At capacity: compete against lowest-salience active item
            lowest = min(self._active, key=lambda x: x.salience_score)

            if candidate.salience_score > lowest.salience_score:
                # Winner: displace lowest
                self._active.remove(lowest)
                self._active.append(candidate)
                lowest.metadata["exit_reason"] = "displaced"
                self._peripheral.append(lowest)
                return GateResult(
                    admitted=True,
                    displaced_item=lowest,
                    transition_type="displaced",
                )

            # Novelty floor: guarantee at least 1 high-novelty item per cycle
            if (not self._novelty_admitted_this_cycle
                    and candidate.novelty_score >= (1.0 - self.novelty_floor_pct)):
                # This candidate is in top novelty percentile
                self._active.remove(lowest)
                self._active.append(candidate)
                lowest.metadata["exit_reason"] = "displaced_by_novelty_floor"
                self._peripheral.append(lowest)
                self._novelty_admitted_this_cycle = True
                return GateResult(
                    admitted=True,
                    displaced_item=lowest,
                    transition_type="novelty_floor",
                )

            # Rejected: goes to peripheral queue
            self._peripheral.append(candidate)
            return GateResult(
                admitted=False,
                rejection_reason=f"salience {candidate.salience_score:.3f} < min active {lowest.salience_score:.3f}",
                transition_type="rejected",
            )

    def mark_consumed(self, item_id: str) -> None:
        """Mark an item as consumed (acted upon). Reduces salience."""
        with self._lock:
            for item in self._active:
                if item.item_id == item_id:
                    item.consumed = True
                    item.salience_score *= self.consumption_decay
                    break

    def get_snapshot(self) -> dict:
        """Dashboard-friendly workspace state snapshot."""
        with self._lock:
            return {
                "cycle": self._cycle,
                "capacity": self.capacity,
                "active_count": len(self._active),
                "peripheral_count": len(self._peripheral),
                "active_items": [i.to_dict() for i in self._active],
                "salience_distribution": {
                    i.item_id: round(i.salience_score, 3) for i in self._active
                },
            }

    def persist_transition(self, result: GateResult, item: WorkspaceItem) -> None:
        """Log transition to PostgreSQL for audit trail."""
        try:
            from app.control_plane.db import execute
            execute(
                """
                INSERT INTO workspace_transitions
                    (transition_type, item_id, displaced_item_id, salience_at_transition, cycle_number)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    result.transition_type,
                    item.item_id,
                    result.displaced_item.item_id if result.displaced_item else None,
                    item.salience_score,
                    self._cycle,
                ),
            )
        except Exception:
            pass

# ── Per-project workspace gates ──────────────────────────────────────────────
# Each project gets its own CompetitiveGate (capacity=3 by default).
# "generic" = default for non-project tasks (capacity=5).
# "__meta__" = global meta-workspace (capacity=7, fed by promotion).

GENERIC_WORKSPACE = "generic"
META_WORKSPACE = "__meta__"
_DEFAULT_PROJECT_CAPACITY = 3
_GENERIC_CAPACITY = 5
_META_CAPACITY = 7

_gates: dict[str, CompetitiveGate] = {}
_gates_lock = threading.Lock()
_scorer: SalienceScorer | None = None


def get_workspace_gate(project_id: str | None = None) -> CompetitiveGate:
    """Get or create workspace gate for a project.

    project_id=None or "generic" → default workspace (capacity=5)
    project_id="__meta__" → global meta-workspace (capacity=7)
    project_id="plg"|"archibal"|etc → project workspace (capacity=3)
    """
    if not project_id:
        project_id = GENERIC_WORKSPACE

    with _gates_lock:
        if project_id not in _gates:
            from app.consciousness.config import load_config
            cfg = load_config()
            if project_id == META_WORKSPACE:
                cap = _META_CAPACITY
            elif project_id == GENERIC_WORKSPACE:
                cap = cfg.workspace_capacity  # Config default (usually 5)
            else:
                cap = _DEFAULT_PROJECT_CAPACITY
            _gates[project_id] = CompetitiveGate(
                capacity=cap,
                novelty_floor_pct=cfg.novelty_floor_pct,
                consumption_decay=cfg.consumption_decay,
            )
        return _gates[project_id]


def create_workspace(project_id: str, capacity: int = 3) -> CompetitiveGate:
    """Create a new project workspace (called from API/dashboard)."""
    with _gates_lock:
        if project_id in _gates:
            return _gates[project_id]
        from app.consciousness.config import load_config
        cfg = load_config()
        gate = CompetitiveGate(
            capacity=max(2, min(9, capacity)),
            novelty_floor_pct=cfg.novelty_floor_pct,
            consumption_decay=cfg.consumption_decay,
        )
        _gates[project_id] = gate
        logger.info(f"workspace: created workspace '{project_id}' capacity={gate.capacity}")
        return gate


def list_workspaces() -> dict[str, dict]:
    """List all workspace gates with their snapshots."""
    with _gates_lock:
        return {pid: gate.get_snapshot() for pid, gate in _gates.items()}


def get_salience_scorer() -> SalienceScorer:
    global _scorer
    if _scorer is None:
        from app.consciousness.config import load_config
        cfg = load_config()
        _scorer = SalienceScorer(
            w_goal=cfg.salience_w_goal,
            w_novelty=cfg.salience_w_novelty,
            w_urgency=cfg.salience_w_urgency,
            w_surprise=cfg.salience_w_surprise,
        )
    return _scorer
