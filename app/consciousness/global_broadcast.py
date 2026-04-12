"""
global_broadcast.py — GWT-3: Global broadcast with agent reactions.

Implements Butlin et al. (2025) GWT-3: every workspace admission triggers a broadcast
to ALL agents simultaneously. Each agent independently evaluates relevance and generates
a reaction. The integration_score measures how broadly information resonated.

Broadcast ≠ task assignment. Broadcasting makes information AVAILABLE; agents decide
independently whether it's relevant. The Commander's delegation logic operates
DOWNSTREAM of broadcast.

DGM Safety: Broadcast routing is infrastructure-level. Agents cannot suppress broadcasts.
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

@dataclass
class AgentReaction:
    """An agent's reaction to a broadcast event."""
    agent_id: str
    reaction_type: str = "NOTED"      # NOTED | RELEVANT | URGENT | ACTIONABLE
    relevance_score: float = 0.0
    relevance_reason: str = ""
    proposed_action: str | None = None

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "reaction_type": self.reaction_type,
            "relevance_score": round(self.relevance_score, 3),
            "relevance_reason": self.relevance_reason[:200],
            "proposed_action": self.proposed_action[:200] if self.proposed_action else None,
        }

@dataclass
class BroadcastEvent:
    """A workspace item broadcast to all agents."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workspace_item_id: str = ""
    content_summary: str = ""
    content_embedding: list[float] = field(default_factory=list)
    broadcast_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    receiving_agents: list[str] = field(default_factory=list)
    reactions: dict[str, AgentReaction] = field(default_factory=dict)
    integration_score: float = 0.0
    cycle_number: int = 0

    def compute_integration_score(self) -> float:
        """Count of RELEVANT+ reactions / total agents."""
        if not self.receiving_agents:
            return 0.0
        relevant_count = sum(
            1 for r in self.reactions.values()
            if r.reaction_type in ("RELEVANT", "URGENT", "ACTIONABLE")
        )
        self.integration_score = round(relevant_count / len(self.receiving_agents), 3)
        return self.integration_score

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "workspace_item_id": self.workspace_item_id,
            "content_summary": self.content_summary[:200],
            "integration_score": self.integration_score,
            "reaction_count": len(self.reactions),
            "reactions": {k: v.to_dict() for k, v in self.reactions.items()},
        }

@dataclass
class AgentBroadcastListener:
    """Per-agent broadcast configuration."""
    agent_id: str
    role: str                          # researcher, coder, writer, etc.
    current_task_embedding: list[float] = field(default_factory=list)
    reaction_threshold: float = 0.30
    attention_budget: int = 3
    broadcasts_processed: int = 0

    def reset_budget(self) -> None:
        self.broadcasts_processed = 0

    def has_budget(self) -> bool:
        return self.broadcasts_processed < self.attention_budget

class GlobalBroadcastEngine:
    """Manages global broadcast of workspace items to all registered agents."""

    def __init__(self):
        self._listeners: dict[str, AgentBroadcastListener] = {}
        self._log: deque[BroadcastEvent] = deque(maxlen=100)
        self._cycle: int = 0

    def register_listener(self, agent_id: str, role: str,
                          task_embedding: list[float] = None) -> None:
        """Register an agent to receive broadcasts."""
        from app.consciousness.config import load_config
        cfg = load_config()
        self._listeners[agent_id] = AgentBroadcastListener(
            agent_id=agent_id,
            role=role,
            current_task_embedding=task_embedding or [],
            reaction_threshold=cfg.reaction_threshold,
            attention_budget=cfg.attention_budget,
        )

    def update_listener_context(self, agent_id: str, task_embedding: list[float]) -> None:
        """Update an agent's current task context (for relevance computation)."""
        if agent_id in self._listeners:
            self._listeners[agent_id].current_task_embedding = task_embedding

    def advance_cycle(self) -> None:
        """Reset per-cycle attention budgets."""
        self._cycle += 1
        for listener in self._listeners.values():
            listener.reset_budget()

    def broadcast(self, item) -> BroadcastEvent:
        """Broadcast a workspace item to all registered agents.

        For each listener:
        1. Compute relevance (embedding cosine similarity)
        2. If relevance > threshold AND budget available: generate reaction
        3. If not: record NOTED (no deep processing)
        4. Compute integration_score

        Returns BroadcastEvent with all reactions.
        """
        from app.consciousness.workspace_buffer import _cosine_sim

        event = BroadcastEvent(
            workspace_item_id=item.item_id,
            content_summary=item.content[:500],
            content_embedding=item.content_embedding,
            receiving_agents=list(self._listeners.keys()),
            cycle_number=self._cycle,
        )

        for agent_id, listener in self._listeners.items():
            # Compute relevance via embedding similarity
            if listener.current_task_embedding and item.content_embedding:
                relevance = _cosine_sim(item.content_embedding, listener.current_task_embedding)
            else:
                relevance = 0.3  # Default when no embedding

            if relevance > listener.reaction_threshold and listener.has_budget():
                # Deep processing: generate reaction
                reaction = self._generate_reaction(
                    agent_id, listener.role, item, relevance
                )
                listener.broadcasts_processed += 1
            else:
                # Shallow: just note it
                reaction = AgentReaction(
                    agent_id=agent_id,
                    reaction_type="NOTED",
                    relevance_score=relevance,
                    relevance_reason="Below threshold or budget exhausted",
                )

            event.reactions[agent_id] = reaction

        event.compute_integration_score()
        self._log.append(event)

        # Update social attention model (Theory of Mind for other agents)
        try:
            from app.consciousness.attention_schema import get_social_attention_model
            social = get_social_attention_model()
            for agent_id, reaction in event.reactions.items():
                listener = self._listeners.get(agent_id)
                role = listener.role if listener else ""
                social.update_from_broadcast_reaction(
                    agent_id=agent_id,
                    role=role,
                    topic=item.content[:100],
                    reaction_type=reaction.reaction_type,
                    relevance_score=reaction.relevance_score,
                )
                social.evaluate_prediction_accuracy(agent_id, reaction.reaction_type)
        except Exception:
            pass

        # Persist to PostgreSQL
        self._persist_event(event)

        logger.info(
            f"GWT-3 broadcast: item={item.item_id[:8]}, "
            f"integration={event.integration_score:.2f}, "
            f"reactions={len(event.reactions)}"
        )
        return event

    def _generate_reaction(self, agent_id: str, role: str,
                           item, relevance: float) -> AgentReaction:
        """Generate agent reaction via local Ollama tier (fast, cheap)."""
        try:
            from app.llm_factory import create_specialist_llm
            llm = create_specialist_llm(max_tokens=200, role="self_improve", force_tier="local")

            prompt = (
                f"You are the {role} agent. A workspace broadcast arrived:\n"
                f'"{item.content[:300]}"\n\n'
                f"Rate relevance (0.0-1.0) and classify: NOTED | RELEVANT | URGENT | ACTIONABLE\n"
                f"If ACTIONABLE, state what action you propose.\n"
                f'Respond in JSON: {{"relevance": float, "type": str, "reason": str, "action": str|null}}'
            )

            raw = str(llm.call(prompt)).strip()

            # Parse JSON response
            import json
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # Extract from markdown fences
                from app.utils import safe_json_parse
                data, _ = safe_json_parse(raw)
                if not data:
                    data = {}

            return AgentReaction(
                agent_id=agent_id,
                reaction_type=data.get("type", "RELEVANT") if relevance > 0.5 else "NOTED",
                relevance_score=float(data.get("relevance", relevance)),
                relevance_reason=str(data.get("reason", ""))[:200],
                proposed_action=data.get("action"),
            )
        except Exception as e:
            # Fallback: heuristic reaction based on embedding similarity
            rtype = "RELEVANT" if relevance > 0.6 else ("NOTED" if relevance < 0.4 else "RELEVANT")
            return AgentReaction(
                agent_id=agent_id,
                reaction_type=rtype,
                relevance_score=relevance,
                relevance_reason=f"Heuristic (LLM unavailable: {str(e)[:50]})",
            )

    def _persist_event(self, event: BroadcastEvent) -> None:
        """Store broadcast event and reactions to PostgreSQL."""
        try:
            from app.control_plane.db import execute
            execute(
                """
                INSERT INTO broadcast_events
                    (event_id, workspace_item_id, broadcast_at, broadcast_cycle,
                     receiving_agents, integration_score)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    event.event_id,
                    event.workspace_item_id,
                    event.broadcast_at,
                    event.cycle_number,
                    event.receiving_agents,
                    event.integration_score,
                ),
            )
            for reaction in event.reactions.values():
                execute(
                    """
                    INSERT INTO broadcast_reactions
                        (event_id, agent_id, reaction_type, relevance_score,
                         relevance_reason, proposed_action)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        event.event_id,
                        reaction.agent_id,
                        reaction.reaction_type,
                        reaction.relevance_score,
                        reaction.relevance_reason,
                        reaction.proposed_action,
                    ),
                )
        except Exception:
            logger.debug("GWT-3: broadcast persistence failed", exc_info=True)

    def get_recent_events(self, n: int = 10) -> list[dict]:
        """Dashboard API: recent broadcast events."""
        return [e.to_dict() for e in list(self._log)[-n:]]

# ── Module-level singleton ──────────────────────────────────────────────────

_engine: GlobalBroadcastEngine | None = None

def get_broadcast_engine() -> GlobalBroadcastEngine:
    global _engine
    if _engine is None:
        _engine = GlobalBroadcastEngine()
        # Register default agent listeners
        for role in ("researcher", "coder", "writer", "media_analyst"):
            _engine.register_listener(role, role)
    return _engine
