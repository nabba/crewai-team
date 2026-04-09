"""
reality_model.py — Explicit Reality Model for AndrusAI agents.

Beautiful Loop theory (Laukkonen/Friston/Chandaria 2025) argues consciousness
requires a coherent world model integrating information across modalities.

For LLM agents, the "world model" is the agent's understanding of:
  - Current task and requirements
  - Relevant facts from RAG/memory
  - Environment state (tools, APIs)
  - Other agents' states
  - Own capabilities and limitations

This module makes the implicit world model EXPLICIT — a structured representation
that can be competed, reflected, precision-weighted, and tracked over time.

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class WorldModelElement:
    """A single element of the agent's reality model with precision weight."""
    element_id: str
    category: str          # task | fact | environment | social | self
    content: str
    precision: float       # 0.0-1.0: confidence in this element
    source: str            # rag | memory | observation | inference | input
    prediction: Optional[str] = None
    actual: Optional[str] = None
    prediction_error: float = 0.0

    def to_dict(self) -> dict:
        return {
            "element_id": self.element_id,
            "category": self.category,
            "content": self.content[:300],
            "precision": round(self.precision, 3),
            "source": self.source,
            "prediction_error": round(self.prediction_error, 3),
        }


@dataclass
class RealityModel:
    """The agent's explicit world model at a point in time."""
    agent_id: str
    step_number: int
    elements: list[WorldModelElement] = field(default_factory=list)
    global_coherence: float = 0.5
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_element(self, element: WorldModelElement) -> None:
        self.elements = [e for e in self.elements if e.element_id != element.element_id]
        self.elements.append(element)

    def get_by_category(self, category: str) -> list[WorldModelElement]:
        return [e for e in self.elements if e.category == category]

    @property
    def high_precision_elements(self) -> list[WorldModelElement]:
        return sorted(self.elements, key=lambda e: e.precision, reverse=True)

    @property
    def low_precision_elements(self) -> list[WorldModelElement]:
        return sorted(self.elements, key=lambda e: e.precision)

    @property
    def mean_precision(self) -> float:
        if not self.elements:
            return 0.5
        return sum(e.precision for e in self.elements) / len(self.elements)

    @property
    def total_prediction_error(self) -> float:
        """Proxy for free energy."""
        return sum(e.prediction_error for e in self.elements)

    def to_context_string(self, max_elements: int = 5) -> str:
        high = self.high_precision_elements[:max_elements // 2 + 1]
        low = self.low_precision_elements[:max_elements // 2]
        lines = ["[World Model]"]
        if high:
            lines.append("Confident: " + "; ".join(
                f"{e.content[:50]}({e.precision:.1f})" for e in high))
        if low:
            lines.append("Uncertain: " + "; ".join(
                f"{e.content[:50]}({e.precision:.1f})" for e in low))
        lines.append(f"Coherence={self.global_coherence:.2f}")
        return " | ".join(lines)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "step_number": self.step_number,
            "element_count": len(self.elements),
            "global_coherence": round(self.global_coherence, 3),
            "mean_precision": round(self.mean_precision, 3),
            "total_prediction_error": round(self.total_prediction_error, 3),
        }


class RealityModelBuilder:
    """Builds a RealityModel from available context sources."""

    def build(
        self,
        agent_id: str,
        step_number: int,
        task_description: str,
        rag_results: list = None,
        memory_results: list = None,
        peer_agent_states: list = None,
        self_assessment: dict = None,
    ) -> RealityModel:
        model = RealityModel(agent_id=agent_id, step_number=step_number)

        # Task understanding (given, high precision)
        model.add_element(WorldModelElement(
            element_id="task_primary", category="task",
            content=task_description[:500], precision=0.8, source="input",
        ))

        # RAG facts
        for i, result in enumerate((rag_results or [])[:5]):
            content = result.get("content", result.get("text", ""))[:300] if isinstance(result, dict) else str(result)[:300]
            score = result.get("relevance_score", result.get("score", 0.5)) if isinstance(result, dict) else 0.5
            model.add_element(WorldModelElement(
                element_id=f"rag_{i}", category="fact",
                content=content, precision=min(float(score), 1.0), source="rag",
            ))

        # Memory elements (older = lower precision)
        for i, mem in enumerate((memory_results or [])[:5]):
            content = mem.get("content", "")[:300] if isinstance(mem, dict) else str(mem)[:300]
            age_days = mem.get("age_days", 0) if isinstance(mem, dict) else 0
            precision = max(0.2, 0.9 - (age_days * 0.01))
            model.add_element(WorldModelElement(
                element_id=f"memory_{i}", category="fact",
                content=content, precision=precision, source="memory",
            ))

        # Peer agent states
        for i, peer in enumerate((peer_agent_states or [])[:3]):
            if isinstance(peer, dict):
                model.add_element(WorldModelElement(
                    element_id=f"peer_{peer.get('agent_id', i)}", category="social",
                    content=f"Agent {peer.get('agent_id', '?')}: {peer.get('status', 'unknown')}",
                    precision=0.6, source="observation",
                ))

        # Self-assessment
        if self_assessment:
            model.add_element(WorldModelElement(
                element_id="self_state", category="self",
                content=json.dumps(self_assessment)[:300],
                precision=0.7, source="inference",
            ))

        # Compute global coherence (simplified: ratio of high-precision elements)
        if model.elements:
            high_prec = sum(1 for e in model.elements if e.precision > 0.6)
            model.global_coherence = high_prec / len(model.elements)

        return model
