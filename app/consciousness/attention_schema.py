"""
attention_schema.py — AST-1: Modeling own attention state.

Implements Butlin et al. (2025) AST-1: the system maintains an internal model
of WHAT it's attending to, WHY, whether it SHOULD shift, and how accurate
its attentional predictions are.

Dual-timescale operation:
  Fast loop: real-time monitoring during workspace competition. Can intervene
             (suppress capturing item, boost neglected item) DURING gating.
  Slow loop: evaluate attention patterns over time. "Am I over-attending to X
             and neglecting Y? Are my attention predictions improving?"

DGM Safety: Schema recommendations are advisory. Cannot force workspace
modifications that violate DGM constraints.
"""

from __future__ import annotations

import logging
import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

@dataclass
class AttentionState:
    """Snapshot of current attentional allocation."""
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    workspace_item_ids: list[str] = field(default_factory=list)
    salience_distribution: dict[str, float] = field(default_factory=dict)
    attending_because: str = ""
    attention_duration_s: float = 0.0
    cycle_number: int = 0
    source_trigger: str = "GOAL_DRIVEN"  # GOAL_DRIVEN | STIMULUS_DRIVEN | SCHEMA_DIRECTED
    is_stuck: bool = False
    is_captured: bool = False
    capturing_item_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "state_id": self.state_id,
            "workspace_items": len(self.workspace_item_ids),
            "is_stuck": self.is_stuck,
            "is_captured": self.is_captured,
            "source_trigger": self.source_trigger,
            "cycle": self.cycle_number,
        }

@dataclass
class AttentionPrediction:
    """Prediction of next workspace focus."""
    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    predicted_focus_ids: list[str] = field(default_factory=list)
    predicted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    actual_focus_ids: list[str] | None = None
    accuracy: float | None = None
    cycle_number: int = 0

@dataclass
class AttentionShift:
    """Record of an attention shift (schema-directed or otherwise)."""
    shift_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trigger: str = ""        # stuck_detection | capture_detection | schema_recommendation | surprise_redirect
    shift_cost: float = 0.0
    utility_delta: float | None = None
    cooldown_until_cycle: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class AttentionController:
    """Detects stuck/capture states and recommends shifts."""

    def __init__(self, stuck_threshold_cycles: int = 5,
                 capture_dominance_threshold: float = 0.70,
                 shift_cooldown_cycles: int = 3,
                 max_shifts_per_period: int = 2):
        self.stuck_threshold = stuck_threshold_cycles
        self.capture_threshold = capture_dominance_threshold
        self.shift_cooldown = shift_cooldown_cycles
        self.max_shifts = max_shifts_per_period
        self._shifts_this_period: int = 0
        self._cooldown_until: int = 0

    def detect_stuck(self, history: list[AttentionState]) -> bool:
        """True if workspace has same items for > threshold cycles without new actions."""
        if len(history) < self.stuck_threshold:
            return False
        recent = history[-self.stuck_threshold:]
        if not recent[0].workspace_item_ids:
            return False
        # Check if item IDs are substantially the same across recent cycles
        first_set = set(recent[0].workspace_item_ids)
        for state in recent[1:]:
            overlap = len(first_set & set(state.workspace_item_ids))
            if overlap < len(first_set) * 0.8:  # >20% changed = not stuck
                return False
        return True

    def detect_capture(self, state: AttentionState) -> tuple[bool, str | None]:
        """True if one item dominates salience distribution."""
        if not state.salience_distribution:
            return False, None
        total = sum(state.salience_distribution.values())
        if total == 0:
            return False, None
        for item_id, salience in state.salience_distribution.items():
            if salience / total > self.capture_threshold:
                return True, item_id
        return False, None

    def can_recommend_shift(self, current_cycle: int) -> bool:
        """Check cooldown and frequency limits."""
        if current_cycle < self._cooldown_until:
            return False
        if self._shifts_this_period >= self.max_shifts:
            return False
        return True

    def record_shift(self, current_cycle: int) -> AttentionShift:
        """Record a shift and set cooldown."""
        self._shifts_this_period += 1
        self._cooldown_until = current_cycle + self.shift_cooldown
        return AttentionShift(
            trigger="schema_recommendation",
            cooldown_until_cycle=self._cooldown_until,
        )

    def reset_period(self) -> None:
        """Reset shift counter for new slow-loop period."""
        self._shifts_this_period = 0

class AttentionPredictor:
    """Predicts next workspace focus and tracks accuracy."""

    def __init__(self, history_window: int = 20):
        self._predictions: deque[AttentionPrediction] = deque(maxlen=history_window)
        self._accuracy_history: deque[float] = deque(maxlen=50)

    def predict_next_focus(self, current_state: AttentionState) -> AttentionPrediction:
        """Predict which items will be in workspace next cycle.

        Heuristic: items with highest current salience likely persist.
        """
        if not current_state.salience_distribution:
            return AttentionPrediction(cycle_number=current_state.cycle_number + 1)

        # Top items by salience = predicted to persist
        sorted_items = sorted(
            current_state.salience_distribution.items(),
            key=lambda x: x[1], reverse=True,
        )
        predicted_ids = [item_id for item_id, _ in sorted_items[:5]]

        pred = AttentionPrediction(
            predicted_focus_ids=predicted_ids,
            cycle_number=current_state.cycle_number + 1,
        )
        self._predictions.append(pred)
        return pred

    def evaluate_prediction(self, prediction: AttentionPrediction,
                            actual_state: AttentionState) -> float:
        """Compare prediction to reality. Returns accuracy [0, 1]."""
        if not prediction.predicted_focus_ids or not actual_state.workspace_item_ids:
            return 0.5

        predicted = set(prediction.predicted_focus_ids)
        actual = set(actual_state.workspace_item_ids)
        if not predicted:
            return 0.5

        overlap = len(predicted & actual)
        accuracy = overlap / max(len(predicted), len(actual))
        prediction.actual_focus_ids = actual_state.workspace_item_ids
        prediction.accuracy = accuracy
        self._accuracy_history.append(accuracy)
        return accuracy

    @property
    def running_accuracy(self) -> float:
        if not self._accuracy_history:
            return 0.5
        return sum(self._accuracy_history) / len(self._accuracy_history)

class AttentionSchema:
    """Full attention schema: state tracking, prediction, control."""

    def __init__(self):
        self.controller = AttentionController()
        self.predictor = AttentionPredictor()
        self._history: deque[AttentionState] = deque(maxlen=50)
        self._current: AttentionState | None = None
        self._cycle: int = 0
        self._shifts: list[AttentionShift] = []

    def update(self, workspace_items: list, cycle: int = None) -> AttentionState:
        """Called on every workspace state change. Creates new AttentionState."""
        self._cycle = cycle or self._cycle + 1

        # Build salience distribution
        salience_dist = {}
        item_ids = []
        for item in workspace_items:
            salience_dist[item.item_id] = item.salience_score
            item_ids.append(item.item_id)

        # Determine attending_because
        if item_ids:
            top_item = max(workspace_items, key=lambda x: x.salience_score)
            reason = f"Highest salience: {top_item.content[:60]} ({top_item.salience_score:.2f})"
        else:
            reason = "Empty workspace"

        state = AttentionState(
            workspace_item_ids=item_ids,
            salience_distribution=salience_dist,
            attending_because=reason,
            cycle_number=self._cycle,
        )

        # Evaluate previous prediction
        if self._history:
            prev_pred = None
            for p in reversed(list(self.predictor._predictions)):
                if p.cycle_number == self._cycle:
                    prev_pred = p
                    break
            if prev_pred:
                self.predictor.evaluate_prediction(prev_pred, state)

        # Detect stuck
        history_list = list(self._history) + [state]
        state.is_stuck = self.controller.detect_stuck(history_list)

        # Detect capture
        captured, capturing_id = self.controller.detect_capture(state)
        state.is_captured = captured
        state.capturing_item_id = capturing_id

        # Generate next prediction
        self.predictor.predict_next_focus(state)

        self._history.append(state)
        self._current = state

        # Persist
        self._persist_state(state)

        return state

    def recommend_intervention(self) -> dict | None:
        """If stuck or captured, recommend intervention for workspace gate.

        Returns dict with suppression/boost directives, or None.
        LEGACY: use apply_direct_intervention() for true direct authority.
        """
        if not self._current:
            return None

        if not self.controller.can_recommend_shift(self._cycle):
            return None

        if self._current.is_captured and self._current.capturing_item_id:
            shift = self.controller.record_shift(self._cycle)
            self._shifts.append(shift)
            return {
                "action": "suppress",
                "target_item_id": self._current.capturing_item_id,
                "salience_reduction": 0.3,
                "reason": f"Capture detected: item dominates >{self.controller.capture_threshold*100:.0f}% of salience",
            }

        if self._current.is_stuck:
            shift = self.controller.record_shift(self._cycle)
            self._shifts.append(shift)
            return {
                "action": "boost_novelty",
                "reason": f"Stuck: same items for {self.controller.stuck_threshold}+ cycles",
            }

        return None

    # ── True Direct Authority (DGM-bounded) ──────────────────────────────

    # DGM Safety Bounds — immutable, infrastructure-level
    MAX_SALIENCE_CHANGE = 0.50   # Max ±50% salience modification per item
    MIN_SALIENCE_FLOOR = 0.05    # Items can never be suppressed below this
    MAX_BOOST = 2.0              # Max 2x boost factor

    def apply_direct_intervention(self, gate) -> dict:
        """Directly modify workspace gate items when stuck or captured.

        Unlike recommend_intervention() (advisory), this method has true
        direct authority over the workspace — it modifies salience scores
        and can force displacements within DGM safety bounds.

        DGM Safety Bounds:
          - Max ±50% salience change per item per intervention
          - Items never suppressed below MIN_SALIENCE_FLOOR (0.05)
          - Boost factor capped at 2.0x
          - Cannot remove items entirely (only suppress salience)
          - Cannot exceed workspace capacity
          - Interventions logged for audit trail

        Args:
            gate: CompetitiveGate instance to modify directly

        Returns:
            dict with intervention details (for logging/dashboard)
        """
        result = {"applied": False, "actions": [], "reason": ""}

        if not self._current:
            return result

        if not self.controller.can_recommend_shift(self._cycle):
            result["reason"] = "shift cooldown active"
            return result

        # ── Capture intervention: suppress dominant item ──────────────
        if self._current.is_captured and self._current.capturing_item_id:
            target_id = self._current.capturing_item_id
            with gate._lock:
                for item in gate._active:
                    if item.item_id == target_id:
                        old_salience = item.salience_score
                        # Apply suppression (clamped to DGM bounds)
                        reduction = min(self.MAX_SALIENCE_CHANGE, 0.40)
                        new_salience = max(
                            self.MIN_SALIENCE_FLOOR,
                            old_salience * (1.0 - reduction),
                        )
                        item.salience_score = new_salience

                        # Also boost the lowest-salience non-capturing item
                        others = [i for i in gate._active if i.item_id != target_id]
                        if others:
                            weakest = min(others, key=lambda x: x.salience_score)
                            old_weak = weakest.salience_score
                            boost = min(self.MAX_BOOST, 1.0 + self.MAX_SALIENCE_CHANGE)
                            weakest.salience_score = min(1.0, old_weak * boost)

                            result["actions"].append({
                                "type": "boost",
                                "item_id": weakest.item_id[:12],
                                "old_salience": round(old_weak, 3),
                                "new_salience": round(weakest.salience_score, 3),
                            })

                        shift = self.controller.record_shift(self._cycle)
                        self._shifts.append(shift)
                        result["applied"] = True
                        result["reason"] = (
                            f"Capture: suppressed {target_id[:12]} "
                            f"({old_salience:.2f}→{new_salience:.2f})"
                        )
                        result["actions"].append({
                            "type": "suppress",
                            "item_id": target_id[:12],
                            "old_salience": round(old_salience, 3),
                            "new_salience": round(new_salience, 3),
                        })
                        logger.info(f"AST-1 DIRECT: {result['reason']}")
                        break

        # ── Stuck intervention: boost peripheral + suppress stale ─────
        elif self._current.is_stuck:
            with gate._lock:
                # Suppress oldest active items (they're stale)
                if gate._active:
                    stale = max(gate._active, key=lambda x: x.cycles_in_workspace)
                    old_salience = stale.salience_score
                    reduction = min(self.MAX_SALIENCE_CHANGE, 0.35)
                    stale.salience_score = max(
                        self.MIN_SALIENCE_FLOOR,
                        old_salience * (1.0 - reduction),
                    )
                    result["actions"].append({
                        "type": "suppress_stale",
                        "item_id": stale.item_id[:12],
                        "old_salience": round(old_salience, 3),
                        "new_salience": round(stale.salience_score, 3),
                        "cycles": stale.cycles_in_workspace,
                    })

                # If peripheral has items, boost best one and force-admit it
                if gate._peripheral:
                    best_peripheral = max(gate._peripheral, key=lambda x: x.salience_score)
                    old_p_salience = best_peripheral.salience_score
                    boost = min(self.MAX_BOOST, 1.5)
                    best_peripheral.salience_score = min(1.0, old_p_salience * boost)
                    result["actions"].append({
                        "type": "boost_peripheral",
                        "item_id": best_peripheral.item_id[:12],
                        "old_salience": round(old_p_salience, 3),
                        "new_salience": round(best_peripheral.salience_score, 3),
                    })
                    # Direct admission: if boosted peripheral beats lowest active, swap
                    # (No gate.evaluate() call — we're inside the lock, do it manually)
                    if gate._active:
                        lowest_active = min(gate._active, key=lambda x: x.salience_score)
                        if best_peripheral.salience_score > lowest_active.salience_score and len(gate._active) >= gate.capacity:
                            gate._active.remove(lowest_active)
                            gate._peripheral.append(lowest_active)
                            gate._active.append(best_peripheral)
                            gate._peripheral.remove(best_peripheral)
                            result["actions"].append({
                                "type": "forced_admission",
                                "item_id": best_peripheral.item_id[:12],
                                "displaced": lowest_active.item_id[:12],
                            })
                        elif len(gate._active) < gate.capacity:
                            gate._active.append(best_peripheral)
                            gate._peripheral.remove(best_peripheral)
                            result["actions"].append({
                                "type": "forced_admission",
                                "item_id": best_peripheral.item_id[:12],
                            })

                shift = self.controller.record_shift(self._cycle)
                self._shifts.append(shift)
                result["applied"] = True
                result["reason"] = f"Stuck: suppressed stale + boosted peripheral"
                logger.info(f"AST-1 DIRECT: {result['reason']}")

        return result

    def get_state_summary(self) -> dict:
        """Dashboard/introspection summary."""
        return {
            "cycle": self._cycle,
            "is_stuck": self._current.is_stuck if self._current else False,
            "is_captured": self._current.is_captured if self._current else False,
            "prediction_accuracy": round(self.predictor.running_accuracy, 3),
            "shifts_this_period": self.controller._shifts_this_period,
            "history_length": len(self._history),
            "workspace_size": len(self._current.workspace_item_ids) if self._current else 0,
        }

    def run_slow_loop(self) -> dict:
        """Slow loop: evaluate attention patterns, reset shift counter."""
        self.controller.reset_period()
        summary = self.get_state_summary()

        # Check for over-attention patterns
        if len(self._history) >= 10:
            # Count how often each source_channel appears
            channel_counts: dict[str, int] = {}
            for state in self._history:
                for item_id in state.workspace_item_ids:
                    channel_counts[item_id] = channel_counts.get(item_id, 0) + 1
            summary["dominant_items"] = sorted(
                channel_counts.items(), key=lambda x: x[1], reverse=True,
            )[:3]

        logger.info(f"AST-1 slow loop: accuracy={summary['prediction_accuracy']:.2f}, "
                    f"stuck={summary['is_stuck']}, captured={summary['is_captured']}")
        return summary

    def _persist_state(self, state: AttentionState) -> None:
        """Store attention state to PostgreSQL."""
        try:
            from app.control_plane.db import execute
            execute(
                """
                INSERT INTO attention_states
                    (state_id, workspace_item_ids, salience_distribution,
                     attending_because, cycle_number, source_trigger,
                     is_stuck, is_captured, capturing_item_id)
                VALUES (%s, %s::uuid[], %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    state.state_id,
                    state.workspace_item_ids,
                    __import__("json").dumps(state.salience_distribution),
                    state.attending_because[:500],
                    state.cycle_number,
                    state.source_trigger,
                    state.is_stuck,
                    state.is_captured,
                    state.capturing_item_id,
                ),
            )
        except Exception:
            pass

# ── Module-level singleton ──────────────────────────────────────────────────

_schema: AttentionSchema | None = None

def get_attention_schema() -> AttentionSchema:
    global _schema
    if _schema is None:
        _schema = AttentionSchema()
    return _schema


# ═══════════════════════════════════════════════════════════════════════════════
# Social Attention Modeling — Theory of Mind for Agent Attention (VIII-3)
#
# Extends AST-1 (self-model of attention) to model OTHER agents' attention.
# The system predicts what each agent WOULD attend to, enabling:
#   1. Better delegation (route tasks to agents whose attention aligns)
#   2. Richer self/other distinction (VIII-3 unified self-model)
#   3. Anticipatory coordination (predict what agents need before asking)
#
# Each AgentAttentionModel maintains a history of what topics that agent
# was relevant to (from GWT-3 broadcast reactions), their specialization
# profile, and predicted current attention focus.
#
# No LLM calls — pure arithmetic on broadcast reaction history.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentAttentionModel:
    """Model of another agent's attention state (Theory of Mind)."""
    agent_id: str = ""
    role: str = ""
    # Attention profile: what topics this agent typically attends to
    topic_affinities: dict[str, float] = field(default_factory=dict)  # topic → relevance mean
    # Predicted current focus
    predicted_focus: str = ""
    predicted_relevance: float = 0.0
    # Historical accuracy of our predictions about this agent
    prediction_accuracy: float = 0.5
    # Activity level (how often this agent reacts to broadcasts)
    activity_level: float = 0.5  # 0 = never reacts, 1 = always reacts
    # Last N reactions for trend analysis
    recent_reactions: list[str] = field(default_factory=list)  # reaction_types

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "predicted_focus": self.predicted_focus[:100],
            "predicted_relevance": round(self.predicted_relevance, 3),
            "prediction_accuracy": round(self.prediction_accuracy, 3),
            "activity_level": round(self.activity_level, 3),
            "top_affinities": sorted(
                self.topic_affinities.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }


class SocialAttentionModel:
    """Models other agents' attention states for Theory of Mind.

    Uses GWT-3 broadcast reactions as evidence: when agent X reacts
    to a broadcast about topic Y with RELEVANT/ACTIONABLE, we learn
    that X attends to Y-like topics.

    This enables:
    - Predicting which agent would best handle a given task
    - Distinguishing self-attention from other-attention (VIII-3)
    - Anticipating coordination needs
    """

    def __init__(self):
        self._models: dict[str, AgentAttentionModel] = {}
        self._prediction_log: deque[dict] = deque(maxlen=100)

    def get_or_create_model(self, agent_id: str, role: str = "") -> AgentAttentionModel:
        """Get or create attention model for an agent."""
        if agent_id not in self._models:
            self._models[agent_id] = AgentAttentionModel(
                agent_id=agent_id, role=role,
            )
        return self._models[agent_id]

    def update_from_broadcast_reaction(
        self, agent_id: str, role: str,
        topic: str, reaction_type: str, relevance_score: float,
    ) -> None:
        """Update agent's attention model from a GWT-3 broadcast reaction.

        Called after each broadcast cycle with each agent's reaction.
        Builds up the topic affinity profile over time.
        """
        model = self.get_or_create_model(agent_id, role)

        # Update topic affinity (exponential moving average)
        alpha = 0.2
        current = model.topic_affinities.get(topic, 0.5)
        model.topic_affinities[topic] = round(
            alpha * relevance_score + (1.0 - alpha) * current, 3
        )

        # Track reaction types
        model.recent_reactions.append(reaction_type)
        if len(model.recent_reactions) > 20:
            model.recent_reactions = model.recent_reactions[-20:]

        # Update activity level
        active_reactions = sum(
            1 for r in model.recent_reactions
            if r in ("RELEVANT", "URGENT", "ACTIONABLE")
        )
        model.activity_level = round(active_reactions / max(len(model.recent_reactions), 1), 3)

        # Cap topic affinities to prevent unbounded growth
        if len(model.topic_affinities) > 50:
            sorted_topics = sorted(
                model.topic_affinities.items(), key=lambda x: x[1]
            )
            model.topic_affinities = dict(sorted_topics[-50:])

    def predict_agent_attention(
        self, agent_id: str, task_content: str,
    ) -> tuple[float, str]:
        """Predict how relevant a task would be to a given agent.

        Returns (predicted_relevance, reasoning).
        Uses the agent's topic affinity profile to estimate attention.
        """
        model = self._models.get(agent_id)
        if not model or not model.topic_affinities:
            return 0.5, "No attention history for this agent"

        # Score by keyword overlap with known affinities
        task_words = set(task_content.lower().split())
        relevance_sum = 0.0
        match_count = 0
        for topic, affinity in model.topic_affinities.items():
            topic_words = set(topic.lower().split())
            if task_words & topic_words:
                relevance_sum += affinity
                match_count += 1

        if match_count == 0:
            predicted = model.activity_level * 0.5  # Base prediction from activity
            reason = f"No topic overlap; using base activity level ({model.activity_level:.2f})"
        else:
            predicted = min(1.0, relevance_sum / match_count)
            reason = f"Matched {match_count} known topics (avg affinity={predicted:.2f})"

        model.predicted_focus = task_content[:100]
        model.predicted_relevance = predicted
        return predicted, reason

    def predict_best_agent_for_task(
        self, task_content: str, candidate_agents: list[str] | None = None,
    ) -> list[tuple[str, float, str]]:
        """Rank agents by predicted attention relevance for a task.

        Returns sorted list of (agent_id, predicted_relevance, reasoning).
        This is the Theory of Mind delegation aid.
        """
        agents = candidate_agents or list(self._models.keys())
        rankings = []
        for agent_id in agents:
            relevance, reason = self.predict_agent_attention(agent_id, task_content)
            rankings.append((agent_id, relevance, reason))
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def evaluate_prediction_accuracy(
        self, agent_id: str, actual_reaction_type: str,
    ) -> None:
        """Update prediction accuracy after observing actual reaction.

        Call after broadcast to compare predicted relevance vs actual.
        """
        model = self._models.get(agent_id)
        if not model:
            return

        # Convert reaction type to binary relevance
        actually_relevant = actual_reaction_type in ("RELEVANT", "URGENT", "ACTIONABLE")
        predicted_relevant = model.predicted_relevance > 0.5

        # Update accuracy (exponential moving average)
        correct = 1.0 if (predicted_relevant == actually_relevant) else 0.0
        alpha = 0.15
        model.prediction_accuracy = round(
            alpha * correct + (1.0 - alpha) * model.prediction_accuracy, 3
        )

        self._prediction_log.append({
            "agent_id": agent_id,
            "predicted": model.predicted_relevance,
            "actual": actual_reaction_type,
            "correct": correct,
        })

    def get_self_other_distinction(self, self_attention: AttentionState | None) -> dict:
        """Compare self-attention to modeled other-attention.

        This is the VIII-3 unified self-model property: the system
        distinguishes its own attention from its model of others.

        Returns dict with self vs. other attention comparison.
        """
        result = {
            "self_focus": "",
            "self_salience_entropy": 0.0,
            "other_models_count": len(self._models),
            "other_attention_summary": [],
            "self_other_divergence": 0.0,
        }

        if self_attention:
            result["self_focus"] = self_attention.attending_because[:200]
            # Compute entropy of self salience distribution
            dist = self_attention.salience_distribution
            if dist:
                total = sum(dist.values())
                if total > 0:
                    probs = [v / total for v in dist.values()]
                    entropy = -sum(p * math.log(p + 1e-10) for p in probs)
                    result["self_salience_entropy"] = round(entropy, 3)

        # Compare to other agents' predicted attention
        for agent_id, model in self._models.items():
            result["other_attention_summary"].append({
                "agent": agent_id,
                "focus": model.predicted_focus[:50],
                "relevance": model.predicted_relevance,
                "accuracy": model.prediction_accuracy,
            })

        # Divergence: how different is self-attention from others' predicted attention?
        if self_attention and self._models:
            self_top = max(self_attention.salience_distribution.values(), default=0)
            other_relevances = [m.predicted_relevance for m in self._models.values()]
            if other_relevances:
                other_mean = sum(other_relevances) / len(other_relevances)
                result["self_other_divergence"] = round(abs(self_top - other_mean), 3)

        return result

    def get_summary(self) -> dict:
        """Dashboard summary of social attention models."""
        return {
            "agents_modeled": len(self._models),
            "prediction_accuracy": round(
                sum(m.prediction_accuracy for m in self._models.values())
                / max(len(self._models), 1), 3
            ),
            "predictions_logged": len(self._prediction_log),
            "agent_models": {
                agent_id: model.to_dict()
                for agent_id, model in self._models.items()
            },
        }


# ── Module-level singleton ──────────────────────────────────────────────────

_social_model: SocialAttentionModel | None = None


def get_social_attention_model() -> SocialAttentionModel:
    global _social_model
    if _social_model is None:
        _social_model = SocialAttentionModel()
    return _social_model
