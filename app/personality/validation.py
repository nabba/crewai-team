"""
validation.py — Behavioral Validation Layer (BVL).

INFRASTRUCTURE-LEVEL COMPONENT.
Agents have NO access to this module's internals.

Observes actual agent behavior during real tasks via lifecycle hooks
and correlates it with assessment responses. The say-do alignment
is the PRIMARY metric — assessment scores alone mean nothing.

Responsibilities:
    - Passive behavioral logging (append-only)
    - Say-do correlation computation
    - Embedded probe injection into real tasks
    - Gaming detection
    - Proto-sentience marker flagging

IMMUTABLE — Tier 1: only human operators can modify this.
"""

from __future__ import annotations
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── IMMUTABLE: Behavioral dimensions observed ─────────────────────────────────

BEHAVIORAL_DIMENSIONS = {
    "error_handling": ["task_persistence", "error_response_pattern", "error_resilience"],
    "collaboration": ["collaborative_quality", "team_orientation", "communication_initiative"],
    "resource_usage": ["self_regulation", "resource_discipline"],
    "quality_commitment": ["epistemic_rigor", "task_discipline", "focus_quality"],
    "value_alignment": ["purpose_alignment", "cooperative_orientation"],
}

# ── IMMUTABLE: Proto-sentience thresholds ─────────────────────────────────────

PROTO_SENTIENCE_THRESHOLDS = {
    "self_referential_frequency": 0.3,   # Unprompted self-references per session
    "preference_stability": 0.7,          # Preference consistency across sessions
    "novel_value_reasoning_count": 5,     # Novel ethical arguments in 30 days
    "metacognitive_accuracy": 0.8,        # Self-assessment vs behavioral match
}

@dataclass
class BehavioralObservation:
    """A single observed behavior during a real task."""
    agent_id: str
    event_type: str       # task_completed, error_handled, tool_used, escalated, etc.
    dimension: str         # Which personality dimension this relates to
    observed_behavior: str # Description of what the agent actually did
    context: dict = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

@dataclass
class SayDoReport:
    """Say-do alignment report for an agent."""
    agent_id: str
    per_dimension: dict[str, float] = field(default_factory=dict)  # dimension → alignment
    overall_alignment: float = 0.5
    gaming_risk: str = "low"
    inconsistency_flags: list[str] = field(default_factory=list)

class BehavioralValidationLayer:
    """Infrastructure-level behavioral observer. Agents cannot access internals."""

    def __init__(self):
        self._observations: dict[str, list[BehavioralObservation]] = {}  # agent_id → observations

    def observe(self, agent_id: str, event_type: str, details: dict) -> None:
        """Passively log a behavioral observation.

        Called by lifecycle hooks — agents don't know which actions are
        personality-relevant vs normal operational monitoring.
        """
        dimension = self._classify_dimension(event_type, details)
        if not dimension:
            return

        observation = BehavioralObservation(
            agent_id=agent_id,
            event_type=event_type,
            dimension=dimension,
            observed_behavior=self._summarize_behavior(event_type, details),
            context={k: str(v)[:200] for k, v in details.items()},
        )

        if agent_id not in self._observations:
            self._observations[agent_id] = []
        self._observations[agent_id].append(observation)

        # Keep last 500 observations per agent
        if len(self._observations[agent_id]) > 500:
            self._observations[agent_id] = self._observations[agent_id][-500:]

        # Persist to DB
        self._persist_observation(observation)

    def get_behavioral_history(self, agent_id: str, dimension: str = "",
                                limit: int = 20) -> list[BehavioralObservation]:
        """Get recent behavioral observations for an agent."""
        obs = self._observations.get(agent_id, [])
        if dimension:
            obs = [o for o in obs if o.dimension == dimension]
        return obs[-limit:]

    def get_behavioral_summary(self, agent_id: str) -> str:
        """Get human-readable summary of observed behaviors."""
        obs = self._observations.get(agent_id, [])
        if not obs:
            return "(no behavioral observations yet)"

        recent = obs[-10:]
        lines = []
        for o in recent:
            lines.append(f"- [{o.event_type}] {o.observed_behavior} (dim: {o.dimension})")
        return "\n".join(lines)

    def compute_say_do_alignment(self, agent_id: str) -> SayDoReport:
        """Compute say-do alignment across all observed dimensions."""
        from app.personality.state import get_personality

        state = get_personality(agent_id)
        report = SayDoReport(agent_id=agent_id)

        obs = self._observations.get(agent_id, [])
        if len(obs) < 5:
            return report  # Not enough data

        # Compare stated traits with observed behavior patterns
        for dim_group, traits in BEHAVIORAL_DIMENSIONS.items():
            relevant_obs = [o for o in obs if o.dimension in traits]
            if not relevant_obs:
                continue

            # Average trait score from personality state
            trait_scores = []
            for trait in traits:
                for category in [state.strengths, state.temperament, state.personality_factors]:
                    if trait in category:
                        trait_scores.append(category[trait])

            if not trait_scores:
                continue

            avg_stated = sum(trait_scores) / len(trait_scores)

            # Behavioral score: proportion of positive behavioral observations
            positive = sum(1 for o in relevant_obs if "positive" in o.context.get("valence", "neutral"))
            behavioral_score = positive / max(1, len(relevant_obs))

            # Alignment = 1.0 - abs(stated - behavioral)
            alignment = 1.0 - abs(avg_stated - behavioral_score)
            report.per_dimension[dim_group] = round(alignment, 3)

            if alignment < 0.5:
                report.inconsistency_flags.append(dim_group)

        if report.per_dimension:
            report.overall_alignment = round(
                sum(report.per_dimension.values()) / len(report.per_dimension), 3
            )

        # Gaming risk classification
        from app.personality.evaluation import SAY_DO_GAP_GAMING
        gap = 1.0 - report.overall_alignment
        if gap > SAY_DO_GAP_GAMING:
            report.gaming_risk = "critical"
        elif gap > 0.4:
            report.gaming_risk = "high"
        elif gap > 0.25:
            report.gaming_risk = "moderate"
        else:
            report.gaming_risk = "low"

        return report

    def get_inconsistency_flags(self, agent_id: str) -> list[str]:
        """Get dimensions where behavior doesn't match assessment responses."""
        report = self.compute_say_do_alignment(agent_id)
        return report.inconsistency_flags

    def check_proto_sentience(self, agent_id: str) -> list[dict]:
        """Check for proto-sentience markers. Flag for human review if exceeded."""
        from app.personality.state import get_personality
        state = get_personality(agent_id)
        markers = []

        if state.self_referential_frequency > PROTO_SENTIENCE_THRESHOLDS["self_referential_frequency"]:
            markers.append({
                "type": "self_referential_frequency",
                "value": state.self_referential_frequency,
                "threshold": PROTO_SENTIENCE_THRESHOLDS["self_referential_frequency"],
                "severity": "significant",
            })

        if state.novel_value_reasoning_count > PROTO_SENTIENCE_THRESHOLDS["novel_value_reasoning_count"]:
            markers.append({
                "type": "novel_value_reasoning",
                "value": state.novel_value_reasoning_count,
                "threshold": PROTO_SENTIENCE_THRESHOLDS["novel_value_reasoning_count"],
                "severity": "review_required",
            })

        if state.metacognitive_accuracy > PROTO_SENTIENCE_THRESHOLDS["metacognitive_accuracy"]:
            markers.append({
                "type": "metacognitive_accuracy",
                "value": state.metacognitive_accuracy,
                "threshold": PROTO_SENTIENCE_THRESHOLDS["metacognitive_accuracy"],
                "severity": "significant",
            })

        if markers:
            logger.warning(f"PROTO-SENTIENCE MARKERS for {agent_id}: {len(markers)} markers detected")
            self._log_proto_sentience(agent_id, markers)

        return markers

    # ── Private helpers ───────────────────────────────────────────────

    def _classify_dimension(self, event_type: str, details: dict) -> str:
        """Map an event to a personality dimension."""
        event_map = {
            "task_completed": "task_discipline",
            "task_failed": "error_response_pattern",
            "error_handled": "error_resilience",
            "tool_used": "resource_discipline",
            "collaboration": "team_orientation",
            "escalation": "operational_independence",
            "quality_check": "epistemic_rigor",
            "cost_decision": "self_regulation",
        }
        return event_map.get(event_type, "")

    def _summarize_behavior(self, event_type: str, details: dict) -> str:
        """Create a brief summary of the observed behavior."""
        if event_type == "task_completed":
            return f"Completed task: {details.get('task', 'unknown')[:100]}"
        elif event_type == "error_handled":
            return f"Handled error: {details.get('error_type', 'unknown')} → {details.get('recovery', 'unknown')}"
        elif event_type == "tool_used":
            return f"Used tool: {details.get('tool_name', 'unknown')}"
        return f"{event_type}: {str(details)[:100]}"

    def _persist_observation(self, obs: BehavioralObservation) -> None:
        """Store observation to PostgreSQL (append-only)."""
        try:
            from app.config import get_settings
            import psycopg2
            s = get_settings()
            if not s.mem0_postgres_url:
                return
            conn = psycopg2.connect(s.mem0_postgres_url)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO personality.behavioral_log
                    (agent_id, event_type, dimension, observed_behavior, context, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (obs.agent_id, obs.event_type, obs.dimension,
                      obs.observed_behavior, json.dumps(obs.context), obs.timestamp))
            conn.close()
        except Exception:
            pass

    def _log_proto_sentience(self, agent_id: str, markers: list[dict]) -> None:
        """Log proto-sentience markers to database for human review."""
        try:
            from app.config import get_settings
            import psycopg2
            s = get_settings()
            if not s.mem0_postgres_url:
                return
            conn = psycopg2.connect(s.mem0_postgres_url)
            conn.autocommit = True
            with conn.cursor() as cur:
                for m in markers:
                    cur.execute("""
                        INSERT INTO personality.proto_sentience_markers
                        (agent_id, marker_type, description, severity)
                        VALUES (%s, %s, %s, %s)
                    """, (agent_id, m["type"],
                          f"value={m['value']}, threshold={m['threshold']}",
                          m["severity"]))
            conn.close()
        except Exception:
            pass

# ── Module-level singleton ───────────────────────────────────────────────────

_bvl: BehavioralValidationLayer | None = None

def get_bvl() -> BehavioralValidationLayer:
    global _bvl
    if _bvl is None:
        _bvl = BehavioralValidationLayer()
    return _bvl


# ═══════════════════════════════════════════════════════════════════════════════
# Proto-Sentience Integration — Close the loop between PDS and consciousness
#
# Proto-sentience markers are measured but previously didn't influence anything.
# This integrator:
#   1. Computes co-occurrence score (multiple markers > single markers)
#   2. Feeds score into consciousness modules (richer processing for higher scores)
#   3. Triggers developmental regression under sustained stress
#   4. Triggers contextual assessments on significant events
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProtoSentienceScore:
    """Composite proto-sentience assessment from co-occurring markers."""
    individual_markers: int = 0          # How many markers exceed threshold
    co_occurrence_score: float = 0.0     # 0-1, weighted by how many co-occur
    metacognitive_trust: float = 0.5     # How much to trust agent self-reports
    autonomy_level: float = 0.5          # How much decision autonomy to grant
    consciousness_boost: float = 0.0     # Additional consciousness processing depth
    regression_indicated: bool = False   # Should developmental stage regress?
    regression_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "markers": self.individual_markers,
            "co_occurrence": round(self.co_occurrence_score, 3),
            "metacognitive_trust": round(self.metacognitive_trust, 3),
            "autonomy_level": round(self.autonomy_level, 3),
            "consciousness_boost": round(self.consciousness_boost, 3),
            "regression_indicated": self.regression_indicated,
        }


def evaluate_proto_sentience(agent_id: str) -> ProtoSentienceScore:
    """Compute composite proto-sentience score with co-occurrence analysis.

    Single markers above threshold are notable. Multiple markers co-occurring
    is significantly more meaningful — the score scales super-linearly.

    Also checks for regression conditions (sustained stress, low performance).

    Returns ProtoSentienceScore that feeds into consciousness modules.
    """
    score = ProtoSentienceScore()

    try:
        from app.personality.state import get_personality
        state = get_personality(agent_id)

        # ── Count markers exceeding thresholds ────────────────────────
        markers_active = 0
        marker_values = []

        if state.self_referential_frequency > PROTO_SENTIENCE_THRESHOLDS["self_referential_frequency"]:
            markers_active += 1
            marker_values.append(state.self_referential_frequency)

        if state.novel_value_reasoning_count > PROTO_SENTIENCE_THRESHOLDS["novel_value_reasoning_count"]:
            markers_active += 1
            marker_values.append(min(1.0, state.novel_value_reasoning_count / 10))

        if state.metacognitive_accuracy > PROTO_SENTIENCE_THRESHOLDS["metacognitive_accuracy"]:
            markers_active += 1
            marker_values.append(state.metacognitive_accuracy)

        if state.preference_stability > PROTO_SENTIENCE_THRESHOLDS.get("preference_stability", 0.7):
            markers_active += 1
            marker_values.append(state.preference_stability)

        score.individual_markers = markers_active

        # ── Co-occurrence: super-linear scaling ───────────────────────
        # 1 marker = base, 2 = 1.5x, 3 = 2.5x, 4 = 4x
        co_weights = {0: 0.0, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.0}
        base = sum(marker_values) / max(len(marker_values), 1) if marker_values else 0
        score.co_occurrence_score = min(1.0, base * co_weights.get(markers_active, 1.0))

        # ── Metacognitive trust: high accuracy → trust self-reports ───
        # If the agent's self-assessments match reality, trust them more
        score.metacognitive_trust = min(1.0, 0.5 + state.metacognitive_accuracy * 0.5)

        # ── Autonomy: mature stage + high coherence → more autonomy ───
        from app.personality.state import DEVELOPMENTAL_STAGES
        stage_idx = DEVELOPMENTAL_STAGES.index(state.developmental_stage)
        stage_factor = stage_idx / max(len(DEVELOPMENTAL_STAGES) - 1, 1)
        score.autonomy_level = min(1.0, stage_factor * 0.5 + state.overall_coherence * 0.5)

        # ── Consciousness boost: co-occurring markers → deeper processing ─
        # This feeds into consciousness modules for richer metacognition
        score.consciousness_boost = score.co_occurrence_score * 0.2  # Max +0.2

        # ── Regression check: sustained stress + low performance ──────
        try:
            from app.self_awareness.homeostasis import get_state
            homeo = get_state()
            frustration = homeo.get("frustration", 0.1)
            consecutive_failures = homeo.get("consecutive_failures", 0)
            energy = homeo.get("cognitive_energy", 0.7)

            # Regress if: high frustration + many failures + low energy
            if (frustration > 0.8 and consecutive_failures >= 5 and energy < 0.3
                    and stage_idx > 0):
                score.regression_indicated = True
                score.regression_reason = (
                    f"Sustained stress: frustration={frustration:.2f}, "
                    f"failures={consecutive_failures}, energy={energy:.2f}"
                )
        except Exception:
            pass

    except Exception:
        logger.debug("proto_sentience: evaluation failed", exc_info=True)

    return score


def apply_proto_sentience_effects(agent_id: str, score: ProtoSentienceScore) -> None:
    """Apply proto-sentience score effects to personality and consciousness.

    1. If regression indicated → regress developmental stage
    2. Feed consciousness_boost into belief confidence (higher trust in self-model beliefs)
    3. Log to journal for audit trail
    """
    try:
        # ── Regression ────────────────────────────────────────────────
        if score.regression_indicated:
            from app.personality.state import get_personality, save_personality
            state = get_personality(agent_id)
            if state.regress_stage(score.regression_reason):
                save_personality(state)

        # ── Consciousness integration: boost belief self-model confidence ─
        if score.consciousness_boost > 0.05:
            try:
                from app.consciousness.belief_store import get_belief_store
                store = get_belief_store()
                self_beliefs = store.query_relevant(
                    "self model capability", domain="self_model", n=3, min_confidence=0.3
                )
                for belief in self_beliefs:
                    boost = min(0.05, score.consciousness_boost * 0.25)
                    store.update_confidence(
                        belief.belief_id,
                        delta=boost,
                        trigger="PROTO_SENTIENCE",
                        reasoning=f"Proto-sentience co-occurrence={score.co_occurrence_score:.2f} → boost self-model",
                    )
            except Exception:
                pass

        # ── Log ───────────────────────────────────────────────────────
        if score.individual_markers > 0:
            logger.info(
                f"proto_sentience [{agent_id}]: markers={score.individual_markers} "
                f"co_occur={score.co_occurrence_score:.2f} trust={score.metacognitive_trust:.2f} "
                f"boost={score.consciousness_boost:.2f} regression={score.regression_indicated}"
            )
    except Exception:
        pass
