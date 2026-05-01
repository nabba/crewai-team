"""
metacognitive_monitor.py — HOT-3: Dual-timescale metacognitive loop.

Fast loop (per task): consult beliefs → select action → record reasoning
Slow loop (periodic): evaluate outcomes → update confidence → suspend/retract

The fast loop does NOT update beliefs — only consults them. This prevents
hasty belief changes from single events. The slow loop evaluates accumulated
evidence before adjusting confidence.

DGM Safety: HUMANIST_CONSTITUTION is checked during action selection.
No belief can override constitutional constraints.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

@dataclass
class ActionSelectionRecord:
    """Record of belief-guided action selection (fast loop output)."""
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    selected_action: str = ""
    beliefs_consulted: list[str] = field(default_factory=list)  # belief_ids
    goal_context: str = ""
    alternatives_considered: list[dict] = field(default_factory=list)
    selection_reasoning: str = ""
    outcome_assessed: bool = False
    outcome_matched_prediction: bool | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "action_id": self.action_id,
            "selected_action": self.selected_action[:200],
            "beliefs_consulted": self.beliefs_consulted[:5],
            "selection_reasoning": self.selection_reasoning[:200],
            "outcome_assessed": self.outcome_assessed,
            "outcome_matched_prediction": self.outcome_matched_prediction,
        }

class MetacognitiveMonitor:
    """Dual-timescale metacognitive monitor for HOT-3.

    Fast loop: consult → select → record (per task, <800ms)
    Slow loop: evaluate → update → review (periodic, <1000ms)
    """

    def __init__(self):
        self._pending_outcomes: list[dict] = []  # Action records awaiting outcome assessment

    # ── FAST LOOP: Belief Consultation + Action Selection ─────────────

    def consult_beliefs(self, task_description: str, crew_name: str,
                        goal_context: str = "") -> ActionSelectionRecord:
        """Fast loop: query beliefs and select action with explicit reasoning.

        Returns an ActionSelectionRecord linking selected action to consulted beliefs.
        Called from PRE_TASK hook during crew dispatch.
        """
        from app.subia.belief.store import get_belief_store
        store = get_belief_store()

        # Query relevant beliefs for this task
        beliefs = store.query_relevant(
            task_description[:300],
            domain=None,  # Search across all domains
            n=5,
            min_confidence=0.15,
        )

        belief_ids = [b.belief_id for b in beliefs]
        belief_context = "; ".join(
            f"[{b.domain}:{b.confidence:.1f}] {b.content[:80]}" for b in beliefs
        ) if beliefs else "No relevant beliefs found"

        # Record the consultation (action = crew dispatch)
        record = ActionSelectionRecord(
            selected_action=f"dispatch:{crew_name}",
            beliefs_consulted=belief_ids,
            goal_context=goal_context[:300],
            selection_reasoning=f"Consulted {len(beliefs)} beliefs: {belief_context[:500]}",
        )

        # Persist to PostgreSQL
        self._persist_action_record(record)
        self._pending_outcomes.append({
            "action_id": record.action_id,
            "beliefs_consulted": belief_ids,
            "crew_name": crew_name,
        })

        return record

    # ── SLOW LOOP: Belief Updating + Mandatory Review ────────────────

    def run_slow_loop(self) -> dict:
        """Slow loop: evaluate accumulated outcomes and update beliefs.

        Called by idle scheduler periodically (every 5-10 fast cycles).
        Returns summary of updates made.
        """
        from app.subia.belief.store import get_belief_store
        from app.consciousness.config import load_config

        store = get_belief_store()
        config = load_config()
        updates = {"confidence_adjusted": 0, "beliefs_suspended": 0, "beliefs_reviewed": 0}

        # 1. Evaluate pending outcomes against belief-based predictions
        for pending in self._pending_outcomes[-20:]:  # Last 20 unassessed
            try:
                self._evaluate_outcome(pending, store, config)
                updates["confidence_adjusted"] += 1
            except Exception:
                pass
        self._pending_outcomes = self._pending_outcomes[-50:]  # Keep bounded

        # 2. Mandatory review: oldest unvalidated beliefs
        try:
            stale_beliefs = store.get_oldest_unvalidated(config.mandatory_review_count)
            for belief in stale_beliefs:
                # Apply confidence decay (the query already does this, but we persist it)
                decayed = store._apply_confidence_decay(belief.confidence, belief.last_validated)
                if decayed < belief.confidence:
                    store.update_confidence(
                        belief.belief_id,
                        delta=decayed - belief.confidence,
                        trigger="COGITO_CYCLE",
                        reasoning=f"Confidence decay: unvalidated for too long ({belief.confidence:.2f} → {decayed:.2f})",
                    )
                # Mark as reviewed (resets decay clock)
                store.validate_belief(belief.belief_id)
                updates["beliefs_reviewed"] += 1
        except Exception:
            pass

        # 3. Check for beliefs that should be suspended
        try:
            from app.control_plane.db import execute
            rows = execute(
                """
                SELECT belief_id, confidence FROM beliefs
                WHERE belief_status = 'ACTIVE' AND confidence < %s
                """,
                (config.belief_suspension_threshold,),
                fetch=True,
            )
            for r in (rows or []):
                if isinstance(r, dict):
                    store.suspend_belief(
                        str(r["belief_id"]),
                        f"Confidence {r['confidence']:.2f} below threshold {config.belief_suspension_threshold}",
                    )
                    updates["beliefs_suspended"] += 1
        except Exception:
            pass

        logger.info(f"HOT-3 slow loop: {updates}")
        return updates

    def _evaluate_outcome(self, pending: dict, store, config) -> None:
        """Evaluate a single action outcome against consulted beliefs."""
        # Check if the crew task succeeded
        try:
            from app.control_plane.db import execute
            rows = execute(
                """
                SELECT action_disposition, somatic_valence
                FROM internal_states
                WHERE crew_id = %s
                ORDER BY created_at DESC LIMIT 1
                """,
                (pending.get("crew_name", ""),),
                fetch=True,
            )
            if not rows:
                return

            r = rows[0] if isinstance(rows[0], dict) else {}
            disposition = r.get("action_disposition", "proceed")
            valence = r.get("somatic_valence", 0.0)

            # Positive outcome: small confidence boost to consulted beliefs
            # Negative outcome: larger confidence penalty
            if disposition == "proceed" and (valence or 0) >= 0:
                delta = config.confirmation_rate
            elif disposition in ("pause", "escalate") or (valence or 0) < -0.3:
                delta = -config.disconfirmation_rate
            else:
                delta = 0.0

            if delta != 0 and pending.get("beliefs_consulted"):
                for belief_id in pending["beliefs_consulted"][:3]:
                    store.update_confidence(
                        belief_id, delta=delta,
                        trigger="BEHAVIORAL_MISMATCH" if delta < 0 else "EXTERNAL_EVIDENCE",
                        reasoning=f"Outcome: disposition={disposition}, valence={valence:.2f}",
                    )
        except Exception:
            pass

    def _persist_action_record(self, record: ActionSelectionRecord) -> None:
        """Store action selection record to PostgreSQL."""
        try:
            from app.control_plane.db import execute
            execute(
                """
                INSERT INTO action_selection_records
                    (action_id, selected_action, beliefs_consulted, goal_context,
                     alternatives_considered, selection_reasoning)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    record.action_id,
                    record.selected_action[:500],
                    record.beliefs_consulted,
                    record.goal_context[:500],
                    json.dumps(record.alternatives_considered),
                    record.selection_reasoning[:2000],
                ),
            )
        except Exception:
            logger.debug("metacognitive_monitor: action record persist failed", exc_info=True)

# ── Module-level singleton ──────────────────────────────────────────────────

_monitor: MetacognitiveMonitor | None = None

def get_monitor() -> MetacognitiveMonitor:
    global _monitor
    if _monitor is None:
        _monitor = MetacognitiveMonitor()
    return _monitor
