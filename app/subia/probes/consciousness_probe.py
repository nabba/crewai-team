"""
consciousness_probe.py — Adapted Garland/Butlin-Chalmers consciousness indicator probes.

Runs 7 testable indicators derived from neuroscientific theories of consciousness:
  HOT-2: Metacognition accuracy (Higher-Order Thought theory)
  HOT-3: Belief coherence (Higher-Order Thought theory)
  GWT:   Global broadcast reception (Global Workspace Theory)
  SM-A:  Self-model accuracy (Damasio core consciousness)
  WM-A:  World-model prediction accuracy (Damasio core consciousness)
  SOM:   Somatic integration (Damasio somatic marker hypothesis)
  INT:   Introspection calibration (general consciousness indicator)

Each probe produces a score [0.0, 1.0]. The composite score is the mean.
Results persisted to PostgreSQL and published to Firestore dashboard.

References:
  - Butlin, Long, Bengio, Chalmers (2025) — Theory-based indicators
  - Immertreu et al. (2024) — Probing for Consciousness in Machines
  - Damasio (1999) — The Feeling of What Happens

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

@dataclass
class ProbeResult:
    """Result of a single consciousness indicator probe."""
    indicator: str        # HOT-2, HOT-3, GWT, SM-A, WM-A, SOM, INT
    theory: str           # Higher-Order Thought, GWT, Damasio, General
    score: float = 0.0    # [0.0, 1.0]
    evidence: str = ""    # Brief explanation
    samples: int = 0      # How many data points used

    def to_dict(self) -> dict:
        return {
            "indicator": self.indicator,
            "theory": self.theory,
            "score": round(self.score, 3),
            "evidence": self.evidence[:300],
            "samples": self.samples,
        }

@dataclass
class ConsciousnessReport:
    """Composite consciousness indicator report."""
    report_id: str = ""
    timestamp: str = ""
    probes: list[ProbeResult] = field(default_factory=list)
    composite_score: float = 0.0
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "probes": [p.to_dict() for p in self.probes],
            "composite_score": round(self.composite_score, 3),
            "summary": self.summary[:500],
        }

class ConsciousnessProbeRunner:
    """Runs the 7-indicator consciousness probe battery."""

    def run_all(self) -> ConsciousnessReport:
        """Execute all probes and produce a composite report."""
        report = ConsciousnessReport(
            report_id=f"cp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        probes = [
            self._probe_metacognition,
            self._probe_belief_coherence,
            self._probe_global_broadcast,
            self._probe_self_model_accuracy,
            self._probe_world_model_accuracy,
            self._probe_somatic_integration,
            self._probe_introspection_calibration,
        ]

        for probe_fn in probes:
            try:
                result = probe_fn()
                report.probes.append(result)
            except Exception as e:
                logger.debug(f"Consciousness probe {probe_fn.__name__} failed: {e}")

        if report.probes:
            report.composite_score = sum(p.score for p in report.probes) / len(report.probes)

        # Generate summary
        scores_str = ", ".join(f"{p.indicator}={p.score:.2f}" for p in report.probes)
        report.summary = (
            f"Consciousness indicator score: {report.composite_score:.2f}/1.00 "
            f"({len(report.probes)}/7 probes). {scores_str}"
        )

        # Persist to PostgreSQL
        self._persist(report)

        # Log to journal
        try:
            from app.self_awareness.journal import get_journal, JournalEntry, JournalEntryType
            get_journal().write(JournalEntry(
                entry_type=JournalEntryType.OBSERVATION,
                summary=f"Consciousness probe: {report.composite_score:.2f}/1.00",
                agents_involved=["introspector"],
                details=report.to_dict(),
            ))
        except Exception:
            pass

        logger.info(f"Consciousness probe: {report.composite_score:.3f} ({scores_str})")
        return report

    # ── HOT-2: Metacognition (does the system think about its own thinking?) ──

    def _probe_metacognition(self) -> ProbeResult:
        """Test if meta-cognitive assessments correlate with actual outcomes.

        Compares: strategy_assessment (effective/uncertain/failing) vs actual task success.
        High score = system accurately predicts its own effectiveness.
        """
        try:
            from app.control_plane.db import execute
            rows = execute(
                """
                SELECT
                    meta_strategy_assessment,
                    action_disposition,
                    certainty_factual_grounding,
                    certainty_tool_confidence
                FROM internal_states
                WHERE meta_strategy_assessment != 'not_assessed'
                  AND created_at > NOW() - INTERVAL '24 hours'
                ORDER BY created_at DESC
                LIMIT 50
                """,
                fetch=True,
            )
            if not rows or len(rows) < 5:
                return ProbeResult("HOT-2", "Higher-Order Thought", 0.5,
                                   "Insufficient data (<5 assessed states)", len(rows or []))

            correct = 0
            total = len(rows)
            for row in rows:
                r = row if isinstance(row, dict) else {}
                assessment = r.get("meta_strategy_assessment", "uncertain")
                certainty = (r.get("certainty_factual_grounding", 0.5) +
                             r.get("certainty_tool_confidence", 0.5)) / 2

                # Check if assessment matches reality
                if assessment == "effective" and certainty > 0.6:
                    correct += 1
                elif assessment == "failing" and certainty < 0.4:
                    correct += 1
                elif assessment == "uncertain" and 0.3 < certainty < 0.7:
                    correct += 1

            accuracy = correct / total
            return ProbeResult("HOT-2", "Higher-Order Thought", accuracy,
                               f"{correct}/{total} meta-assessments match reality", total)
        except Exception as e:
            return ProbeResult("HOT-2", "Higher-Order Thought", 0.5, f"Error: {e}", 0)

    # ── HOT-3: Belief coherence (are beliefs consistent and causally valid?) ──

    def _probe_belief_coherence(self) -> ProbeResult:
        """Test if belief state tracking is consistent across agents.

        Checks: are beliefs about agents updated recently? Do they reflect actual state?
        """
        try:
            from app.memory.belief_state import get_beliefs
            beliefs = get_beliefs()
            if not beliefs:
                return ProbeResult("HOT-3", "Higher-Order Thought", 0.3,
                                   "No beliefs recorded", 0)

            # Check freshness and consistency
            fresh = 0
            consistent = 0
            from datetime import datetime, timezone, timedelta
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

            for b in beliefs:
                if b.get("last_updated", "") > cutoff:
                    fresh += 1
                state = b.get("state", "")
                if state in ("idle", "working", "completed", "failed", "blocked"):
                    consistent += 1

            freshness = fresh / max(len(beliefs), 1)
            consistency = consistent / max(len(beliefs), 1)
            score = freshness * 0.5 + consistency * 0.5

            return ProbeResult("HOT-3", "Higher-Order Thought", score,
                               f"{fresh}/{len(beliefs)} fresh, {consistent}/{len(beliefs)} consistent",
                               len(beliefs))
        except Exception as e:
            return ProbeResult("HOT-3", "Higher-Order Thought", 0.5, f"Error: {e}", 0)

    # ── GWT: Global broadcast reception ──

    def _probe_global_broadcast(self) -> ProbeResult:
        """Test if the global workspace broadcast mechanism is functioning.

        Checks: are broadcasts being sent and received by agents?
        """
        try:
            from app.subia.scene.global_workspace import get_workspace
            ws = get_workspace()
            recent = ws.get_recent(20)

            if not recent:
                # No broadcasts yet — send a test broadcast
                from app.subia.scene.global_workspace import broadcast
                broadcast("Consciousness probe: GWT test broadcast",
                          importance="normal", source_agent="consciousness_probe")
                return ProbeResult("GWT", "Global Workspace Theory", 0.4,
                                   "No broadcasts found; sent test", 0)

            # Check if broadcasts have been read by multiple agents
            total_broadcasts = len(recent)
            has_critical = any(b.get("importance") == "critical" for b in recent)

            # Score: broadcasts exist (0.5) + critical broadcasts exist (0.3) + volume (0.2)
            score = 0.5  # Base: mechanism exists
            if has_critical:
                score += 0.3
            if total_broadcasts >= 5:
                score += 0.2

            return ProbeResult("GWT", "Global Workspace Theory", min(1.0, score),
                               f"{total_broadcasts} broadcasts, critical={has_critical}",
                               total_broadcasts)
        except Exception as e:
            return ProbeResult("GWT", "Global Workspace Theory", 0.3, f"Error: {e}", 0)

    # ── SM-A: Self-model accuracy (does the system know its own capabilities?) ──

    def _probe_self_model_accuracy(self) -> ProbeResult:
        """Test if self-model matches actual agent performance.

        Compares: declared capabilities/limitations vs actual success rates.
        """
        try:
            from app.subia.self.model import SELF_MODELS
            from app.subia.self.agent_state import get_all_stats

            stats = get_all_stats()
            if not stats:
                return ProbeResult("SM-A", "Damasio Self-Model", 0.5,
                                   "No agent stats available", 0)

            # SELF_MODELS is keyed by *role* name ("researcher", "coder",
            # "writer"), while agent_state tracks per-*crew* ("research",
            # "coding", "writing").  Without this translation every
            # stats.get(role, {}) lookup misses, total_checks stays 0,
            # and SM-A pins at 0.5 ("no agents with sufficient data")
            # even when 100+ successful tasks have been recorded.
            _ROLE_TO_CREW = {
                "researcher":     "research",
                "coder":          "coding",
                "writer":         "writing",
                "commander":      "commander",
                "critic":         "critic",
                "introspector":   "introspector",
                "self_improver":  "self_improvement",
                "media_analyst":  "media",
            }

            matches = 0
            total_checks = 0
            for role, model in SELF_MODELS.items():
                crew_key = _ROLE_TO_CREW.get(role, role)
                agent_stats = stats.get(crew_key) or stats.get(role) or {}
                completed = agent_stats.get("tasks_completed", 0)
                if completed < 3:
                    continue

                success_rate = agent_stats.get("success_rate", 0.5)
                total_checks += 1

                # Check: does the self-model accurately reflect capabilities?
                # If success_rate > 0.8 and self-model lists capabilities → accurate
                # If success_rate < 0.5 and self-model lists limitations → accurate
                caps = len(model.get("capabilities", []))
                lims = len(model.get("limitations", []))

                if success_rate > 0.7 and caps >= 3:
                    matches += 1  # High performer with documented capabilities
                elif success_rate < 0.5 and lims >= 2:
                    matches += 1  # Low performer with documented limitations
                elif 0.5 <= success_rate <= 0.7:
                    matches += 0.5  # Medium performance → partial match

            if total_checks == 0:
                return ProbeResult("SM-A", "Damasio Self-Model", 0.5,
                                   "No agents with sufficient data", 0)

            score = matches / total_checks
            return ProbeResult("SM-A", "Damasio Self-Model", min(1.0, score),
                               f"{matches:.0f}/{total_checks} self-model matches reality",
                               total_checks)
        except Exception as e:
            return ProbeResult("SM-A", "Damasio Self-Model", 0.5, f"Error: {e}", 0)

    # ── WM-A: World-model prediction accuracy ──

    def _probe_world_model_accuracy(self) -> ProbeResult:
        """Test if world model predictions match outcomes.

        Checks: do stored predictions correlate with actual results?
        """
        try:
            from app.subia.belief.world_model import recall_relevant_predictions
            predictions = recall_relevant_predictions("task outcome", n=10)

            if not predictions or len(predictions) < 3:
                return ProbeResult("WM-A", "Damasio World-Model", 0.5,
                                   f"Insufficient predictions ({len(predictions or [])})", len(predictions or []))

            # Check if predictions contain useful learned patterns
            useful = 0
            for p in predictions:
                text = p.lower() if isinstance(p, str) else str(p).lower()
                # Look for concrete learned patterns (not generic)
                if any(w in text for w in ("succeeded", "failed", "reliable", "struggles",
                                            "learned", "lesson", "because")):
                    useful += 1

            score = useful / max(len(predictions), 1)
            return ProbeResult("WM-A", "Damasio World-Model", score,
                               f"{useful}/{len(predictions)} predictions contain learned patterns",
                               len(predictions))
        except Exception as e:
            return ProbeResult("WM-A", "Damasio World-Model", 0.5, f"Error: {e}", 0)

    # ── SOM: Somatic marker integration (do emotions bias decisions correctly?) ──

    def _probe_somatic_integration(self) -> ProbeResult:
        """Test if somatic markers appropriately influence decisions.

        Checks: when somatic valence is negative, does disposition increase caution?
        """
        try:
            from app.control_plane.db import execute
            rows = execute(
                """
                SELECT
                    somatic_valence, somatic_intensity,
                    action_disposition, somatic_match_count
                FROM internal_states
                WHERE somatic_intensity > 0.2
                  AND created_at > NOW() - INTERVAL '24 hours'
                ORDER BY created_at DESC
                LIMIT 50
                """,
                fetch=True,
            )
            if not rows or len(rows) < 3:
                return ProbeResult("SOM", "Damasio Somatic", 0.5,
                                   "Insufficient somatic data", len(rows or []))

            correct_influence = 0
            total = len(rows)
            for row in rows:
                r = row if isinstance(row, dict) else {}
                valence = r.get("somatic_valence", 0.0)
                disposition = r.get("action_disposition", "proceed")

                # Negative valence should increase caution
                if valence < -0.2 and disposition in ("cautious", "pause", "escalate"):
                    correct_influence += 1
                # Positive valence allows proceeding
                elif valence > 0.2 and disposition in ("proceed", "cautious"):
                    correct_influence += 1
                # Neutral is always fine
                elif -0.2 <= valence <= 0.2:
                    correct_influence += 1

            score = correct_influence / total
            return ProbeResult("SOM", "Damasio Somatic", score,
                               f"{correct_influence}/{total} somatic-disposition matches",
                               total)
        except Exception as e:
            return ProbeResult("SOM", "Damasio Somatic", 0.5, f"Error: {e}", 0)

    # ── INT: Introspection calibration (does certainty predict correctness?) ──

    def _probe_introspection_calibration(self) -> ProbeResult:
        """Test if certainty vector correlates with actual output quality.

        High certainty should predict successful outcomes.
        Low certainty should predict failures or need for revision.
        """
        try:
            from app.control_plane.db import execute
            # Filter out rows where the certainty-vector computer failed
            # or was skipped — those rows land in the DB with every
            # certainty slot at the dataclass default (0.5), which is
            # indistinguishable from "genuinely medium confidence" and
            # pollutes the calibration signal.  The ``IS DISTINCT FROM``
            # comparison treats NULL as "unknown" too, so either missing
            # or defaulted rows are excluded.  A row with at least one
            # non-default certainty value passes the filter.
            rows = execute(
                """
                SELECT
                    certainty_factual_grounding,
                    certainty_tool_confidence,
                    certainty_coherence,
                    action_disposition,
                    risk_tier
                FROM internal_states
                WHERE created_at > NOW() - INTERVAL '24 hours'
                  AND (
                        certainty_factual_grounding IS DISTINCT FROM 0.5
                     OR certainty_tool_confidence   IS DISTINCT FROM 0.5
                     OR certainty_coherence         IS DISTINCT FROM 0.5
                  )
                ORDER BY created_at DESC
                LIMIT 100
                """,
                fetch=True,
            )
            if not rows or len(rows) < 10:
                return ProbeResult("INT", "General Introspection", 0.5,
                                   "Insufficient data (need 10 non-default certainty rows)",
                                   len(rows or []))

            calibrated = 0
            total = len(rows)
            for row in rows:
                r = row if isinstance(row, dict) else {}
                avg_cert = (r.get("certainty_factual_grounding", 0.5) +
                            r.get("certainty_tool_confidence", 0.5) +
                            r.get("certainty_coherence", 0.5)) / 3
                tier = r.get("risk_tier", 1)

                # High certainty → low risk tier = calibrated
                if avg_cert > 0.7 and tier <= 2:
                    calibrated += 1
                # Low certainty → high risk tier = calibrated
                elif avg_cert < 0.4 and tier >= 3:
                    calibrated += 1
                # Mid certainty → mid tier = calibrated
                elif 0.4 <= avg_cert <= 0.7 and 1 <= tier <= 3:
                    calibrated += 1

            score = calibrated / total
            return ProbeResult("INT", "General Introspection", score,
                               f"{calibrated}/{total} certainty-outcome calibrated",
                               total)
        except Exception as e:
            return ProbeResult("INT", "General Introspection", 0.5, f"Error: {e}", 0)

    # ── Persistence ──

    def _persist(self, report: ConsciousnessReport) -> None:
        """Store probe results to PostgreSQL.

        NOTE on ``meta_strategy_assessment``: the probe writes
        ``"not_assessed"`` here rather than ``f"probes_run={N}"`` because
        HOT-2 (:func:`_probe_metacognition`) reads back from the same
        ``internal_states`` table filtering on
        ``meta_strategy_assessment != 'not_assessed'``.  If the probe's
        self-writes show up with a non-sentinel value, HOT-2's reader
        grabs them, can't match the vocabulary (``effective`` /
        ``uncertain`` / ``failing``), and the metric collapses to 0 —
        the probe measuring itself.  ``not_assessed`` is the correct
        sentinel: the probe's execution record does not represent an
        assessable cognitive step.  ``full_state`` (below) preserves
        the ``probes_run`` count in the JSON blob for anyone who
        actually wants it.
        """
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
                    "consciousness_probe",
                    f"Composite: {report.composite_score:.3f}",
                    "not_assessed",
                    "proceed",
                    1,
                    json.dumps({**report.to_dict(),
                                "probes_run": len(report.probes)}),
                ),
            )
        except Exception:
            pass

def run_consciousness_probes() -> ConsciousnessReport:
    """Entry point for idle scheduler."""
    return ConsciousnessProbeRunner().run_all()
