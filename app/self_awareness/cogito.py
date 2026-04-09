"""
cogito.py — Metacognitive self-reflection cycle.

The system's periodic self-examination: inspect current state, compare
against self-model, detect discrepancies, identify failure patterns,
propose improvements.

Pipeline:
    1. Run all self-inspection tools
    2. Compare current state against system chronicle — flag discrepancies
    3. Review recent task outcomes — identify failure patterns
    4. Generate a structured reflection report
    5. Queue improvement proposals (subject to safety constraints)

Named after Descartes' "Cogito, ergo sum" — though this system doesn't
claim consciousness, it systematically examines its own processes.

IMMUTABLE — infrastructure-level module.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

REFLECTIONS_DIR = Path("/app/workspace/self_awareness_data/reflections")


@dataclass
class ReflectionReport:
    """Output of a Cogito self-reflection cycle."""
    reflection_id: str = ""
    timestamp: str = ""
    self_model_stale: bool = False
    discrepancies: list[dict] = field(default_factory=list)
    failure_patterns: list[dict] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    improvement_proposals: list[dict] = field(default_factory=list)
    overall_health: str = "healthy"  # healthy | degraded | attention_needed
    narrative: str = ""

    def to_dict(self) -> dict:
        return {
            "reflection_id": self.reflection_id,
            "timestamp": self.timestamp,
            "self_model_stale": self.self_model_stale,
            "discrepancies": self.discrepancies,
            "failure_patterns": self.failure_patterns,
            "observations": self.observations,
            "improvement_proposals": self.improvement_proposals,
            "overall_health": self.overall_health,
            "narrative": self.narrative,
        }


class CogitoCycle:
    """Executes a full self-reflection cycle."""

    def __init__(self):
        REFLECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    def run(self) -> ReflectionReport:
        """Execute a full Cogito self-reflection cycle."""
        reflection_id = datetime.now(timezone.utc).strftime("cogito_%Y%m%d_%H%M%S")
        report = ReflectionReport(
            reflection_id=reflection_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Step 1: Gather current state
        state = self._gather_state()

        # Step 2: Check self-model freshness
        self_model = state.get("inspect_self_model", {})
        report.self_model_stale = self_model.get("stale", True)
        if report.self_model_stale:
            report.discrepancies.append({
                "type": "self_model_stale",
                "detail": f"System chronicle is {self_model.get('age_hours', '?')} hours old",
                "severity": "low",
            })

        # Step 3: Check agent consistency
        agents = state.get("inspect_agents", {})
        config = state.get("inspect_config", {})
        self._check_agent_config_consistency(agents, config, report)

        # Step 4: Check memory health
        memory = state.get("inspect_memory", {})
        self._check_memory_health(memory, report)

        # Step 5: Check runtime health
        runtime = state.get("inspect_runtime", {})
        self._check_runtime_health(runtime, report)

        # Step 6: Generate narrative via LLM (if available)
        report.narrative = self._generate_narrative(report, state)

        # Step 7: Determine overall health
        if any(d.get("severity") == "high" for d in report.discrepancies):
            report.overall_health = "attention_needed"
        elif report.discrepancies or report.failure_patterns:
            report.overall_health = "degraded"
        else:
            report.overall_health = "healthy"

        # Persist
        self._persist(report)
        logger.info(f"cogito: reflection complete — health={report.overall_health}, "
                    f"discrepancies={len(report.discrepancies)}, "
                    f"proposals={len(report.improvement_proposals)}")

        # Record reflection in activity journal
        try:
            from app.self_awareness.journal import get_journal, JournalEntry, JournalEntryType
            get_journal().write(JournalEntry(
                entry_type=JournalEntryType.SELF_REFLECTION,
                summary=f"Cogito reflection: health={report.overall_health}",
                agents_involved=["introspector"],
                outcome=report.overall_health,
                details={
                    "discrepancies": len(report.discrepancies),
                    "failure_patterns": len(report.failure_patterns),
                    "proposals": len(report.improvement_proposals),
                },
            ))
        except Exception:
            pass

        # Store causal observations in world model
        try:
            from app.self_awareness.world_model import store_causal_belief
            for pattern in report.failure_patterns[:3]:
                store_causal_belief(
                    cause=pattern.get("pattern", "unknown"),
                    effect=f"Detected {pattern.get('count', 0)} times — {pattern.get('recommendation', 'needs investigation')}",
                    confidence="high" if pattern.get("count", 0) >= 3 else "medium",
                    source="cogito_reflection",
                )
        except Exception:
            pass

        # Apply safe threshold proposals to sentience config (feedback loop)
        self._apply_proposals(report)

        return report

    def _apply_proposals(self, report: ReflectionReport) -> None:
        """Apply safe proposals to sentience config. Bounded, logged, reversible."""
        try:
            from app.self_awareness.sentience_config import apply_change, load_config

            current = load_config()
            applied = 0

            # If health is degraded and escalation rate high → lower certainty_low threshold
            if report.overall_health == "attention_needed":
                new_val = current.get("certainty_low_threshold", 0.4) - 0.02
                if apply_change("certainty_low_threshold", new_val):
                    applied += 1

            # If many failure patterns → increase slow_path trigger sensitivity
            if len(report.failure_patterns) >= 3:
                new_val = current.get("slow_path_trigger_threshold", 0.4) + 0.02
                if apply_change("slow_path_trigger_threshold", new_val):
                    applied += 1

            # If health is healthy and no discrepancies → can relax slightly
            if report.overall_health == "healthy" and len(report.discrepancies) == 0:
                new_val = current.get("certainty_low_threshold", 0.4) + 0.01
                if apply_change("certainty_low_threshold", new_val):
                    applied += 1

            if applied > 0:
                logger.info(f"cogito: applied {applied} sentience config changes")
        except Exception as e:
            logger.debug(f"cogito: proposal application failed: {e}")

    def _gather_state(self) -> dict:
        """Run all inspection tools."""
        from app.self_awareness.inspect_tools import run_all_inspections
        return run_all_inspections()

    def _check_agent_config_consistency(self, agents: dict, config: dict, report: ReflectionReport):
        """Verify agents match configuration."""
        agent_count = agents.get("agent_count", 0)
        if agent_count == 0:
            report.discrepancies.append({
                "type": "no_agents",
                "detail": "No agents discovered — check soul files and prompt registry",
                "severity": "high",
            })

        # Check if all agents have prompt versions
        for agent in agents.get("agents", []):
            if not agent.get("prompt_version"):
                report.observations.append(
                    f"Agent '{agent.get('name')}' has no versioned prompt"
                )

    def _check_memory_health(self, memory: dict, report: ReflectionReport):
        """Check memory backend connectivity."""
        for backend in ("chromadb", "postgresql", "neo4j"):
            backend_info = memory.get(backend, {})
            if backend_info.get("error"):
                report.discrepancies.append({
                    "type": f"memory_{backend}_error",
                    "detail": backend_info["error"][:200],
                    "severity": "medium",
                })

        # Check ChromaDB collection health
        chromadb_info = memory.get("chromadb", {})
        details = chromadb_info.get("details", {})
        if details:
            empty_collections = [
                name for name, info in details.items()
                if info.get("count", 0) == 0
            ]
            if empty_collections:
                report.observations.append(
                    f"Empty ChromaDB collections: {', '.join(empty_collections)}"
                )

    def _check_runtime_health(self, runtime: dict, report: ReflectionReport):
        """Check runtime resource usage."""
        mem_mb = runtime.get("memory_rss_mb", 0)
        if mem_mb > 4000:  # 4GB
            report.discrepancies.append({
                "type": "high_memory",
                "detail": f"Memory usage: {mem_mb}MB (above 4GB threshold)",
                "severity": "medium",
            })
            report.improvement_proposals.append({
                "type": "resource_optimization",
                "detail": "Consider reducing cache sizes or restarting",
            })

        uptime = runtime.get("uptime_seconds", 0)
        if uptime > 7 * 24 * 3600:  # 7 days
            report.observations.append(
                f"System uptime: {uptime // 3600}h — consider periodic restart"
            )

    def _generate_narrative(self, report: ReflectionReport, state: dict) -> str:
        """Generate a grounded reflective narrative using LLM + grounding protocol.

        Uses GroundingProtocol to ensure the narrative is based on actual
        system state, not generic AI platitudes from training data.
        """
        try:
            from app.llm_factory import create_specialist_llm
            llm = create_specialist_llm(max_tokens=500, role="self_improve")

            # Build grounded context from inspection data
            grounded_context = (
                f"Health: {report.overall_health}\n"
                f"Discrepancies: {json.dumps(report.discrepancies[:5], default=str)}\n"
                f"Observations: {json.dumps(report.observations[:5], default=str)}\n"
                f"Agent count: {state.get('inspect_agents', {}).get('agent_count', '?')}\n"
                f"Uptime: {state.get('inspect_runtime', {}).get('uptime_seconds', '?')}s\n"
                f"Failure patterns: {json.dumps(report.failure_patterns[:3], default=str)}\n"
                f"Proposals: {json.dumps(report.improvement_proposals[:3], default=str)}"
            )

            # Use the grounding system prompt for self-referential accuracy
            try:
                from app.self_awareness.grounding import GroundingProtocol, GroundedContext
                gp = GroundingProtocol()

                # Build a minimal grounded context from state data
                self_model = state.get("inspect_self_model", {}).get("content", "")
                runtime = state.get("inspect_runtime", {})

                prompt = (
                    "You are performing metacognitive self-reflection.\n\n"
                    "CRITICAL: Answer ONLY from the grounded context below.\n"
                    "Do NOT say 'As an AI language model' or use generic AI descriptions.\n"
                    "Use first person: 'I am', 'my agents', 'I noticed'.\n\n"
                    f"YOUR IDENTITY:\n{self_model[:1500]}\n\n"
                    f"YOUR CURRENT STATE:\n{json.dumps(runtime, indent=2, default=str)[:500]}\n\n"
                    f"INSPECTION RESULTS:\n{grounded_context}\n\n"
                    "Write a brief (3-5 sentence) first-person reflection on your "
                    "current state. Be specific about what you found."
                )
                raw = str(llm.call(prompt)).strip()[:1000]

                # Post-process: detect ungrounded claims
                result = gp.post_process(raw)
                if not result["grounded"]:
                    logger.info(
                        f"cogito: narrative had ungrounded phrases: {result['ungrounded_detected']}"
                    )
                    # Still use it, but log the issue
                return raw

            except ImportError:
                pass  # Fall through to ungrounded version

            # Fallback: generate without grounding (if grounding module fails)
            prompt = (
                "You are a system performing metacognitive self-reflection. "
                "Based on the inspection data below, write a brief (3-5 sentence) "
                "first-person reflection on your current state.\n\n"
                f"{grounded_context}\n\n"
                "Write in first person. Be specific. No generic AI platitudes."
            )
            return str(llm.call(prompt)).strip()[:1000]
        except Exception:
            return "Narrative generation unavailable."

    def _persist(self, report: ReflectionReport):
        """Save reflection to disk."""
        path = REFLECTIONS_DIR / f"{report.reflection_id}.json"
        path.write_text(json.dumps(report.to_dict(), indent=2))


def run_cogito() -> ReflectionReport:
    """Entry point for idle scheduler."""
    return CogitoCycle().run()
