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

        return report

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
        """Generate a reflective narrative using LLM."""
        try:
            from app.llm_factory import create_specialist_llm
            llm = create_specialist_llm(max_tokens=500, role="self_improve")

            prompt = (
                "You are a system performing metacognitive self-reflection. "
                "Based on the inspection data below, write a brief (3-5 sentence) "
                "first-person reflection on your current state.\n\n"
                f"Health: {report.overall_health}\n"
                f"Discrepancies: {json.dumps(report.discrepancies[:5])}\n"
                f"Observations: {json.dumps(report.observations[:5])}\n"
                f"Agent count: {state.get('inspect_agents', {}).get('agent_count', '?')}\n"
                f"Uptime: {state.get('inspect_runtime', {}).get('uptime_seconds', '?')}s\n\n"
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
