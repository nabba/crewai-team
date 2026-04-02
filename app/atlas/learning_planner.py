"""
learning_planner.py — Auto-generates learning plans for capability gaps.

When the system receives a task that exceeds its current competence,
the Learning Planner:
  1. Identifies what's missing (via competence tracker)
  2. Prioritizes learning tasks by importance
  3. Estimates time for each learning task
  4. Generates an ordered plan using available subsystems

Also includes Quality Evaluator logic for knowledge quality scoring
with decay, usage tracking, and freshness monitoring.

IMMUTABLE — infrastructure-level module.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ── Learning step types ──────────────────────────────────────────────────────

LEARNING_METHODS = {
    "api_scout": {
        "name": "API Scout Discovery",
        "description": "Discover and learn an API via documentation",
        "estimated_minutes": 15,
        "domains": ["apis"],
    },
    "video_learn": {
        "name": "Video Learning",
        "description": "Extract knowledge from YouTube tutorials",
        "estimated_minutes": 10,
        "domains": ["concepts", "patterns"],
    },
    "code_forge_build": {
        "name": "Code Forge Build",
        "description": "Build and test a code solution",
        "estimated_minutes": 15,
        "domains": ["patterns", "recipes"],
    },
    "web_research": {
        "name": "Web Research",
        "description": "Search and synthesize from web sources",
        "estimated_minutes": 5,
        "domains": ["concepts", "apis"],
    },
    "trial_and_error": {
        "name": "Experimentation",
        "description": "Try approaches and learn from results",
        "estimated_minutes": 20,
        "domains": ["apis", "patterns"],
    },
}


@dataclass
class LearningStep:
    """A single step in a learning plan."""
    step_id: str = ""
    method: str = ""              # key from LEARNING_METHODS
    target: str = ""              # what to learn (API name, concept, etc.)
    domain: str = ""              # apis | concepts | patterns
    priority: int = 0             # 1 = highest
    estimated_minutes: int = 0
    rationale: str = ""           # why this step is needed
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"       # pending | in_progress | completed | failed
    result: str = ""


@dataclass
class LearningPlan:
    """An ordered plan for learning missing capabilities."""
    task_description: str = ""
    steps: list[LearningStep] = field(default_factory=list)
    total_estimated_minutes: int = 0
    readiness_before: dict = field(default_factory=dict)
    created_at: str = ""
    status: str = "pending"       # pending | executing | completed | failed

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class LearningPlanner:
    """Generates and manages learning plans for capability gaps."""

    def __init__(self):
        pass

    def create_plan(self, task_description: str, requirements: list[dict] | None = None) -> LearningPlan:
        """Create a learning plan for a task.

        Args:
            task_description: What the task needs to accomplish
            requirements: Optional explicit requirements [{domain, name}]
                         If not provided, will infer from task description.

        Returns:
            LearningPlan with ordered steps
        """
        plan = LearningPlan(
            task_description=task_description,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Step 1: Determine requirements
        if not requirements:
            requirements = self._infer_requirements(task_description)

        # Step 2: Check competence
        from app.atlas.competence_tracker import get_tracker
        tracker = get_tracker()
        readiness = tracker.check_task_readiness(requirements)
        plan.readiness_before = readiness

        if readiness["ready"]:
            plan.status = "completed"
            return plan

        # Step 3: Generate learning steps for gaps
        step_num = 0
        for gap in readiness["unknown"]:
            step_num += 1
            step = self._create_step_for_gap(gap, step_num)
            plan.steps.append(step)

        for stale in readiness["stale"]:
            step_num += 1
            step = self._create_refresh_step(stale, step_num)
            plan.steps.append(step)

        # Step 4: Order by priority and dependencies
        plan.steps.sort(key=lambda s: s.priority)

        # Step 5: Calculate total time
        plan.total_estimated_minutes = sum(s.estimated_minutes for s in plan.steps)

        logger.info(f"learning_planner: created plan with {len(plan.steps)} steps, "
                    f"est. {plan.total_estimated_minutes} min for: {task_description[:80]}")

        return plan

    def execute_plan(self, plan: LearningPlan) -> LearningPlan:
        """Execute a learning plan step by step.

        Returns: Updated plan with results.
        """
        plan.status = "executing"

        for step in plan.steps:
            # Check dependencies
            deps_met = all(
                self._find_step(plan, dep_id).status == "completed"
                for dep_id in step.depends_on
                if self._find_step(plan, dep_id)
            )
            if not deps_met:
                continue

            step.status = "in_progress"
            try:
                result = self._execute_step(step)
                step.status = "completed" if result.get("success") else "failed"
                step.result = json.dumps(result)[:500]
            except Exception as e:
                step.status = "failed"
                step.result = str(e)[:500]

        # Check if all steps completed
        completed = all(s.status == "completed" for s in plan.steps)
        plan.status = "completed" if completed else "failed"

        return plan

    # ── Step creation ─────────────────────────────────────────────────────

    def _create_step_for_gap(self, gap: dict, step_num: int) -> LearningStep:
        """Create a learning step for a missing capability."""
        domain = gap.get("domain", "")
        name = gap.get("name", "")

        # Select best method based on domain
        if domain == "apis":
            method = "api_scout"
            rationale = f"Need to discover and learn the {name} API"
        elif domain == "concepts":
            method = "video_learn"
            rationale = f"Need to understand {name} — video tutorials are most effective"
        elif domain == "patterns":
            method = "web_research"
            rationale = f"Need to learn the {name} pattern"
        else:
            method = "trial_and_error"
            rationale = f"Need to learn about {name} through experimentation"

        method_info = LEARNING_METHODS.get(method, {})

        return LearningStep(
            step_id=f"step_{step_num}",
            method=method,
            target=name,
            domain=domain,
            priority=step_num,
            estimated_minutes=method_info.get("estimated_minutes", 15),
            rationale=rationale,
        )

    def _create_refresh_step(self, stale: dict, step_num: int) -> LearningStep:
        """Create a refresh step for stale knowledge."""
        name = stale.get("name", "")
        return LearningStep(
            step_id=f"refresh_{step_num}",
            method="web_research",
            target=name,
            domain="apis",
            priority=step_num + 100,  # Lower priority than new learning
            estimated_minutes=5,
            rationale=f"Knowledge of {name} is stale (confidence={stale.get('confidence', 0):.2f})",
        )

    # ── Step execution ────────────────────────────────────────────────────

    def _execute_step(self, step: LearningStep) -> dict:
        """Execute a single learning step."""
        if step.method == "api_scout":
            return self._exec_api_scout(step.target)
        elif step.method == "video_learn":
            return self._exec_video_learn(step.target)
        elif step.method == "code_forge_build":
            return self._exec_code_forge(step.target)
        elif step.method == "web_research":
            return self._exec_web_research(step.target)
        elif step.method == "trial_and_error":
            return self._exec_trial(step.target)
        return {"success": False, "error": f"Unknown method: {step.method}"}

    def _exec_api_scout(self, target: str) -> dict:
        """Execute API Scout discovery."""
        try:
            from app.atlas.api_scout import get_scout
            scout = get_scout()
            result = scout.build_and_register(target)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)[:200]}

    def _exec_video_learn(self, target: str) -> dict:
        """Execute video learning search."""
        try:
            from app.atlas.video_learner import get_learner
            learner = get_learner()
            results = learner.learn_from_search(f"{target} tutorial", max_videos=2)
            return {
                "success": len(results) > 0,
                "videos_processed": len(results),
                "concepts": sum(len(r.concepts) for r in results),
                "recipes": sum(len(r.code_recipes) for r in results),
            }
        except Exception as e:
            return {"success": False, "error": str(e)[:200]}

    def _exec_code_forge(self, target: str) -> dict:
        """Execute Code Forge build."""
        try:
            from app.atlas.code_forge import get_forge
            forge = get_forge()
            result = forge.build_and_register(f"Build a {target}")
            return {"success": result.success, "skill_id": result.skill_id}
        except Exception as e:
            return {"success": False, "error": str(e)[:200]}

    def _exec_web_research(self, target: str) -> dict:
        """Execute web research."""
        try:
            from app.tools.brave_search import search_brave
            results = search_brave(f"{target} documentation tutorial", count=3)
            if results:
                from app.atlas.competence_tracker import get_tracker
                tracker = get_tracker()
                tracker.register(
                    domain="concepts",
                    name=target,
                    confidence=0.5,
                    source="web_research",
                )
                return {"success": True, "sources_found": len(results)}
            return {"success": False, "error": "No results found"}
        except Exception as e:
            return {"success": False, "error": str(e)[:200]}

    def _exec_trial(self, target: str) -> dict:
        """Execute trial-and-error learning."""
        # This would integrate with the sandbox for experimentation
        return {"success": False, "error": "Trial-and-error not yet implemented"}

    # ── Requirement inference ─────────────────────────────────────────────

    def _infer_requirements(self, task_description: str) -> list[dict]:
        """Use LLM to infer what capabilities a task needs."""
        prompt = f"""What technical capabilities are needed to complete this task?

Task: {task_description}

Return as JSON array:
[
  {{"domain": "apis", "name": "API or service name"}},
  {{"domain": "concepts", "name": "technical concept"}},
  {{"domain": "patterns", "name": "code pattern"}}
]

Only include specific, named capabilities. Return ONLY valid JSON."""

        try:
            from app.llm_factory import create_specialist_llm
            llm = create_specialist_llm(max_tokens=500, role="research")
            raw = str(llm.call(prompt)).strip()

            import re
            json_match = re.search(r'\[[\s\S]+\]', raw)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass

        return []

    def _find_step(self, plan: LearningPlan, step_id: str) -> Optional[LearningStep]:
        """Find a step by ID in a plan."""
        for step in plan.steps:
            if step.step_id == step_id:
                return step
        return None

    def format_plan(self, plan: LearningPlan) -> str:
        """Generate human-readable plan summary."""
        lines = [
            f"📋 Learning Plan for: {plan.task_description[:80]}",
            f"   Status: {plan.status}",
            f"   Estimated time: {plan.total_estimated_minutes} minutes",
            "",
        ]

        if plan.readiness_before.get("known"):
            lines.append("   Known capabilities:")
            for k in plan.readiness_before["known"]:
                lines.append(f"     ✅ {k['name']} (confidence={k['confidence']:.2f})")
            lines.append("")

        if plan.steps:
            lines.append("   Learning steps:")
            for step in plan.steps:
                status_icon = {
                    "pending": "⏳",
                    "in_progress": "🔄",
                    "completed": "✅",
                    "failed": "❌",
                }.get(step.status, "⏳")
                lines.append(
                    f"     {status_icon} [{step.method}] {step.target} "
                    f"(~{step.estimated_minutes} min) — {step.rationale}"
                )

        return "\n".join(lines)


# ── Quality Evaluator ────────────────────────────────────────────────────────


class QualityEvaluator:
    """Evaluates and tracks knowledge quality across the system.

    Monitors:
      - Discovery success rate (API Scout)
      - Video extraction quality (first-run success %)
      - Fix iteration count (Code Forge debug cycles)
      - Knowledge freshness (stale API integrations)
    """

    def __init__(self):
        self._metrics: dict[str, list[float]] = {
            "api_discovery_success_rate": [],
            "video_extraction_quality": [],
            "code_forge_first_run_success": [],
            "code_forge_avg_debug_iterations": [],
        }

    def record_api_discovery(self, success: bool) -> None:
        self._metrics["api_discovery_success_rate"].append(1.0 if success else 0.0)

    def record_video_extraction(self, recipes_working: int, recipes_total: int) -> None:
        if recipes_total > 0:
            self._metrics["video_extraction_quality"].append(recipes_working / recipes_total)

    def record_code_forge_result(self, success_first_try: bool, debug_iterations: int) -> None:
        self._metrics["code_forge_first_run_success"].append(1.0 if success_first_try else 0.0)
        self._metrics["code_forge_avg_debug_iterations"].append(float(debug_iterations))

    def get_quality_report(self) -> dict:
        """Return quality metrics summary."""
        report = {}
        for metric, values in self._metrics.items():
            if values:
                recent = values[-50:]  # Last 50 data points
                report[metric] = {
                    "current": sum(recent) / len(recent),
                    "samples": len(recent),
                    "trend": self._compute_trend(recent),
                }
            else:
                report[metric] = {"current": 0, "samples": 0, "trend": "no_data"}
        return report

    def _compute_trend(self, values: list[float]) -> str:
        """Detect if metric is improving, stable, or declining."""
        if len(values) < 10:
            return "insufficient_data"
        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        diff = second_half - first_half
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        return "stable"

    def format_report(self) -> str:
        """Human-readable quality report."""
        report = self.get_quality_report()
        lines = ["📊 Knowledge Quality Report", ""]
        for metric, data in report.items():
            trend_icon = {"improving": "📈", "declining": "📉", "stable": "➡️"}.get(
                data.get("trend", ""), "❓"
            )
            lines.append(
                f"  {metric}: {data['current']:.1%} "
                f"(n={data['samples']}) {trend_icon} {data.get('trend', '')}"
            )
        return "\n".join(lines)


# ── Module-level singletons ──────────────────────────────────────────────────

_planner: LearningPlanner | None = None
_evaluator: QualityEvaluator | None = None


def get_planner() -> LearningPlanner:
    global _planner
    if _planner is None:
        _planner = LearningPlanner()
    return _planner


def get_evaluator() -> QualityEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = QualityEvaluator()
    return _evaluator
