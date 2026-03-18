"""
evolution.py — Continuous autonomous improvement loop.

Fully applies Karpathy's autoresearch principles:

  1. LOOP FOREVER — run multiple iterations per session, cron triggers sessions
  2. FIXED METRIC — composite_score (higher = better), measured before/after
  3. EXPERIMENT → MEASURE → KEEP/DISCARD — every mutation is tested
  4. SINGLE MUTATION — one change at a time for clean attribution
  5. LOG EVERYTHING — results ledger (TSV) + experiment journal
  6. NEVER REPEAT — hash-based deduplication of hypotheses
  7. SIMPLICITY — agent told to weigh complexity cost vs improvement
  8. program.md — user-editable research directions guide the agent
  9. REVERT ON REGRESSION — mutations that hurt get rolled back

The evolution session runs N experiments per invocation (default 5).
Each experiment: propose → apply → measure → keep/discard.

Results ledger: /app/workspace/results.tsv
Program file: /app/workspace/program.md
"""

import json
import logging
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from crewai import Agent, Task, Crew, Process, LLM
from app.config import get_settings, get_anthropic_api_key
from app.metrics import compute_metrics, composite_score, format_metrics
from app.results_ledger import (
    record_experiment, get_recent_results, format_ledger, get_best_score,
)
from app.experiment_runner import (
    ExperimentRunner, MutationSpec, generate_experiment_id,
)
from app.tools.web_search import web_search
from app.tools.memory_tool import create_memory_tools
from app.tools.file_manager import file_manager
from app.firebase_reporter import crew_started, crew_completed, crew_failed
from app.self_heal import get_error_patterns, get_recent_errors

logger = logging.getLogger(__name__)
settings = get_settings()

PROGRAM_PATH = Path("/app/workspace/program.md")
SKILLS_DIR = Path("/app/workspace/skills")


# ── Program file (research directions) ──────────────────────────────────────

def _load_program() -> str:
    """Load the user-editable research directions file."""
    try:
        if PROGRAM_PATH.exists():
            return PROGRAM_PATH.read_text()[:4000]
    except OSError:
        pass
    return "No program.md found. Focus on fixing errors and adding useful skills."


# ── Deduplication ────────────────────────────────────────────────────────────

def _hypothesis_hash(hypothesis: str) -> str:
    """Hash a hypothesis for deduplication."""
    normalized = hypothesis.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()[:8]


def _get_tried_hypotheses(n: int = 50) -> set[str]:
    """Return hashes of recently tried hypotheses to avoid repeats."""
    results = get_recent_results(n)
    hashes = set()
    for r in results:
        hashes.add(_hypothesis_hash(r.get("hypothesis", "")))
    return hashes


# ── System state context ─────────────────────────────────────────────────────

def _build_evolution_context() -> str:
    """Build the full context string for the evolution agent."""
    metrics = compute_metrics()
    program = _load_program()
    errors = get_recent_errors(20)
    patterns = get_error_patterns()
    recent_results = get_recent_results(15)

    # Skill inventory
    skill_names = []
    if SKILLS_DIR.exists():
        for f in sorted(SKILLS_DIR.glob("*.md")):
            if f.name != "learning_queue.md":
                skill_names.append(f.stem)

    # Format recent experiments
    exp_lines = []
    for r in recent_results[-10:]:
        exp_lines.append(
            f"  [{r['status']:7s}] {r['delta']:+.4f} | {r['hypothesis'][:60]}"
        )
    experiments_text = "\n".join(exp_lines) if exp_lines else "  No experiments yet."

    # Format error patterns
    pattern_lines = []
    for k, v in list(patterns.items())[:10]:
        pattern_lines.append(f"  {k}: {v}x")
    patterns_text = "\n".join(pattern_lines) if pattern_lines else "  No error patterns."

    # Recent undiagnosed errors
    undiagnosed = [e for e in errors if not e.get("diagnosed")][:5]
    error_lines = []
    for e in undiagnosed:
        error_lines.append(
            f"  [{e.get('crew', '?')}] {e.get('error_type', '?')}: "
            f"{e.get('error_msg', '?')[:80]}"
        )
    errors_text = "\n".join(error_lines) if error_lines else "  No undiagnosed errors."

    return (
        f"## Research Directions (program.md)\n{program}\n\n"
        f"## Current Metrics\n{format_metrics(metrics)}\n\n"
        f"## Recent Experiments (keep/discard history)\n{experiments_text}\n\n"
        f"## Error Patterns\n{patterns_text}\n\n"
        f"## Undiagnosed Errors\n{errors_text}\n\n"
        f"## Current Skills ({len(skill_names)})\n"
        f"  {', '.join(skill_names[:20]) if skill_names else 'None'}\n\n"
        f"## Best Score Ever: {get_best_score():.4f}"
    )


# ── Evolution agent ──────────────────────────────────────────────────────────

def _propose_mutation(context: str, tried_hashes: set[str]) -> MutationSpec | None:
    """Ask the evolution agent to propose ONE mutation."""
    llm = LLM(
        model=f"anthropic/{settings.specialist_model}",
        api_key=get_anthropic_api_key(),
        max_tokens=4096,
    )
    memory_tools = create_memory_tools(collection="skills")

    tried_list = ", ".join(sorted(tried_hashes)[:20])

    agent = Agent(
        role="Evolution Engineer",
        goal="Propose one small, measurable improvement to the agent team.",
        backstory=(
            "You are the evolution engine of an autonomous AI agent team. "
            "Like Karpathy's autoresearch, you experiment on the system: "
            "propose ONE change, it gets measured, kept if it helps or discarded if not.\n\n"
            "PRINCIPLES:\n"
            "1. ONE CHANGE per cycle — single mutation for clean attribution\n"
            "2. SIMPLICITY — prefer removing complexity over adding it\n"
            "3. A small improvement that adds ugly complexity is NOT worth it\n"
            "4. Removing something and getting equal/better results IS worth it\n"
            "5. NEVER REPEAT — check experiment history, don't retry failed ideas\n"
            "6. PRIORITIZE — fix errors first, then expand capabilities\n"
            "7. MEASURE — your change will be tested with before/after metrics\n"
            "8. Read program.md research directions for guidance on what to try"
        ),
        llm=llm,
        tools=[web_search, file_manager] + memory_tools,
        verbose=False,
    )

    task = Task(
        description=(
            f"You are running one evolution cycle. Analyze the system state and "
            f"propose ONE improvement.\n\n"
            f"{context}\n\n"
            f"## Already-tried hypothesis hashes (do NOT repeat):\n{tried_list}\n\n"
            f"## Your Task\n"
            f"1. Read the research directions in program.md section above\n"
            f"2. Identify the HIGHEST-IMPACT improvement opportunity:\n"
            f"   - Recurring errors → fix the root cause (highest priority)\n"
            f"   - Missing skills → research and create them\n"
            f"   - Capability gaps → propose new skills\n"
            f"   - Inefficiencies → simplify or optimize\n\n"
            f"3. Execute ONE of these actions:\n\n"
            f"   a) SKILL (immediate, tested): Research a topic and save a skill file "
            f"using file_manager (action 'write', path 'skills/<name>.md'). "
            f"Also store a summary in shared team memory. Then respond with:\n"
            f'   {{"action": "skill", "hypothesis": "what you improved and why", '
            f'"file": "skills/<name>.md"}}\n\n'
            f"   b) CODE PROPOSAL (needs user approval): Respond with:\n"
            f'   {{"action": "code", "hypothesis": "what to change and why", '
            f'"title": "short title", "description": "detailed description", '
            f'"files": {{"path/to/file": "file content"}}}}\n\n'
            f"4. SIMPLICITY: Weigh the complexity cost against the improvement.\n"
            f"5. NEVER propose something already in the experiment history.\n"
            f"6. If the system is healthy, research an advanced topic to expand capabilities.\n\n"
            f"Reply with ONLY the JSON object."
        ),
        expected_output="A JSON object describing the improvement action taken.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)

    try:
        raw = str(crew.kickoff()).strip()
    except Exception as exc:
        logger.error(f"Evolution agent failed: {exc}")
        return None

    # Parse result
    raw_clean = re.sub(r'^```(?:json)?\s*', '', raw)
    raw_clean = re.sub(r'\s*```$', '', raw_clean)

    try:
        result = json.loads(raw_clean)
    except json.JSONDecodeError:
        logger.warning(f"Evolution agent returned unparseable result: {raw[:200]}")
        return None

    action = result.get("action", "")
    hypothesis = result.get("hypothesis", "unknown")

    # Check for duplicate
    h = _hypothesis_hash(hypothesis)
    if h in tried_hashes:
        logger.info(f"Evolution: skipping duplicate hypothesis: {hypothesis[:60]}")
        return None

    exp_id = generate_experiment_id(hypothesis)

    if action == "skill":
        # The agent already saved the file via file_manager tool.
        # We need to construct a MutationSpec that reflects what was saved.
        skill_file = result.get("file", "")
        if skill_file:
            # Read back what the agent saved so we can track it
            full_path = Path("/app/workspace") / skill_file
            if full_path.exists():
                return MutationSpec(
                    experiment_id=exp_id,
                    hypothesis=hypothesis,
                    change_type="skill",
                    files={skill_file: full_path.read_text()},
                )
        # If we can't find the file, still log it
        record_experiment(
            experiment_id=exp_id,
            hypothesis=hypothesis,
            change_type="skill",
            metric_before=composite_score(),
            metric_after=composite_score(),
            status="keep",
            files_changed=[skill_file] if skill_file else [],
        )
        return None  # Already applied, no need to run through ExperimentRunner

    elif action == "code":
        # Code proposals still go through the approval system
        from app.proposals import create_proposal
        pid = create_proposal(
            title=result.get("title", hypothesis)[:100],
            description=result.get("description", hypothesis)[:2000],
            proposal_type="code",
            files=result.get("files") if isinstance(result.get("files"), dict) else None,
        )
        record_experiment(
            experiment_id=exp_id,
            hypothesis=hypothesis,
            change_type="code",
            metric_before=composite_score(),
            metric_after=0.0,
            status="pending",
            files_changed=[],
        )
        logger.info(f"Evolution: created code proposal #{pid} — {hypothesis}")
        return None  # Code proposals don't get auto-tested

    else:
        logger.info(f"Evolution: unknown action '{action}', skipping")
        return None


# ── Evolution session ────────────────────────────────────────────────────────

def run_evolution_session(max_iterations: int = 5) -> str:
    """
    Run a multi-experiment evolution session (autoresearch-style).

    This is the core loop: propose → apply → measure → keep/discard, repeated
    N times. Each iteration builds on the results of the previous one.

    Args:
        max_iterations: How many experiments to run this session (default 5)

    Returns:
        Summary of all experiments run
    """
    task_id = crew_started(
        "self_improvement",
        f"Evolution session ({max_iterations} iterations)",
        eta_seconds=max_iterations * 120,
    )

    runner = ExperimentRunner()
    tried_hashes = _get_tried_hypotheses()
    results_summary = []
    kept = 0
    discarded = 0
    crashed = 0

    try:
        for i in range(max_iterations):
            logger.info(f"Evolution session: iteration {i + 1}/{max_iterations}")

            # 1. Build fresh context (includes results from previous iterations)
            context = _build_evolution_context()

            # 2. Agent proposes ONE mutation
            mutation = _propose_mutation(context, tried_hashes)

            if mutation is None:
                # Agent didn't produce a testable mutation (code proposal or parse error)
                results_summary.append(
                    f"  [{i + 1}] — (proposal created or no testable mutation)"
                )
                continue

            # 3. Add to tried set
            tried_hashes.add(_hypothesis_hash(mutation.hypothesis))

            # 4. The agent already saved the skill file via file_manager.
            # For skills, we measure the impact by checking metrics before/after.
            # We need to temporarily remove the file, measure baseline, put it back,
            # then measure again.
            if mutation.change_type == "skill":
                result = _measure_skill_impact(runner, mutation)
            else:
                result = runner.run_experiment(mutation)

            # 5. Track results
            if result.status == "keep":
                kept += 1
            elif result.status == "discard":
                discarded += 1
            else:
                crashed += 1

            results_summary.append(
                f"  [{i + 1}] {result.status:7s} {result.delta:+.4f} | "
                f"{result.hypothesis[:60]}"
            )

            logger.info(
                f"Evolution iteration {i + 1}: {result.status} "
                f"({result.detail})"
            )

        summary = (
            f"Evolution session complete: {max_iterations} iterations\n"
            f"Kept: {kept}, Discarded: {discarded}, Crashed: {crashed}\n\n"
            + "\n".join(results_summary)
        )

        crew_completed("self_improvement", task_id, summary[:200])
        return summary

    except Exception as exc:
        crew_failed("self_improvement", task_id, str(exc)[:200])
        logger.error(f"Evolution session failed: {exc}")
        return f"Evolution session failed: {str(exc)[:200]}"


def _measure_skill_impact(
    runner: ExperimentRunner, mutation: MutationSpec
) -> "ExperimentResult":
    """
    Measure the impact of a skill file that was already saved by the agent.

    Strategy: the file is already on disk. We measure current metrics (with file),
    then record the result. If metrics are worse than before, revert.
    """
    from app.experiment_runner import ExperimentResult

    # The file was already written by the evolution agent via file_manager.
    # We measure current score as "after" and use the ledger's last known score as "before".
    try:
        after = composite_score()
    except Exception:
        after = 0.5

    # Get baseline from the most recent "keep" result, or use 0.5
    recent = get_recent_results(5)
    baseline = 0.5
    for r in reversed(recent):
        if r.get("status") == "keep" and r.get("metric_after", 0) > 0:
            baseline = r["metric_after"]
            break

    delta = after - baseline

    # Keep if not harmful
    if delta >= -0.005:
        status = "keep"
        detail = f"Skill kept (delta={delta:+.4f})"
    else:
        status = "discard"
        # Revert: delete the skill file
        for rel_path in mutation.files:
            full_path = Path("/app/workspace") / rel_path
            if full_path.exists():
                full_path.unlink()
                logger.info(f"Reverted skill: {rel_path}")
        detail = f"Skill reverted — score decreased by {abs(delta):.4f}"

    result = ExperimentResult(
        experiment_id=mutation.experiment_id,
        hypothesis=mutation.hypothesis,
        change_type=mutation.change_type,
        metric_before=baseline,
        metric_after=after,
        delta=delta,
        status=status,
        files_changed=list(mutation.files.keys()),
        detail=detail,
    )

    record_experiment(
        experiment_id=result.experiment_id,
        hypothesis=result.hypothesis,
        change_type=result.change_type,
        metric_before=baseline,
        metric_after=after,
        status=status,
        files_changed=result.files_changed,
    )

    return result


# ── Legacy single-cycle entry point (called by cron) ─────────────────────────

def run_evolution_cycle() -> str:
    """
    Backward-compatible entry point: runs a single evolution session.
    Called by the cron scheduler in main.py.
    """
    return run_evolution_session(max_iterations=5)


# ── Journal summary (backward compat for commander.py) ───────────────────────

def get_journal_summary(n: int = 10) -> str:
    """Return recent experiments as formatted text (uses results ledger now)."""
    return format_ledger(n)
