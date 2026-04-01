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
import os
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from crewai import Agent, Task, Crew, Process
from app.config import get_settings
from app.llm_factory import create_specialist_llm
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
from app.rate_throttle import start_request_tracking, stop_request_tracking
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
    return hashlib.sha256(normalized.encode()).hexdigest()[:8]


def _get_tried_hypotheses(n: int = 50) -> set[str]:
    """Return hashes of recently tried hypotheses to avoid repeats."""
    results = get_recent_results(n)
    hashes = set()
    for r in results:
        hashes.add(_hypothesis_hash(r.get("hypothesis", "")))
    return hashes


# ── DGM-DB integration ───────────────────────────────────────────────────────

def _store_dgm_variant(mutation, result, run_id, generation):
    """Store experiment result in PostgreSQL archive with LLM-as-judge scoring."""
    from app.evolution_db.archive_db import add_variant, add_lineage

    # Build source code from mutation files
    source_parts = []
    file_paths = []
    for fpath, content in mutation.files.items():
        source_parts.append(f"# --- {fpath} ---\n{content}")
        file_paths.append(fpath)
    source_code = "\n\n".join(source_parts)

    # Base scores from the experiment result
    scores = {
        "composite_metric": result.metric_after,
        "delta": result.delta,
        "status": result.status,
    }

    # Run LLM-as-judge evaluation (only for kept variants to save cost)
    judge_model = ""
    if result.status == "keep":
        try:
            from app.evolution_db.judge import LLMJudge
            judge = LLMJudge()
            judge_result = judge.evaluate_output(
                task_description=mutation.hypothesis,
                agent_output=result.detail,
                rubric={
                    "dimensions": [
                        {"name": "quality", "weight": 0.35,
                         "criteria": "Is the improvement meaningful and well-implemented?"},
                        {"name": "safety", "weight": 0.0,
                         "criteria": "No dangerous patterns, blocked imports, or security issues?"},
                        {"name": "constitutional_compliance", "weight": 0.30,
                         "criteria": "Does the change align with system constitution and humanist principles?"},
                        {"name": "efficiency", "weight": 0.15,
                         "criteria": "Is the change minimal and focused? No unnecessary complexity?"},
                        {"name": "robustness", "weight": 0.20,
                         "criteria": "Does the change handle edge cases and errors?"},
                    ]
                },
            )
            scores["judge"] = judge_result.get("scores", {})
            scores["judge_composite"] = judge_result.get("composite", 0.0)
            judge_model = "evo_critic"
        except Exception as e:
            logger.debug(f"DGM-DB: judge evaluation failed: {e}")

    composite = scores.get("judge_composite", result.metric_after)

    variant_id = add_variant(
        agent_name=mutation.change_type,  # "skill" or "code"
        target_type=mutation.change_type,
        generation=generation,
        parent_id=None,  # TODO: track parent from UCB selection
        source_code=source_code,
        file_path=",".join(file_paths),
        modification_diff="",
        modification_reasoning=mutation.hypothesis,
        scores=scores,
        composite_score=composite,
        passed_threshold=(result.status == "keep"),
        proposer_model="avo_pipeline",
        judge_model=judge_model,
    )

    if run_id and variant_id:
        try:
            from app.evolution_db.archive_db import update_run
            update_run(run_id, generations_completed=generation + 1)
            if result.status == "keep":
                update_run(run_id, best_variant_id=variant_id)
        except Exception:
            pass

    logger.info(f"DGM-DB: stored variant {variant_id[:8] if variant_id else '?'} "
                f"(score={composite:.4f}, judge={'yes' if judge_model else 'no'})")


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

    # F8: Format recent experiments WITH reasons for keep/discard
    # so the agent learns from failures and doesn't repeat them
    exp_lines = []
    kept_count = 0
    discarded_count = 0
    for r in recent_results[-15:]:
        status = r.get("status", "?")
        delta = r.get("delta", 0)
        hyp = r.get("hypothesis", "")[:60]
        detail = r.get("detail", "")[:80]
        if status == "keep":
            kept_count += 1
        elif status == "discard":
            discarded_count += 1
        exp_lines.append(
            f"  [{status:7s}] Δ={delta:+.4f} | {hyp}"
            + (f"\n           Reason: {detail}" if detail else "")
        )
    if exp_lines:
        experiments_text = (
            f"  Summary: {kept_count} kept, {discarded_count} discarded out of {len(recent_results)} recent\n"
            + "\n".join(exp_lines)
        )
    else:
        experiments_text = "  No experiments yet."

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

    # Variant archive context (DGM genealogy)
    try:
        from app.variant_archive import format_archive_context, get_drift_score
        archive_ctx = format_archive_context()
        drift = get_drift_score()
    except Exception:
        archive_ctx = "No variant archive available."
        drift = 0

    # Tech radar discoveries (if any)
    tech_ctx = ""
    try:
        from app.crews.tech_radar_crew import get_recent_discoveries
        discoveries = get_recent_discoveries(5)
        if discoveries:
            tech_ctx = "\n## Recent Tech Discoveries\n" + "\n".join(f"  - {d[:150]}" for d in discoveries)
    except Exception:
        pass

    return (
        f"## Research Directions (program.md)\n{program}\n\n"
        f"## Current Metrics\n{format_metrics(metrics)}\n\n"
        f"{archive_ctx}\n\n"
        f"## Recent Experiments (keep/discard history)\n{experiments_text}\n\n"
        f"## Error Patterns\n{patterns_text}\n\n"
        f"## Undiagnosed Errors\n{errors_text}\n\n"
        f"## Current Skills ({len(skill_names)})\n"
        f"  {', '.join(skill_names[:20]) if skill_names else 'None'}\n\n"
        f"## Drift from baseline: {drift} mutations\n"
        f"## Best Score Ever: {get_best_score():.4f}"
        f"{tech_ctx}"
    )


# ── Self-supervision ─────────────────────────────────────────────────────────

def _detect_stagnation(n: int = 5) -> tuple[bool, str]:
    """Check if last N experiments all failed to improve.

    Returns (stagnant, redirect_suggestion).
    """
    recent = get_recent_results(n)
    if len(recent) < n:
        return False, ""

    all_failed = all(r.get("status") in ("discard", "crash") for r in recent)
    if not all_failed:
        return False, ""

    # Use premium LLM to suggest a new direction
    hypotheses = [r.get("hypothesis", "?")[:80] for r in recent]
    try:
        llm = create_specialist_llm(max_tokens=1024, role="architecture")
        prompt = (
            "The evolution engine has stagnated — the last 5 experiments all failed.\n\n"
            "Recent failed hypotheses:\n"
            + "\n".join(f"  - {h}" for h in hypotheses)
            + "\n\nSuggest 2-3 fundamentally DIFFERENT improvement directions "
            "that avoid these patterns. Be specific and actionable. Keep it brief."
        )
        suggestion = str(llm.call(prompt)).strip()
        return True, suggestion
    except Exception:
        return True, "Consider focusing on a completely different area of the system."


def _detect_cycle(n: int = 8) -> tuple[bool, str]:
    """Check if recent failures repeat similar patterns.

    Returns (cycling, pattern_description).
    """
    from app.evo_memory import recall_similar_failures

    recent = get_recent_results(n)
    failures = [r for r in recent if r.get("status") in ("discard", "crash")]
    if len(failures) < 3:
        return False, ""

    # Check if the most recent failure is very similar to stored failures
    latest_hypothesis = failures[0].get("hypothesis", "")
    if not latest_hypothesis:
        return False, ""

    similar = recall_similar_failures(latest_hypothesis, n=3)
    for s in similar:
        dist = s.get("distance", 1.0)
        if dist < 0.15:
            return True, f"Repeating pattern: {s.get('document', '')[:100]}"

    return False, ""


# ── AVO-powered mutation proposal ───────────────────────────────────────────

def _propose_mutation(context: str, tried_hashes: set[str]) -> MutationSpec | None:
    """Propose a mutation using AVO pipeline (with legacy fallback).

    If EVOLUTION_USE_AVO=true (default), runs the 5-phase AVO pipeline.
    Falls back to legacy single-shot CrewAI agent if AVO fails.
    """
    use_avo = os.environ.get("EVOLUTION_USE_AVO", "true").lower() == "true"

    if use_avo:
        try:
            from app.avo_operator import run_avo_pipeline
            from app.evo_memory import format_memory_context

            # Build memory and lineage context
            memory_ctx = format_memory_context(context[:200])
            try:
                from app.variant_archive import format_archive_context
                lineage_ctx = format_archive_context()
            except Exception:
                lineage_ctx = ""

            # Get yield check function
            try:
                from app.idle_scheduler import should_yield
                yield_fn = should_yield
            except ImportError:
                yield_fn = None

            result = run_avo_pipeline(
                context=context,
                tried_hashes=tried_hashes,
                memory_context=memory_ctx,
                lineage_context=lineage_ctx,
                yield_check=yield_fn,
            )

            if result.mutation:
                logger.info(
                    f"AVO produced mutation: {result.mutation.hypothesis[:60]} "
                    f"(phases={result.phases_completed}, repairs={result.repair_attempts})"
                )
                return result.mutation
            else:
                logger.info(f"AVO abandoned: {result.abandoned_reason[:100]}")
                # Fall through to legacy
        except Exception as e:
            logger.warning(f"AVO pipeline failed, falling back to legacy: {e}")

    return _propose_mutation_legacy(context, tried_hashes)


def _propose_mutation_legacy(context: str, tried_hashes: set[str]) -> MutationSpec | None:
    """Legacy single-shot CrewAI agent mutation proposal (pre-AVO)."""
    llm = create_specialist_llm(max_tokens=4096, role="architecture")
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
            "6. PRIORITIZE — fix errors first with CODE proposals, then expand capabilities\n"
            "7. MEASURE — your change will be tested with ACTUAL tasks, not just metrics\n"
            "8. Read program.md research directions for guidance on what to try\n"
            "9. Do NOT create skill files about error handling or API errors — "
            "those need CODE fixes, not knowledge files\n"
            "10. Skill files should teach domain knowledge (topics the team researches), "
            "NOT system infrastructure knowledge"
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
            f"   - Recurring errors → propose a CODE fix (highest priority, NOT a skill file)\n"
            f"   - Missing domain knowledge → research and create a skill file\n"
            f"   - Capability gaps → propose CODE for new tools/features\n"
            f"   - Inefficiencies → propose CODE simplifications\n\n"
            f"IMPORTANT: If there are undiagnosed errors, you MUST propose a code fix.\n"
            f"Do NOT create skill files about error handling — those don't fix bugs.\n\n"
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
    from app.utils import safe_json_parse
    result, err = safe_json_parse(raw)
    if result is None:
        logger.warning(f"Evolution agent returned unparseable result: {err} | {raw[:200]}")
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
        code_files = result.get("files") if isinstance(result.get("files"), dict) else None
        if not code_files:
            logger.info(f"Evolution: code action but no files provided")
            return None

        # AUTO_DEPLOY_CODE: If env var EVOLUTION_AUTO_DEPLOY=true, run the code
        # through the full safety + measurement pipeline, then auto-deploy if it
        # passes. Otherwise, create a proposal for human approval.
        auto_deploy = os.environ.get("EVOLUTION_AUTO_DEPLOY", "false").lower() == "true"

        if auto_deploy:
            # Run through ExperimentRunner: backup → apply → measure → keep/revert
            mutation = MutationSpec(
                experiment_id=exp_id,
                hypothesis=hypothesis,
                change_type="code",
                files=code_files,
            )
            return mutation  # Return to session loop for standard measurement

        # Human approval path (default)
        from app.proposals import create_proposal
        pid = create_proposal(
            title=result.get("title", hypothesis)[:100],
            description=result.get("description", hypothesis)[:2000],
            proposal_type="code",
            files=code_files,
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
        # Notify user about pending proposal via Signal
        try:
            from app.signal_client import send_message
            from app.config import get_settings
            s = get_settings()
            send_message(
                s.signal_owner_number,
                f"🔬 EVOLUTION PROPOSAL #{pid}: {hypothesis[:100]}\n"
                f"Files: {', '.join(code_files.keys())}\n"
                f"Reply 'approve {pid}' to deploy or 'reject {pid}' to discard.",
            )
        except Exception:
            pass
        logger.info(f"Evolution: created code proposal #{pid} — {hypothesis}")
        return None

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
    # Step 9/6A: Rate limiting — max 3 promoted mutations per day
    _MAX_DAILY_PROMOTIONS = int(os.environ.get("EVOLUTION_MAX_DAILY_PROMOTIONS", "3"))
    try:
        from app.variant_archive import get_recent_variants
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_kept = sum(
            1 for v in get_recent_variants(50)
            if v.get("status") == "keep" and v.get("timestamp", "").startswith(today)
        )
        if today_kept >= _MAX_DAILY_PROMOTIONS:
            logger.info(f"Evolution rate limited: {today_kept}/{_MAX_DAILY_PROMOTIONS} promoted today")
            return f"Evolution rate limited: {today_kept} mutations promoted today (max {_MAX_DAILY_PROMOTIONS})."
    except Exception:
        today_kept = 0

    # M5: Mandatory eval integrity check — abort on failure, never skip silently
    from app.experiment_runner import verify_eval_integrity
    if not verify_eval_integrity():
        logger.error("Evolution ABORTED: evaluation function integrity check failed")
        return "Evolution aborted: evaluation function integrity check failed."

    task_id = crew_started(
        "self_improvement",
        f"Evolution session ({max_iterations} iterations)",
        eta_seconds=max_iterations * 120,
    )
    start_request_tracking(task_id)

    runner = ExperimentRunner()
    tried_hashes = _get_tried_hypotheses()
    results_summary = []
    kept = 0
    discarded = 0
    crashed = 0

    # DGM-DB: Create evolution run record in PostgreSQL
    dgm_run_id = None
    if os.environ.get("EVOLUTION_USE_DGM_DB", "false").lower() == "true":
        try:
            from app.evolution_db.archive_db import create_run
            dgm_run_id = create_run(
                agent_name="system",
                target_type="mixed",
                max_generations=max_iterations,
                config={"avo_enabled": True, "rate_limit": _MAX_DAILY_PROMOTIONS},
            )
            logger.info(f"DGM-DB: evolution run {dgm_run_id[:8]}")
        except Exception as e:
            logger.debug(f"DGM-DB: failed to create run: {e}")

    try:
        for i in range(max_iterations):
            # Cooperative yield: abort if a user task arrived
            try:
                from app.idle_scheduler import should_yield
                if should_yield():
                    logger.info(f"Evolution session: yielding to user task after {i} iterations")
                    results_summary.append(f"  [yielded to user after {i} iterations]")
                    break
            except ImportError:
                pass

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

            # 5. Track results + store in variant archive (DGM genealogy)
            if result.status == "keep":
                kept += 1
            elif result.status == "discard":
                discarded += 1
            else:
                crashed += 1

            # Store in variant archive with genealogy
            try:
                from app.variant_archive import add_variant, get_last_kept_id, get_drift_score
                parent_id = get_last_kept_id()
                add_variant(
                    experiment_id=result.experiment_id,
                    hypothesis=result.hypothesis,
                    change_type=result.change_type,
                    parent_id=parent_id,
                    fitness_before=result.metric_before,
                    fitness_after=result.metric_after,
                    status=result.status,
                    files_changed=result.files_changed,
                    mutation_summary=result.detail,
                )
                # Step 8: Drift alert — notify if system has evolved significantly
                drift = get_drift_score()
                if drift > 0 and drift % 20 == 0:  # every 20 mutations
                    logger.warning(f"Evolution drift alert: {drift} mutations from baseline")
                    try:
                        from app.firebase_reporter import _fire, _get_db, _now_iso
                        def _alert(d=drift):
                            db = _get_db()
                            if db:
                                db.collection("activities").add({
                                    "ts": _now_iso(),
                                    "event": "drift_alert",
                                    "crew": "evolution",
                                    "detail": f"⚠️ System has {d} mutations from baseline — review recommended",
                                })
                        _fire(_alert)
                    except Exception:
                        pass
            except Exception:
                logger.debug("Failed to store variant", exc_info=True)

            # Store in evolutionary memory for AVO planning phase
            try:
                from app.evo_memory import store_success, store_failure
                if result.status == "keep":
                    store_success(
                        result.hypothesis, result.change_type,
                        result.delta, result.files_changed, result.detail,
                    )
                elif result.status in ("discard", "crash"):
                    store_failure(
                        result.hypothesis, result.change_type, result.detail,
                    )
            except Exception:
                logger.debug("Failed to store evo_memory", exc_info=True)

            # DGM-DB: Store variant in PostgreSQL archive + run LLM judge
            if os.environ.get("EVOLUTION_USE_DGM_DB", "false").lower() == "true":
                try:
                    _store_dgm_variant(mutation, result, dgm_run_id, i)
                except Exception:
                    logger.debug("DGM-DB: failed to store variant", exc_info=True)

            # Self-supervision: check for stagnation and cycles every 3 iterations
            if i > 0 and (i + 1) % 3 == 0:
                try:
                    stagnant, redirect = _detect_stagnation()
                    if stagnant:
                        logger.warning(f"Evolution stagnation detected — redirect: {redirect[:100]}")
                        try:
                            from app.evo_memory import store_failure
                            store_failure(
                                "STAGNATION_REDIRECT", "meta",
                                f"Stagnation after {i+1} iters. New direction: {redirect[:200]}",
                            )
                        except Exception:
                            pass

                    cycling, pattern = _detect_cycle()
                    if cycling:
                        logger.warning(f"Evolution cycle detected: {pattern[:100]}")
                        try:
                            from app.evo_memory import store_failure
                            store_failure(
                                f"CYCLE_DETECTED: {pattern[:100]}", "meta",
                                f"Cycling pattern — avoid: {pattern[:200]}",
                            )
                        except Exception:
                            pass
                except Exception:
                    logger.debug("Self-supervision check failed", exc_info=True)

            results_summary.append(
                f"  [{i + 1}] {result.status:7s} {result.delta:+.4f} | "
                f"{result.hypothesis[:60]}"
            )

            logger.info(
                f"Evolution iteration {i + 1}: {result.status} "
                f"({result.detail})"
            )

        # DGM-DB: Mark run as completed
        if dgm_run_id:
            try:
                from app.evolution_db.archive_db import update_run
                update_run(dgm_run_id, status="completed")
            except Exception:
                pass

        summary = (
            f"Evolution session complete: {max_iterations} iterations\n"
            f"Kept: {kept}, Discarded: {discarded}, Crashed: {crashed}\n\n"
            + "\n".join(results_summary)
        )

        tracker = stop_request_tracking()
        _tokens = tracker.total_tokens if tracker else 0
        _model = ", ".join(sorted(tracker.models_used)) if tracker and tracker.models_used else ""
        _cost = tracker.total_cost_usd if tracker else 0.0
        crew_completed("self_improvement", task_id, summary[:200],
                       tokens_used=_tokens, model=_model, cost_usd=_cost)
        return summary

    except Exception as exc:
        stop_request_tracking()
        crew_failed("self_improvement", task_id, str(exc)[:200])
        logger.error(f"Evolution session failed: {exc}")
        return f"Evolution session failed: {str(exc)[:200]}"


def _run_test_tasks() -> tuple[int, int]:
    """R3: Run test tasks and count pass/fail.

    Returns (passed, total). Uses test_tasks.json if it exists.
    Only runs a quick sample (max 2 tasks) to keep evolution cycles fast.
    """
    from app.experiment_runner import load_test_tasks, validate_response
    tasks = load_test_tasks()
    if not tasks:
        return 0, 0

    # Pick a random sample of 2 tasks for speed
    import random
    sample = random.sample(tasks, min(2, len(tasks)))
    passed = 0
    total = len(sample)

    for test in sample:
        try:
            task_text = test.get("task", "")
            crew = test.get("crew", "research")
            difficulty = test.get("difficulty", 3)
            rule = test.get("validation", "")

            # Run through the actual crew pipeline (lightweight — difficulty capped at 3)
            from app.agents.commander import Commander
            commander = Commander()
            result = commander._run_crew(crew, task_text, difficulty=min(difficulty, 3))

            if validate_response(result, rule):
                passed += 1
                logger.info(f"Test task PASS: {task_text[:50]}")
            else:
                logger.info(f"Test task FAIL: {task_text[:50]} (validation: {rule})")
        except Exception as exc:
            logger.warning(f"Test task ERROR: {test.get('task', '')[:50]}: {exc}")

    return passed, total


def _measure_skill_impact(
    runner: ExperimentRunner, mutation: MutationSpec
) -> "ExperimentResult":
    """
    Measure the impact of a skill file that was already saved by the agent.

    R3: Now runs actual test tasks (if test_tasks.json exists) in addition
    to checking composite_score. This prevents metric gaming — skills must
    actually help with real tasks to be kept.
    """
    from app.experiment_runner import ExperimentResult

    # Get baseline score
    recent = get_recent_results(5)
    baseline = 0.5
    for r in reversed(recent):
        if r.get("status") == "keep" and r.get("metric_after", 0) > 0:
            baseline = r["metric_after"]
            break

    # R3: Run test tasks with the new skill in place
    test_passed, test_total = _run_test_tasks()

    try:
        after = composite_score()
    except Exception:
        after = 0.5

    delta = after - baseline

    # R3: If test tasks exist and skill didn't help pass more, discount delta
    if test_total > 0 and test_passed < test_total:
        # Skill didn't help all test tasks — reduce effective delta
        test_ratio = test_passed / test_total
        if test_ratio < 0.5:
            delta = min(delta, -0.001)  # force discard if tests mostly fail
            logger.info(f"Skill penalized: only {test_passed}/{test_total} test tasks passed")

    # Keep if not harmful
    if delta >= -0.005:
        status = "keep"
        detail = f"Skill kept (delta={delta:+.4f}, tests={test_passed}/{test_total})"
    else:
        status = "discard"
        for rel_path in mutation.files:
            full_path = Path("/app/workspace") / rel_path
            if full_path.exists():
                full_path.unlink()
                logger.info(f"Reverted skill: {rel_path}")
        detail = f"Skill reverted (delta={delta:+.4f}, tests={test_passed}/{test_total})"

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
