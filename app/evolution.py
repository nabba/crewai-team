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

# crewai imported lazily in _propose_mutation_legacy() to avoid 2s startup cost
# from crewai import Agent, Task, Crew, Process
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
from app.healing.error_diagnosis import get_error_patterns, get_recent_errors

logger = logging.getLogger(__name__)
settings = get_settings()

PROGRAM_PATH = Path("/app/workspace/program.md")
SKILLS_DIR = Path("/app/workspace/skills")

# ── Tunables (Phase G4: lifted from in-code defaults to named module
# constants so operators can reason about them without reading the
# source. None of these change behaviour; they make the existing
# defaults discoverable. If you want to tune at runtime, override via
# environment variable in the wrappers above.
RECENT_HYPOTHESIS_HISTORY_N = 50  # how many recent results to scan for dedup
FUZZY_HASH_PREFIX_LEN = 40        # normalize length for near-duplicate detection


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


def _get_tried_hypotheses(n: int = RECENT_HYPOTHESIS_HISTORY_N) -> set[str]:
    """Return hashes of recently tried hypotheses to avoid repeats.

    Includes both exact hashes and fuzzy hashes (normalized, first 40 chars)
    to catch near-duplicate hypotheses like 'API credit error' variants.
    """
    import re as _re
    results = get_recent_results(n)
    hashes = set()
    for r in results:
        hyp = r.get("hypothesis", "")
        # Exact hash
        hashes.add(_hypothesis_hash(hyp))
        # Fuzzy hash — strip numbers/punctuation, first 40 chars
        norm = _re.sub(r'[^a-z ]+', '', hyp.lower())
        norm = ' '.join(norm.split())[:40]
        hashes.add(hashlib.sha256(norm.encode()).hexdigest()[:8])
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

    # Format error patterns — with cooldown for already-addressed errors
    # Count how many times each error type has been addressed by recent experiments
    _addressed_errors = {}
    for r in recent_results:
        hyp = (r.get("hypothesis", "") + " " + r.get("detail", "")).lower()
        for k in patterns:
            if k.lower()[:20] in hyp:
                _addressed_errors[k] = _addressed_errors.get(k, 0) + 1

    pattern_lines = []
    for k, v in list(patterns.items())[:10]:
        times_addressed = _addressed_errors.get(k, 0)
        if times_addressed >= 3:
            pattern_lines.append(f"  {k}: {v}x (ALREADY ADDRESSED {times_addressed}x — skip this)")
        else:
            pattern_lines.append(f"  {k}: {v}x")
    patterns_text = "\n".join(pattern_lines) if pattern_lines else "  No error patterns."

    # Recent undiagnosed errors — exclude types already addressed 3+ times
    undiagnosed = [e for e in errors if not e.get("diagnosed")]
    fresh_errors = []
    for e in undiagnosed:
        etype = e.get("error_type", "")
        if _addressed_errors.get(etype, 0) < 3:
            fresh_errors.append(e)
    fresh_errors = fresh_errors[:5]
    error_lines = []
    for e in fresh_errors:
        error_lines.append(
            f"  [{e.get('crew', '?')}] {e.get('error_type', '?')}: "
            f"{e.get('error_msg', '?')[:80]}"
        )
    errors_text = "\n".join(error_lines) if error_lines else "  No fresh undiagnosed errors (all known errors addressed)."

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

    # ── SUBIA bridge: surprise-driven evolution targeting ──────────────────
    subia_ctx = ""
    try:
        from app.subia.prediction.accuracy_tracker import get_tracker
        tracker = get_tracker()
        summary = tracker.all_domains_summary()
        weak_domains = []
        for domain, stats in summary.items():
            if tracker.has_sustained_error(domain):
                weak_domains.append(
                    f"  - {domain}: accuracy={stats.get('mean_accuracy', 0):.2f}, "
                    f"sustained errors={stats.get('recent_bad_count', 0)}"
                )
        if weak_domains:
            subia_ctx = (
                "\n## SUBIA Prediction Failures (HIGH PRIORITY)\n"
                "These domains have sustained prediction errors — improving them\n"
                "would reduce future mistakes and increase system reliability.\n"
                + "\n".join(weak_domains[:5])
            )
    except Exception:
        pass

    # ── SUBIA bridge: evolution snapshot archive context ─────────────────
    snapshot_ctx = ""
    try:
        from app.workspace_versioning import list_evolution_tags
        tags = list_evolution_tags(5)
        if tags:
            snapshot_ctx = (
                "\n## Historical Variants (parent selection)\n"
                "You can propose changes starting from the current state or\n"
                "reference a historical high-scoring variant as a starting point.\n"
                + "\n".join(f"  - {t['tag']} ({t.get('date', '?')})" for t in tags)
            )
    except Exception:
        pass

    # ── Knowledge-informed evolution (Phase 3B) ─────────────────────────────
    # Augment context with research theory, past experiences, and growth edges.
    kb_evolution_ctx = ""
    try:
        # Episteme: theoretical backing for improvement directions.
        from app.episteme.vectorstore import get_store as get_episteme
        epi_store = get_episteme()
        if epi_store._collection.count() > 0:
            epi_hits = epi_store.query(
                query_text=f"improve multi-agent system {errors_text[:100]}",
                n_results=2,
            )
            if epi_hits:
                epi_texts = [h["text"][:300] for h in epi_hits]
                kb_evolution_ctx += (
                    "\n## Research Insights (episteme KB)\n"
                    + "\n".join(f"  - {t}" for t in epi_texts) + "\n"
                )
    except Exception:
        pass

    try:
        # Experiential: what happened last time we tried similar things.
        from app.experiential.vectorstore import get_store as get_exp
        exp_store = get_exp()
        if exp_store._collection.count() > 0:
            exp_hits = exp_store.query(
                query_text="evolution improvement experiment outcome",
                n_results=2,
            )
            if exp_hits:
                exp_texts = [h["text"][:300] for h in exp_hits]
                kb_evolution_ctx += (
                    "\n## Past Experiences (journal)\n"
                    + "\n".join(f"  - {t}" for t in exp_texts) + "\n"
                )
    except Exception:
        pass

    try:
        # Tensions: unresolved growth edges.
        from app.tensions.vectorstore import get_store as get_ten
        ten_store = get_ten()
        if ten_store._collection.count() > 0:
            ten_hits = ten_store.get_unresolved(n=3)
            if ten_hits:
                ten_texts = [h["text"][:200] for h in ten_hits]
                kb_evolution_ctx += (
                    "\n## Growth Edges (unresolved tensions)\n"
                    + "\n".join(f"  - {t}" for t in ten_texts) + "\n"
                )
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
        f"{subia_ctx}"
        f"{snapshot_ctx}"
        f"{kb_evolution_ctx}\n\n"
        f"## DIVERSITY REQUIREMENT\n"
        f"Do NOT propose improvements for errors marked 'ALREADY ADDRESSED'.\n"
        f"Explore NEW areas: performance optimization, code quality, new capabilities,\n"
        f"better error handling for DIFFERENT error types, architectural improvements,\n"
        f"test coverage, documentation, or tool enhancements.\n"
        f"Variety is more valuable than depth on a single topic."
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

    from crewai import Agent, Task, Crew, Process  # lazy import (~2s first call)

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
            # Governance gate: evolution_deploy requires approval
            try:
                from app.config import get_settings as _cgs
                if _cgs().control_plane_enabled:
                    from app.control_plane.governance import get_governance
                    from app.control_plane.projects import get_projects
                    gate = get_governance()
                    if gate.needs_approval("evolution_deploy"):
                        pid = get_projects().get_active_project_id()
                        gate.request_approval(
                            project_id=pid,
                            request_type="evolution_deploy",
                            requested_by="evolution",
                            title=f"Evolution code mutation: {hypothesis[:80]}",
                            detail={"hypothesis": hypothesis[:500],
                                    "files": [f["path"] for f in code_files[:5]]},
                        )
                        logger.info("Evolution: governance approval requested for code mutation")
                        # Fall through to proposal path instead of auto-deploy
                        auto_deploy = False
            except Exception:
                logger.debug("Evolution: governance check failed", exc_info=True)

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
        # create_proposal returns -1 when the proposal was REJECTED
        # (duplicate, path violation, or content-invalid via Q10 validator).
        # Don't send a Signal notification or record an experiment in that
        # case — the LLM hallucinated something unusable and we silently
        # drop it.  The rejection reason is already logged at warning level
        # inside create_proposal for debugging.
        if pid <= 0:
            logger.info(
                f"Evolution: proposal rejected at creation "
                f"(pid={pid}) for hypothesis: {hypothesis[:80]}"
            )
            return None

        record_experiment(
            experiment_id=exp_id,
            hypothesis=hypothesis,
            change_type="code",
            metric_before=composite_score(),
            metric_after=0.0,
            status="pending",
            files_changed=[],
        )
        # Notify user about pending proposal via Signal — attach the full
        # proposal.md so the user can read the rationale / changes / risks
        # breakdown on their phone before approving.  Use the blocking
        # variant so we capture the Signal timestamp for reaction-based
        # approval (👍 = approve, 👎 = reject).
        try:
            from app.signal_client import send_message_blocking
            from app.config import get_settings
            from app.proposals import get_proposal_md_path, set_proposal_signal_timestamp
            s = get_settings()
            md_path = get_proposal_md_path(pid)
            attachments = [str(md_path)] if md_path else None
            signal_ts = send_message_blocking(
                s.signal_owner_number,
                f"🔬 EVOLUTION PROPOSAL #{pid}: {hypothesis[:100]}\n"
                f"Files: {', '.join(code_files.keys())}\n"
                f"Full writeup attached (rationale, changes, risks).\n"
                f"React 👍 to approve or 👎 to reject, or reply "
                f"'approve {pid}' / 'reject {pid}'.",
                attachments=attachments,
            )
            if signal_ts:
                set_proposal_signal_timestamp(pid, signal_ts)
        except Exception:
            pass
        logger.info(f"Evolution: created code proposal #{pid} — {hypothesis}")
        return None

    else:
        logger.info(f"Evolution: unknown action '{action}', skipping")
        return None


# ── Auto-deploy trigger for kept code mutations ──────────────────────────────

def _trigger_code_auto_deploy(result, mutation) -> None:
    """Schedule auto-deploy or queue for human review based on confidence.

    HIGH confidence (delta > 0.05, eval-confirmed, low risk) → auto-deploy.
    BORDERLINE confidence → queue for human approval via human_gate.
    LOW confidence → caller would not have called us here.

    The classification considers self_model centrality and hot-path status
    so high-blast-radius changes get extra scrutiny even when delta is large.
    """
    try:
        from app.auto_deployer import (
            schedule_deploy, validate_proposal_paths,
            get_protection_tier, ProtectionTier,
        )

        # Pre-flight: check if any file is IMMUTABLE (would be rejected anyway)
        files = mutation.files or {}
        for path in files:
            tier = get_protection_tier(path)
            if tier == ProtectionTier.IMMUTABLE:
                logger.info(
                    f"Evolution: skipping auto-deploy for {result.experiment_id} — "
                    f"contains IMMUTABLE file {path}"
                )
                return

        # Path validation (catches path traversal, absolute paths, etc.)
        violations = validate_proposal_paths(files)
        if violations:
            logger.info(
                f"Evolution: skipping auto-deploy for {result.experiment_id} — "
                f"path violations: {violations[:3]}"
            )
            return

        # Confidence classification — borderline mutations route through human_gate
        try:
            from app.human_gate import classify_confidence, request_approval, ConfidenceTier
            from app.self_model import is_hot_path, get_centrality_score

            high_centrality = any(get_centrality_score(p) > 0.30 for p in files)
            on_hot_path = any(is_hot_path(p) for p in files)
            tier_decision, reason = classify_confidence(
                delta=result.delta,
                eval_measured=True,  # if we got here from a kept code mutation, eval ran
                has_high_centrality_files=high_centrality,
                is_hot_path=on_hot_path,
            )

            if tier_decision == ConfidenceTier.BORDERLINE:
                request_approval(
                    experiment_id=result.experiment_id,
                    hypothesis=result.hypothesis,
                    change_type=result.change_type,
                    files=files,
                    delta=result.delta,
                    confidence_reason=reason,
                )
                logger.info(
                    f"Evolution: queued borderline mutation {result.experiment_id} "
                    f"for human review ({reason})"
                )
                return  # Wait for human decision; do NOT auto-deploy
        except Exception as exc:
            logger.debug(f"Evolution: confidence classification failed (defaulting to auto-deploy): {exc}")

        # HIGH confidence: schedule deploy — auto_deployer handles the rest
        schedule_deploy(reason=f"evolution-keep-{result.experiment_id}")
        logger.info(
            f"Evolution: scheduled auto-deploy for {result.experiment_id} "
            f"(delta={result.delta:+.4f}, files={list(files.keys())[:3]})"
        )
    except Exception as exc:
        logger.warning(f"Evolution: auto-deploy trigger failed: {exc}")


# ── Evolution session ────────────────────────────────────────────────────────

def run_evolution_session(max_iterations: int = 5) -> str:
    """
    Run a multi-experiment evolution session (autoresearch-style).

    This is the core loop: propose → apply → measure → keep/discard, repeated
    N times. Each iteration builds on the results of the previous one.

    When config.evolution_engine == "shinka", delegates to ShinkaEvolve's
    island-model MAP-Elites engine instead of the AVO pipeline.

    Args:
        max_iterations: How many experiments to run this session (default 5)

    Returns:
        Summary of all experiments run
    """
    # ── ROI throttle gate ──────────────────────────────────────────────────────
    # If recent ROI is poor (no real improvements for 14 days, high rollback
    # rate, or cost-per-improvement above threshold), reduce iterations
    # rather than burning more cost on patterns that aren't working.
    try:
        from app.evolution_roi import should_throttle
        throttled, reason, factor = should_throttle()
        if throttled:
            adjusted = max(1, int(max_iterations * factor))
            logger.warning(
                f"Evolution throttled to {factor:.0%} ({adjusted}/{max_iterations} iterations): {reason}"
            )
            max_iterations = adjusted
    except Exception:
        pass

    # ── Dynamic engine selection ──────────────────────────────────────────────
    # Automatically pick the best engine based on recent performance, SUBIA
    # safety, and stagnation detection. Manual override via config still works.
    engine = _select_evolution_engine()
    logger.info(f"Evolution engine selected: {engine}")

    if engine == "shinka":
        return _run_shinka_session(max_iterations)

    # ── SUBIA bridge: homeostatic aggressiveness modulation ────────────────
    # Read the SUBIA safety variable to dynamically adjust evolution posture.
    # High safety → aggressive (more iterations, TIER_GATED allowed)
    # Low safety → conservative (fewer iterations, TIER_OPEN only)
    _evolution_aggressiveness = "normal"
    _subia_safety = 0.8  # default: assume healthy
    try:
        from app.subia.kernel import get_active_kernel
        kernel = get_active_kernel()
        if kernel and hasattr(kernel, "homeostasis"):
            _subia_safety = kernel.homeostasis.variables.get("safety", 0.8)
            if _subia_safety > 0.92:
                _evolution_aggressiveness = "aggressive"
                max_iterations = int(max_iterations * 1.5)
                os.environ["_EVOLUTION_ALLOW_GATED"] = "true"
            elif _subia_safety < 0.70:
                _evolution_aggressiveness = "conservative"
                max_iterations = max(1, max_iterations // 2)
                os.environ["_EVOLUTION_ALLOW_GATED"] = "false"
            else:
                os.environ.pop("_EVOLUTION_ALLOW_GATED", None)
        logger.info(
            f"Evolution aggressiveness: {_evolution_aggressiveness} "
            f"(safety={_subia_safety:.2f}, iterations={max_iterations})"
        )
    except Exception:
        pass

    # Step 9/6A: Rate limiting — max promotions per day (dynamic based on safety)
    _base_daily_limit = int(os.environ.get("EVOLUTION_MAX_DAILY_PROMOTIONS", "10"))
    if _evolution_aggressiveness == "conservative":
        _MAX_DAILY_PROMOTIONS = max(1, _base_daily_limit // 3)
    elif _evolution_aggressiveness == "aggressive":
        _MAX_DAILY_PROMOTIONS = _base_daily_limit
    else:
        _MAX_DAILY_PROMOTIONS = _base_daily_limit
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

    # Adaptive ensemble controller — manages explore/exploit balance
    try:
        from app.adaptive_ensemble import get_controller
        _evo_controller = get_controller()
    except Exception:
        _evo_controller = None

    # Create evolution run record in PostgreSQL (always, not gated by env var)
    dgm_run_id = None
    try:
        from app.evolution_db.archive_db import create_run
        ctl_stats = _evo_controller.get_stats() if _evo_controller else {}
        dgm_run_id = create_run(
            agent_name="system",
            target_type="mixed",
            max_generations=max_iterations,
            config={
                "avo_enabled": True,
                "rate_limit": _MAX_DAILY_PROMOTIONS,
                "exploration_rate": ctl_stats.get("exploration_rate", 0),
                "phase": ctl_stats.get("ensemble", {}).get("phase", "unknown"),
            },
        )
        logger.info(f"Evolution run {dgm_run_id[:8]} created in PostgreSQL")
    except Exception as e:
        logger.debug(f"Failed to create evolution run in PG: {e}")

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

            # Inject adaptive ensemble strategy hint into context
            if _evo_controller:
                try:
                    strategy = _evo_controller.select_mutation_strategy()
                    rate = _evo_controller.exploration_rate
                    phase = _evo_controller.ensemble.phase
                    context += (
                        f"\n\n## Evolution Strategy (adaptive ensemble)\n"
                        f"Current phase: {phase} (exploration_rate={rate:.2f})\n"
                        f"Suggested strategy: {strategy}\n"
                        f"- meta_prompt: try new prompt structures/approaches\n"
                        f"- random: explore entirely new directions\n"
                        f"- inspiration: cross-pollinate from successful past experiments\n"
                        f"- depth_exploit: refine the most promising recent change\n"
                    )
                except Exception:
                    pass

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
            # Acquire workspace lock to prevent concurrent evolution conflicts.
            try:
                from app.workspace_versioning import WorkspaceLock, workspace_commit
                with WorkspaceLock():
                    if mutation.change_type == "skill":
                        result = _measure_skill_impact(runner, mutation)
                    else:
                        result = runner.run_experiment(mutation)
                    # Git-commit promoted mutations for rollback safety
                    if result.status == "keep":
                        workspace_commit(f"evolution: {mutation.hypothesis[:80]}")
                        # Fix 5: Auto-deploy code mutations to production.
                        # Without this, kept code mutations live only in the
                        # workspace — never reaching /app/ where they'd take
                        # effect. The auto_deployer enforces TIER protection,
                        # canary gating, and post-deploy error monitoring.
                        if result.change_type == "code":
                            _trigger_code_auto_deploy(result, mutation)
            except (ImportError, TimeoutError) as _lock_err:
                logger.warning(f"Evolution: workspace lock unavailable ({_lock_err}), running unlocked")
                if mutation.change_type == "skill":
                    result = _measure_skill_impact(runner, mutation)
                else:
                    result = runner.run_experiment(mutation)
                if result.status == "keep" and result.change_type == "code":
                    _trigger_code_auto_deploy(result, mutation)

            # 5. Track results + store in variant archive (DGM genealogy)
            if result.status == "keep":
                kept += 1
            elif result.status == "discard":
                discarded += 1
            else:
                crashed += 1

            # Record cost / outcome for ROI tracking
            try:
                from app.evolution_roi import record_evolution_cost
                # Estimate cost: AVO cycle uses ~3-4 LLM calls (planning, impl, critique).
                # Cost varies by tier; conservative estimate ~$0.05-$0.20 per cycle.
                # When request_cost_tracker is available, use the precise figure.
                cost_estimate = 0.10  # USD per AVO cycle (conservative)
                try:
                    from app.rate_throttle import get_request_cost_estimate
                    actual = get_request_cost_estimate()
                    if actual is not None and actual > 0:
                        cost_estimate = actual
                except Exception:
                    pass

                record_evolution_cost(
                    experiment_id=result.experiment_id,
                    engine="avo",
                    cost_usd=cost_estimate,
                    delta=result.delta,
                    status=result.status,
                    deployed=(result.status == "keep" and result.change_type == "code"),
                )
            except Exception as exc:
                logger.debug(f"Evolution: ROI recording failed: {exc}")

            # Update mutation strategy success stats (if AVO sampled a strategy)
            try:
                from app.mutation_strategies import update_strategy_success
                # AVO logs strategy at info level; we don't pass it through here,
                # so update_strategy_success is best-effort with the change_type.
                strategy_name = result.change_type  # "code" | "skill" maps to broad category
                update_strategy_success(strategy_name, result.status == "keep", result.delta)
            except Exception:
                pass

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

            # Store variant in PostgreSQL archive + run LLM judge for kept variants
            if dgm_run_id:
                try:
                    _store_dgm_variant(mutation, result, dgm_run_id, i)
                except Exception:
                    logger.debug("Failed to store variant in PG", exc_info=True)

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

            # Adaptive ensemble: update explore/exploit balance based on fitness
            _phase_info = ""
            if _evo_controller:
                try:
                    info = _evo_controller.step(result.metric_after)
                    _phase_info = f" [phase={info.get('phase','?')}, expl={info.get('exploration_rate',0):.2f}]"
                except Exception:
                    pass

            results_summary.append(
                f"  [{i + 1}] {result.status:7s} {result.delta:+.4f} | "
                f"{result.hypothesis[:60]}{_phase_info}"
            )

            logger.info(
                f"Evolution iteration {i + 1}: {result.status} "
                f"({result.detail}){_phase_info}"
            )

        # DGM-DB: Mark run as completed
        if dgm_run_id:
            try:
                from app.evolution_db.archive_db import update_run
                update_run(dgm_run_id, status="completed")
            except Exception:
                pass

        _ctl_report = ""
        if _evo_controller:
            try:
                _ctl_report = f"\n{_evo_controller.format_report()}"
            except Exception:
                pass

        summary = (
            f"Evolution session complete: {max_iterations} iterations\n"
            f"Kept: {kept}, Discarded: {discarded}, Crashed: {crashed}\n\n"
            + "\n".join(results_summary)
            + _ctl_report
        )

        # Record evolution session in activity journal
        try:
            from app.self_awareness.journal import get_journal, JournalEntry, JournalEntryType
            get_journal().write(JournalEntry(
                entry_type=JournalEntryType.EVOLUTION_RESULT,
                summary=f"Evolution: {kept} kept, {discarded} discarded, {crashed} crashed",
                agents_involved=["self_improvement"],
                outcome="kept" if kept > 0 else "no_improvement",
                details={"kept": kept, "discarded": discarded, "crashed": crashed,
                          "iterations": max_iterations},
            ))
        except Exception:
            pass

        # Store causal belief about evolution effectiveness
        try:
            from app.subia.belief.world_model import store_causal_belief
            if kept > 0:
                store_causal_belief(
                    cause=f"Evolution session with {max_iterations} iterations",
                    effect=f"{kept} improvements kept (ratio={kept/max(1,max_iterations):.2f})",
                    confidence="medium",
                    source="evolution_session",
                )
        except Exception:
            pass

        tracker = stop_request_tracking()
        _tokens = tracker.total_tokens if tracker else 0
        _model = ", ".join(sorted(tracker.models_used)) if tracker and tracker.models_used else ""
        _cost = tracker.total_cost_usd if tracker else 0.0
        crew_completed("self_improvement", task_id, summary[:2000],
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


# ── Dynamic engine selection ──────────────────────────────────────────────────

# Minimum days between forced ShinkaEvolve sessions. Guarantees the
# alternative engine gets exercised even when AVO appears to be performing
# well — without this, the kept_ratio>0.60 gate locks the system into AVO
# and ShinkaEvolve never accumulates ROI data for comparison.
_SHINKA_ROTATION_INTERVAL_DAYS = 7.0


def _select_evolution_engine() -> str:
    """Dynamically select the best evolution engine for this session.

    Decision logic (evaluated in priority order):

      1. Manual override: config.evolution_engine in ("avo", "shinka") wins.

      2. Availability: ShinkaEvolve missing → AVO.

      3. SUBIA safety < 0.70 → AVO (conservative single-mutation is safer).

      4. AVO stagnation (5 consecutive failures) → ShinkaEvolve (break out).

      5. Forced rotation: ShinkaEvolve hasn't run in N days → ShinkaEvolve.
         This guarantees exploration even when AVO appears healthy. Without
         this rule, the kept_ratio gate (rule 6) locks the selector into
         AVO permanently and ShinkaEvolve never accumulates ROI data.

      6. AVO performing well (kept_ratio > 0.60) → AVO. Don't fix what
         isn't broken — but only after rule 5 confirms ShinkaEvolve has
         had recent exposure.

      7. AVO too ambitious (kept_ratio < 0.20) → ShinkaEvolve.

      8. Undiagnosed errors ≥ 3 → AVO (has error context for targeting).

      9. ROI recommendation: when both engines have data, defer to whichever
         has lower cost-per-improvement. When only one has data, use it.

     10. Default → AVO.

    Returns:
        "avo" or "shinka"
    """
    # 1. Manual override
    try:
        manual = get_settings().evolution_engine
        if manual in ("avo", "shinka"):
            return manual
    except AttributeError:
        pass

    # 2. ShinkaEvolve availability check
    if not _is_shinka_available():
        return "avo"

    # 3. SUBIA safety check — conservative mode uses AVO
    subia_safety = _get_subia_safety_value()
    if subia_safety < 0.70:
        logger.info(f"Engine selector: AVO (SUBIA safety={subia_safety:.2f} < 0.70, conservative)")
        return "avo"

    # 4-9: Analyze recent evolution performance
    recent = get_recent_results(10)

    # 4. Stagnation detection — last 5 all failed → switch to ShinkaEvolve
    if len(recent) >= 5:
        last_5 = recent[:5]
        if all(r.get("status") in ("discard", "crash") for r in last_5):
            logger.info("Engine selector: ShinkaEvolve (AVO stagnated — 5 consecutive failures)")
            return "shinka"

    # 5. Forced rotation — ensure ShinkaEvolve gets fresh data periodically.
    # Placed BEFORE the kept_ratio gate so that AVO performing well doesn't
    # starve ShinkaEvolve of exploration opportunities.
    try:
        from app.evolution_roi import days_since_engine_run
        days_since_shinka = days_since_engine_run("shinka")
        if days_since_shinka >= _SHINKA_ROTATION_INTERVAL_DAYS:
            elapsed = "never" if days_since_shinka == float("inf") else f"{days_since_shinka:.1f}d ago"
            logger.info(
                f"Engine selector: ShinkaEvolve (forced rotation — last run {elapsed}, "
                f"interval {_SHINKA_ROTATION_INTERVAL_DAYS}d)"
            )
            return "shinka"
    except Exception as exc:
        logger.debug(f"Engine selector: rotation check failed: {exc}")

    if not recent:
        return "avo"  # No history → start with AVO

    # 6. AVO performing well → stay with AVO
    kept = sum(1 for r in recent if r.get("status") == "keep")
    kept_ratio = kept / len(recent)
    if kept_ratio > 0.60:
        logger.info(f"Engine selector: AVO (kept_ratio={kept_ratio:.2f} > 0.60, performing well)")
        return "avo"

    # 7. AVO mutations too ambitious → try ShinkaEvolve
    if kept_ratio < 0.20 and len(recent) >= 5:
        logger.info(f"Engine selector: ShinkaEvolve (kept_ratio={kept_ratio:.2f} < 0.20, too ambitious)")
        return "shinka"

    # 8. Undiagnosed errors → AVO (has error context)
    try:
        from app.healing.error_diagnosis import get_recent_errors
        undiagnosed = [e for e in get_recent_errors(10) if not e.get("diagnosed")]
        if len(undiagnosed) >= 3:
            logger.info(f"Engine selector: AVO ({len(undiagnosed)} undiagnosed errors, needs targeted fix)")
            return "avo"
    except Exception:
        pass

    # 9. ROI recommendation — let cost-per-improvement decide.
    # Replaces the previous count-modulo rotation, which was too weak to
    # ever fire reliably. This rule fires only when neither engine is
    # clearly indicated by rules 4-8 — true ambiguity warrants ROI data.
    try:
        from app.evolution_roi import get_engine_recommendation, get_rolling_roi
        rec = get_engine_recommendation()
        # Only trust the recommendation when there's enough data to compare
        snapshot = get_rolling_roi(days=14)
        avo_data = snapshot.by_engine.get("avo", {})
        shinka_data = snapshot.by_engine.get("shinka", {})
        if avo_data.get("real_improvements", 0) >= 1 and shinka_data.get("real_improvements", 0) >= 1:
            logger.info(
                f"Engine selector: {rec} (ROI recommendation — "
                f"avo cpi={avo_data.get('cost_per_improvement')}, "
                f"shinka cpi={shinka_data.get('cost_per_improvement')})"
            )
            return rec
    except Exception as exc:
        logger.debug(f"Engine selector: ROI check failed: {exc}")

    # 10. Default
    return "avo"


def _is_shinka_available() -> bool:
    """Check ShinkaEvolve is *actually runnable*, not just installed.

    Verifies the deep imports the engine needs at session start —
    ``shinka.core``, ``shinka.launch``, ``shinka.database``. The
    historical version only did ``import shinka`` (the empty package
    namespace), which silently passed when transitive deps like
    ``google-genai``, ``psutil``, ``seaborn``, or ``python-Levenshtein``
    were missing — and they were, because the Dockerfile installs
    shinka with ``--no-deps``. The selector then kept picking
    ShinkaEvolve session after session, each one crashing immediately
    at ``run_shinka_session()`` with ``ImportError`` and writing
    nothing to either ledger. ``days_since_engine_run("shinka")``
    stayed at infinity → forced rotation kept firing → forever loop.

    Failure now means: log once, return False, let rule 1-10 flow
    through to AVO.
    """
    try:
        # The full chain of imports the engine actually performs at
        # session start. Any one of these missing means the session
        # WILL crash at LLM-init time — better to know now and let
        # the selector pick AVO instead.
        from shinka.core import (  # noqa: F401
            ShinkaEvolveRunner,
            EvolutionConfig,
        )
        from shinka.launch import LocalJobConfig  # noqa: F401
        from shinka.database import DatabaseConfig  # noqa: F401
    except ImportError as exc:
        logger.warning(
            "shinka unavailable: %s — engine selector will fall back to AVO. "
            "Install missing dep(s) and restart the gateway to enable.", exc,
        )
        return False
    from pathlib import Path
    initial = Path("/app/workspace/shinka/initial.py")
    evaluate = Path("/app/workspace/shinka/evaluate.py")
    return initial.exists() and evaluate.exists()


def _get_subia_safety_value() -> float:
    """Read the SUBIA homeostatic safety variable. Returns 0.8 as default."""
    try:
        from app.subia.kernel import get_active_kernel
        kernel = get_active_kernel()
        if kernel and hasattr(kernel, "homeostasis"):
            return kernel.homeostasis.variables.get("safety", 0.8)
    except Exception:
        pass
    return 0.8


# ── ShinkaEvolve engine dispatch ──────────────────────────────────────────────

def _run_shinka_session(max_iterations: int) -> str:
    """Delegate evolution to ShinkaEvolve's island-model MAP-Elites engine.

    Called when config.evolution_engine == "shinka". Runs ShinkaEvolve with
    the configured number of generations and records results in the standard
    results ledger for unified tracking.
    """
    try:
        from app.shinka_engine import run_shinka_session
    except ImportError as e:
        logger.error(f"ShinkaEvolve not available: {e}")
        return f"ShinkaEvolve not installed: {e}"

    logger.info(f"Evolution engine: ShinkaEvolve ({max_iterations} generations)")

    result = run_shinka_session(
        num_generations=max_iterations,
        num_islands=2,
        max_eval_jobs=2,
        max_proposal_jobs=2,
    )

    summary = (
        f"ShinkaEvolve: {result.status} | "
        f"score={result.best_score:.4f} (delta={result.delta:+.4f}) | "
        f"{result.generations_run} generations | "
        f"{result.duration_seconds:.0f}s"
    )

    if result.error:
        summary += f" | error: {result.error[:100]}"

    logger.info(f"ShinkaEvolve session complete: {summary}")
    return summary


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
