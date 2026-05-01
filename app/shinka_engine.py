"""
shinka_engine.py — ShinkaEvolve integration for AndrusAI.

Wraps ShinkaEvolve's ShinkaEvolveRunner as an alternative evolution engine
that can be selected via config.evolution_engine = "shinka".

ShinkaEvolve uses island-model MAP-Elites with LLM-generated patches
and EVOLVE-BLOCK markers for targeted code mutation. It provides:
  - Multi-island population with migration
  - UCB1 model selection across multiple LLMs
  - Diff, full-replacement, and crossover patch types
  - Novelty scoring via code embeddings
  - Async parallel evaluation

The integration:
  1. Reads AndrusAI's LLM configuration and maps to ShinkaEvolve model strings
  2. Points ShinkaEvolve at workspace/shinka/initial.py and evaluate.py
  3. Runs a bounded evolution session (num_generations from config)
  4. Extracts the best variant and applies it to the workspace
  5. Records results in the standard results_ledger

TIER_IMMUTABLE — this module is part of the evolution infrastructure.
"""

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SHINKA_DIR = Path("/app/workspace/shinka")
SHINKA_RESULTS_DIR = Path("/app/workspace/shinka_results")
INITIAL_PY = SHINKA_DIR / "initial.py"
EVALUATE_PY = SHINKA_DIR / "evaluate.py"


@dataclass
class ShinkaResult:
    """Result from a ShinkaEvolve evolution session."""
    status: str  # "improved", "no_improvement", "error", "skipped"
    best_score: float = 0.0
    baseline_score: float = 0.0
    delta: float = 0.0
    generations_run: int = 0
    variants_evaluated: int = 0
    best_program_path: str = ""
    error: str = ""
    duration_seconds: float = 0.0


def _read_subia_safety() -> str:
    """Read SUBIA homeostatic safety variable and return posture string.

    Returns:
        "aggressive" if safety > 0.92 (system healthy, take bigger risks)
        "conservative" if safety < 0.70 (integrity concerns, be careful)
        "normal" otherwise
    """
    try:
        from app.subia.kernel import get_active_kernel
        kernel = get_active_kernel()
        if kernel and hasattr(kernel, "homeostasis"):
            safety = kernel.homeostasis.variables.get("safety", 0.8)
            logger.info(f"shinka_engine: SUBIA safety={safety:.2f}")
            if safety > 0.92:
                return "aggressive"
            elif safety < 0.70:
                return "conservative"
    except Exception as e:
        logger.debug(f"shinka_engine: SUBIA read failed (normal fallback): {e}")
    return "normal"


def _build_subia_task_prompt() -> str:
    """Build a SUBIA-aware task prompt for ShinkaEvolve's LLM.

    Injects prediction failure context so ShinkaEvolve's mutations
    are steered toward fixing blind spots identified by SUBIA.
    """
    base_prompt = (
        "You are optimizing agent utility functions for AndrusAI. "
        "Improve tool selection accuracy, response formatting quality, "
        "and task routing correctness. The combined_score measures "
        "accuracy across test cases — higher is better."
    )

    # Add SUBIA surprise signals if available
    try:
        from app.subia.prediction.accuracy_tracker import get_tracker
        tracker = get_tracker()
        summary = tracker.all_domains_summary()
        weak_domains = []
        for domain, stats in summary.items():
            if tracker.has_sustained_error(domain):
                acc = stats.get("mean_accuracy", 0)
                weak_domains.append(f"{domain} (accuracy={acc:.2f})")

        if weak_domains:
            base_prompt += (
                "\n\nPRIORITY: The system has sustained prediction failures in these areas: "
                + ", ".join(weak_domains[:5])
                + ". Focus improvements on these weak spots."
            )
    except Exception:
        pass

    # Add homeostatic context
    try:
        from app.subia.kernel import get_active_kernel
        kernel = get_active_kernel()
        if kernel and hasattr(kernel, "homeostasis"):
            vars_ = kernel.homeostasis.variables
            safety = vars_.get("safety", 0.8)
            coherence = vars_.get("coherence", 0.5)
            if safety < 0.80:
                base_prompt += (
                    f"\n\nCAUTION: System safety is low ({safety:.2f}). "
                    "Prefer conservative, well-tested changes over ambitious rewrites."
                )
            if coherence < 0.50:
                base_prompt += (
                    f"\n\nNOTE: System coherence is low ({coherence:.2f}). "
                    "Prioritize consistency and reliability improvements."
                )
    except Exception:
        pass

    return base_prompt


def _map_llm_models() -> list[str]:
    """Map AndrusAI's LLM configuration to ShinkaEvolve model strings.

    Returns model identifiers that ShinkaEvolve's registry recognises —
    NOT Bedrock-style ARNs. Run ``shinka_models --verbose`` inside the
    container to see the live allowlist; the strings below are pulled
    from there.

    2026-04-30 fix: previous mapping returned ``us.anthropic.claude-
    sonnet-4-20250514-v1:0`` (Bedrock; needs AWS_* env that we don't
    set) and ``openrouter/deepseek/deepseek-chat-v3-0324`` (not in
    ShinkaEvolve's OpenRouter allowlist). Result: every ShinkaEvolve
    session crashed at LLM-init with ``Requested model(s) are
    unavailable``, no ledger record was written, and the engine
    selector kept picking ShinkaEvolve forever (since
    ``days_since_engine_run("shinka")`` stayed at ``inf``).
    """
    models: list[str] = []

    # Anthropic — direct API (NOT Bedrock).
    try:
        from app.config import get_settings
        settings = get_settings()
        if hasattr(settings, "anthropic_api_key"):
            key = settings.anthropic_api_key
            if hasattr(key, "get_secret_value"):
                key = key.get_secret_value()
            if key and len(key) > 10:
                # ShinkaEvolve registry exposes Sonnet 4.6 as
                # ``claude-sonnet-4-6`` (the current production sonnet).
                models.append("claude-sonnet-4-6")
    except Exception:
        pass

    # OpenRouter — pick a coding-strong model that's in shinka's allowlist.
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    if openrouter_key:
        models.append("qwen/qwen3-coder")

    # Local Ollama (still optional — useful when API budget is tight).
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        import requests
        resp = requests.get(f"{ollama_host}/api/tags", timeout=2)
        if resp.status_code == 200:
            tags = resp.json().get("models", [])
            for tag in tags:
                name = tag.get("name", "")
                if "coder" in name.lower() or "qwen" in name.lower():
                    models.append(f"local/{name}@{ollama_host}/v1")
                    break
    except Exception:
        pass

    # Fallback when no API keys are set: OpenRouter via the same coder
    # we map above. (Trying ShinkaEvolve without ANY model is pointless
    # but the safety net keeps the function total.)
    if not models:
        models.append("qwen/qwen3-coder")

    return models


def _get_embedding_model() -> str | None:
    """Get embedding model for novelty scoring.

    Returns None to disable embedding-based novelty (uses LLM-based instead).
    Embeddings require an OpenAI key which we may not have.
    """
    if os.environ.get("OPENAI_API_KEY"):
        return "text-embedding-3-small"
    return None


def run_shinka_session(
    num_generations: int = 20,
    num_islands: int = 2,
    max_eval_jobs: int = 2,
    max_proposal_jobs: int = 2,
) -> ShinkaResult:
    """Run a ShinkaEvolve evolution session.

    This is the main entry point called from evolution.py when
    config.evolution_engine == "shinka".

    Args:
        num_generations: How many generations to evolve.
        num_islands: Number of population islands.
        max_eval_jobs: Concurrent evaluation jobs.
        max_proposal_jobs: Concurrent LLM proposal jobs.

    Returns:
        ShinkaResult with status, scores, and best program path.
    """
    start = time.monotonic()

    # Gate: verify shinka files exist
    if not INITIAL_PY.exists():
        return ShinkaResult(
            status="skipped",
            error=f"Missing {INITIAL_PY}",
            duration_seconds=time.monotonic() - start,
        )
    if not EVALUATE_PY.exists():
        return ShinkaResult(
            status="skipped",
            error=f"Missing {EVALUATE_PY}",
            duration_seconds=time.monotonic() - start,
        )

    # Measure baseline from current initial.py
    baseline_score = _measure_baseline()

    # ── SUBIA bridge: homeostatic safety modulation ─────────────────────────
    # Read the safety variable from SUBIA's homeostatic state to dynamically
    # adjust ShinkaEvolve's aggressiveness — same pattern as AVO in evolution.py.
    #
    # High safety (>0.92) → AGGRESSIVE: more generations, more islands, higher budget
    # Normal (0.70-0.92)  → STANDARD: default parameters
    # Low safety (<0.70)  → CONSERVATIVE: fewer generations, single island, lower budget
    subia_posture = _read_subia_safety()
    if subia_posture == "aggressive":
        num_generations = int(num_generations * 1.5)
        num_islands = max(num_islands, 3)
        max_eval_jobs = max(max_eval_jobs, 3)
        max_proposal_jobs = max(max_proposal_jobs, 3)
        api_budget = 8.0
        logger.info("shinka_engine: AGGRESSIVE posture (SUBIA safety > 0.92)")
    elif subia_posture == "conservative":
        num_generations = max(5, num_generations // 2)
        num_islands = 1
        max_eval_jobs = 1
        max_proposal_jobs = 1
        api_budget = 2.0
        logger.info("shinka_engine: CONSERVATIVE posture (SUBIA safety < 0.70)")
    else:
        api_budget = 5.0
        logger.info("shinka_engine: NORMAL posture")

    try:
        from shinka.core import ShinkaEvolveRunner, EvolutionConfig
        from shinka.launch import LocalJobConfig
        from shinka.database import DatabaseConfig
    except ImportError as e:
        return ShinkaResult(
            status="error",
            error=f"ShinkaEvolve not installed: {e}",
            baseline_score=baseline_score,
            duration_seconds=time.monotonic() - start,
        )

    # Prepare results directory
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = SHINKA_RESULTS_DIR / f"run_{ts}"
    results_dir.mkdir(parents=True, exist_ok=True)

    llm_models = _map_llm_models()
    embedding_model = _get_embedding_model()
    logger.info(
        f"shinka_engine: models={llm_models}, embedding={embedding_model}, "
        f"gens={num_generations}, islands={num_islands}, budget=${api_budget}"
    )

    # Build SUBIA-aware task prompt with prediction failure context
    task_prompt = _build_subia_task_prompt()

    try:
        evo_config = EvolutionConfig(
            num_generations=num_generations,
            init_program_path=str(INITIAL_PY),
            results_dir=str(results_dir),
            language="python",
            llm_models=llm_models,
            llm_dynamic_selection="ucb" if len(llm_models) > 1 else None,
            embedding_model=embedding_model,
            task_sys_msg=task_prompt,
            max_api_costs=api_budget,
        )

        job_config = LocalJobConfig(
            eval_program_path=str(EVALUATE_PY),
        )

        db_config = DatabaseConfig(
            db_path=str(results_dir / "evolution_db.sqlite"),
            num_islands=num_islands,
            archive_size=max(20, num_generations),
            migration_interval=max(5, num_generations // 4),
        )

        runner = ShinkaEvolveRunner(
            evo_config=evo_config,
            job_config=job_config,
            db_config=db_config,
            max_evaluation_jobs=max_eval_jobs,
            max_proposal_jobs=max_proposal_jobs,
            verbose=True,
        )

        logger.info(f"shinka_engine: starting {num_generations} generations on {num_islands} islands")
        runner.run()

        # Extract results
        best_score, best_path = _extract_best_result(results_dir)
        delta = best_score - baseline_score

        duration = time.monotonic() - start

        if delta > 0 and best_path:
            # Apply the best variant to the workspace
            _apply_best_variant(best_path)
            logger.info(
                f"shinka_engine: improved! score={best_score:.4f} "
                f"(delta={delta:+.4f}, {duration:.0f}s)"
            )

            # Record in results ledger
            _record_result(baseline_score, best_score, delta, "keep")

            return ShinkaResult(
                status="improved",
                best_score=best_score,
                baseline_score=baseline_score,
                delta=delta,
                generations_run=num_generations,
                best_program_path=str(best_path),
                duration_seconds=duration,
            )
        else:
            logger.info(
                f"shinka_engine: no improvement (best={best_score:.4f}, "
                f"baseline={baseline_score:.4f}, {duration:.0f}s)"
            )
            _record_result(baseline_score, best_score, delta, "discard")

            return ShinkaResult(
                status="no_improvement",
                best_score=best_score,
                baseline_score=baseline_score,
                delta=delta,
                generations_run=num_generations,
                duration_seconds=duration,
            )

    except Exception as e:
        duration = time.monotonic() - start
        logger.error(f"shinka_engine: error: {e}")
        return ShinkaResult(
            status="error",
            error=str(e)[:500],
            baseline_score=baseline_score,
            duration_seconds=duration,
        )


def _measure_baseline() -> float:
    """Run the current initial.py through the test suite to get baseline score."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("initial", str(INITIAL_PY))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            score, _ = mod.run_evaluation()
            return float(score)
    except Exception as e:
        logger.warning(f"shinka_engine: baseline measurement failed: {e}")
    return 0.0


def _extract_best_result(results_dir: Path) -> tuple[float, Path | None]:
    """Extract the best program and its score from ShinkaEvolve results.

    ShinkaEvolve stores results in an SQLite database. We also check
    for best_metrics.json as a simpler fallback.
    """
    best_score = 0.0
    best_path = None

    # Try SQLite database first
    try:
        from shinka.database import ProgramDatabase
        db_path = results_dir / "evolution_db.sqlite"
        if db_path.exists():
            db = ProgramDatabase.load(str(db_path))
            best = db.get_best_program()
            if best and best.fitness > best_score:
                best_score = best.fitness
                # Write best program to a file
                best_file = results_dir / "best_program.py"
                best_file.write_text(best.program_str)
                best_path = best_file
    except Exception as e:
        logger.debug(f"shinka_engine: DB extraction failed: {e}")

    # Fallback: check for best_metrics.json
    metrics_file = results_dir / "best_metrics.json"
    if metrics_file.exists():
        try:
            data = json.loads(metrics_file.read_text())
            score = data.get("combined_score", 0.0)
            if score > best_score:
                best_score = score
                program_file = results_dir / "best_program.py"
                if program_file.exists():
                    best_path = program_file
        except Exception:
            pass

    return best_score, best_path


def _apply_best_variant(best_path: Path) -> None:
    """Apply the best evolved variant back to workspace/shinka/initial.py.

    Creates a backup before overwriting.
    """
    try:
        # Backup current
        backup = INITIAL_PY.with_suffix(".py.bak")
        shutil.copy2(INITIAL_PY, backup)

        # Apply best variant
        shutil.copy2(best_path, INITIAL_PY)

        # Git commit
        from app.workspace_versioning import workspace_commit
        workspace_commit("evolution: ShinkaEvolve improved initial.py")

        logger.info(f"shinka_engine: applied best variant from {best_path}")
    except Exception as e:
        logger.error(f"shinka_engine: failed to apply variant: {e}")


def _record_result(
    baseline: float,
    after: float,
    delta: float,
    status: str,
) -> None:
    """Record the ShinkaEvolve result in BOTH ledgers.

    1. ``results_ledger`` (shared with AVO) — used by the evolution
       dashboard, retrospective crew, and ``get_recent_results`` for
       per-experiment history.

    2. ``evolution_roi`` — used by the engine selector's forced-rotation
       rule (``days_since_engine_run("shinka")``) and ROI recommendation.

    Pre-2026-04-30 only #1 fired, which meant ``days_since_engine_run``
    always returned ``inf`` even after a successful shinka run — keeping
    the forced-rotation rule firing every cycle. Diagnosed alongside
    the missing-deps issue.
    """
    try:
        from app.results_ledger import record_experiment
        from app.experiment_runner import generate_experiment_id
        experiment_id = generate_experiment_id("shinka-evolve")
        record_experiment(
            experiment_id=experiment_id,
            hypothesis="ShinkaEvolve island-model evolution of agent utilities",
            change_type="code",
            metric_before=baseline,
            metric_after=after,
            status=status,
            files_changed="workspace/shinka/initial.py",
            detail=f"ShinkaEvolve delta={delta:+.4f}",
        )
    except Exception as e:
        logger.warning(f"shinka_engine: results_ledger recording failed: {e}")
        return

    # ── ROI ledger — what the engine selector rule 5 reads ────────────
    try:
        from app.evolution_roi import record_evolution_cost
        # Cost estimate: ShinkaEvolve sessions burn $1-5 of API budget
        # depending on generations × islands; the api_budget cap in
        # run_shinka_session is the upper bound. Use 2.0 as a midpoint
        # estimate when we don't have a precise tracker handle. A
        # future improvement: thread the actual usage from shinka's
        # cost summary back here.
        cost_estimate = 2.0
        try:
            from app.rate_throttle import get_request_cost_estimate
            actual = get_request_cost_estimate()
            if actual is not None and actual > 0:
                cost_estimate = actual
        except Exception:
            pass
        record_evolution_cost(
            experiment_id=experiment_id,
            engine="shinka",
            cost_usd=cost_estimate,
            delta=delta,
            status=status,
            deployed=(status == "keep"),
        )
    except Exception as e:
        logger.warning(f"shinka_engine: ROI recording failed: {e}")
