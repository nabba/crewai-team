"""Inline ShinkaEvolve runs scoped to a single coding session.

Where :func:`app.shinka_engine.run_shinka_session` is the *bulk*
entry point — runs against the fixed ``INITIAL_PY`` / ``EVALUATE_PY``
constants on every self-improvement cycle — this bridge is the
*coder-agent inline* entry point: an agent inside an ACTIVE coding
session can ask "evolve this single file against this single
evaluator and give me back the best diff." The diff flows out of
the session through the existing :mod:`app.change_requests` gate
on submit; nothing lands in the repo without operator approval.

Hard caps (always enforced; callers can request smaller, never larger):

  ``MAX_GENERATIONS_INLINE``   20    one-shot tool, not a bulk cycle
  ``MAX_ISLANDS_INLINE``        3
  ``MAX_COST_USD_INLINE``       5.0  per-run dollar cap on LLM proposals

Refusals (request never reaches ShinkaEvolveRunner):

  - Session not found OR session not ACTIVE
  - Either path resolves outside the session's worktree
  - Either path doesn't exist in the worktree
  - The repo-relative ``initial_path`` is in TIER_IMMUTABLE OR under
    ``app/subia/`` OR is the Tier-3 anchor ``app/affect/goal_emitter.py``
    (uses :func:`app.architecture_requests.validator.is_protected_path`
    so the same protection vocabulary is enforced everywhere).

Output:

  ``EvolutionResult.diff`` is a unified diff vs the initial program
  *only when an improvement was found*. The bridge never applies
  the diff — that's the coder agent's choice via ``coding_session_submit``.

The actual ShinkaEvolveRunner invocation is behind an injectable
``runner_factory`` so tests can drive the orchestration without
needing the shinka package or LLM credentials.
"""
from __future__ import annotations

import difflib
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.architecture_requests.validator import is_protected_path
from app.coding_session.models import Status

logger = logging.getLogger(__name__)


MAX_GENERATIONS_INLINE = 20
MAX_ISLANDS_INLINE = 3
MAX_COST_USD_INLINE = 5.0


@dataclass(frozen=True)
class RunnerOutput:
    """Normalised output of one ShinkaEvolveRunner run."""

    best_score: float
    baseline_score: float
    best_program_path: Path | None  # None if evolution found no improvement
    generations_run: int
    variants_evaluated: int
    error: str = ""


@dataclass(frozen=True)
class EvolutionResult:
    """The bridge's result for an inline evolution call."""

    status: str
    # status ∈ {"improved", "no_improvement", "error",
    #          "shinka_unavailable", "refused", "disabled"}
    # ``disabled`` (Q7.4) means the operator turned the inline-evolve
    # master switch OFF in runtime_settings.
    baseline_score: float = 0.0
    best_score: float = 0.0
    delta: float = 0.0
    diff: str = ""
    generations_run: int = 0
    variants_evaluated: int = 0
    duration_seconds: float = 0.0
    error: str = ""
    refusal_reason: str = ""


RunnerFactory = Callable[..., RunnerOutput]


def evolve_in_session(
    *,
    session_id: str,
    initial_path: str,
    evaluate_path: str,
    num_generations: int = 5,
    num_islands: int = 1,
    max_cost_usd: float = 1.5,
    runner_factory: RunnerFactory | None = None,
    manager: object | None = None,
) -> EvolutionResult:
    """Run an inline ShinkaEvolve evolution scoped to one coding session.

    See module docstring for the refusal vocabulary, hard caps, and
    output discipline.

    The ``manager`` and ``runner_factory`` parameters are dependency-
    injection seams for tests. Production callers leave them ``None``;
    the bridge resolves the global manager and the real shinka runner.

    Every call (including refused / disabled ones) appends a row to
    the per-session evolution audit JSONL via
    :func:`app.coding_session.evolution_audit.append_run` — that
    audit survives worktree cleanup, so the operator can see what
    the agent attempted even after the session terminates.
    """
    start = time.monotonic()

    num_generations = max(1, min(num_generations, MAX_GENERATIONS_INLINE))
    num_islands = max(1, min(num_islands, MAX_ISLANDS_INLINE))
    max_cost_usd = max(0.01, min(max_cost_usd, MAX_COST_USD_INLINE))

    # Master switch (Q7.4 — PROGRAM §45.4). When OFF the operator has
    # deliberately disabled inline evolution; refuse before validating
    # paths or invoking the runner.
    if not _inline_evolve_enabled():
        result = EvolutionResult(
            status="disabled",
            refusal_reason=(
                "shinka_inline_evolve_enabled is OFF in runtime_settings; "
                "the operator disabled inline coding-session evolution. "
                "Flip the switch in /cp/settings to re-enable."
            ),
            duration_seconds=time.monotonic() - start,
        )
        _audit_run_safely(
            session_id=session_id,
            initial_path=initial_path,
            evaluate_path=evaluate_path,
            num_generations=num_generations,
            num_islands=num_islands,
            max_cost_usd=max_cost_usd,
            result=result,
        )
        return result

    refusal = _validate_request(session_id, initial_path, evaluate_path, manager)
    if refusal is not None:
        result = EvolutionResult(
            status="refused",
            refusal_reason=refusal.reason,
            duration_seconds=time.monotonic() - start,
        )
        _audit_run_safely(
            session_id=session_id,
            initial_path=initial_path,
            evaluate_path=evaluate_path,
            num_generations=num_generations,
            num_islands=num_islands,
            max_cost_usd=max_cost_usd,
            result=result,
        )
        return result
    session = _get_session(session_id, manager)
    assert session is not None  # _validate_request would have refused

    worktree = Path(session.worktree_path).resolve()
    initial = (worktree / initial_path).resolve()
    evaluate = (worktree / evaluate_path).resolve()

    results_dir = (
        worktree / ".shinka_inline"
        / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    factory = runner_factory or _default_runner_factory
    output = factory(
        initial_path=initial,
        evaluate_path=evaluate,
        results_dir=results_dir,
        num_generations=num_generations,
        num_islands=num_islands,
        max_cost_usd=max_cost_usd,
    )

    if output.error:
        status = (
            "shinka_unavailable"
            if "shinka" in output.error.lower() and "not" in output.error.lower()
            else "error"
        )
        result = EvolutionResult(
            status=status,
            error=output.error,
            duration_seconds=time.monotonic() - start,
        )
        _audit_run_safely(
            session_id=session_id,
            initial_path=initial_path,
            evaluate_path=evaluate_path,
            num_generations=num_generations,
            num_islands=num_islands,
            max_cost_usd=max_cost_usd,
            result=result,
        )
        return result

    delta = output.best_score - output.baseline_score
    if delta <= 0 or output.best_program_path is None:
        result = EvolutionResult(
            status="no_improvement",
            baseline_score=output.baseline_score,
            best_score=output.best_score,
            delta=delta,
            generations_run=output.generations_run,
            variants_evaluated=output.variants_evaluated,
            duration_seconds=time.monotonic() - start,
        )
        _audit_run_safely(
            session_id=session_id,
            initial_path=initial_path,
            evaluate_path=evaluate_path,
            num_generations=num_generations,
            num_islands=num_islands,
            max_cost_usd=max_cost_usd,
            result=result,
        )
        return result

    diff_text = _compute_diff(initial, Path(output.best_program_path), initial_path)
    result = EvolutionResult(
        status="improved",
        baseline_score=output.baseline_score,
        best_score=output.best_score,
        delta=delta,
        diff=diff_text,
        generations_run=output.generations_run,
        variants_evaluated=output.variants_evaluated,
        duration_seconds=time.monotonic() - start,
    )
    _audit_run_safely(
        session_id=session_id,
        initial_path=initial_path,
        evaluate_path=evaluate_path,
        num_generations=num_generations,
        num_islands=num_islands,
        max_cost_usd=max_cost_usd,
        result=result,
    )
    return result


def _inline_evolve_enabled() -> bool:
    """Q7.4 master switch read. Fail-open (True) if runtime_settings is
    unavailable — the bridge has its own refusal layers for safety,
    so the switch is a deliberate operator OFF, not a default-OFF."""
    try:
        from app.runtime_settings import get_shinka_inline_evolve_enabled
        return bool(get_shinka_inline_evolve_enabled())
    except Exception:
        return True


def _audit_run_safely(
    *,
    session_id: str,
    initial_path: str,
    evaluate_path: str,
    num_generations: int,
    num_islands: int,
    max_cost_usd: float,
    result: EvolutionResult,
) -> None:
    """Append the run to the per-session evolution audit JSONL.

    Failure-isolated by design — the audit is operator-visibility
    sugar, not a correctness requirement. The bridge keeps running
    even when the audit can't be written.
    """
    try:
        from app.coding_session.evolution_audit import append_run
        append_run(
            session_id=session_id,
            agent_id=_resolve_agent_id_for_audit(),
            initial_path=initial_path,
            evaluate_path=evaluate_path,
            num_generations=num_generations,
            num_islands=num_islands,
            max_cost_usd=max_cost_usd,
            result=result,
        )
    except Exception:
        logger.debug(
            "evolve_in_session: audit append failed", exc_info=True,
        )


def _resolve_agent_id_for_audit() -> str:
    """Best-effort agent-id resolution for the audit row. The bridge
    itself doesn't have an agent-id parameter (the existing tool layer
    in :mod:`app.tools.coding_session_tools` does the resolution); fall
    back to the env var the tool layer sets.
    """
    return os.environ.get("BOTARMY_CURRENT_AGENT_ID", "coder")


# ── Internals ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Refusal:
    reason: str
    session: object | None = None


def _validate_request(
    session_id: str,
    initial_path: str,
    evaluate_path: str,
    manager: object | None,
) -> _Refusal | None:
    """Run all refusal checks; return None if the request is admissible."""
    session = _get_session(session_id, manager)
    if session is None:
        return _Refusal(reason=f"session {session_id!r} not found")
    if session.status is not Status.ACTIVE:
        return _Refusal(
            reason=(
                f"session {session_id!r} is not ACTIVE "
                f"(status={session.status.value if hasattr(session.status, 'value') else session.status})"
            ),
            session=session,
        )

    worktree = Path(session.worktree_path).resolve()
    if not worktree.exists():
        return _Refusal(reason=f"session worktree {worktree} does not exist", session=session)

    for label, p in (("initial_path", initial_path), ("evaluate_path", evaluate_path)):
        if p.startswith("/") or ".." in p.split("/"):
            return _Refusal(
                reason=f"{label} {p!r} must be relative to the worktree without traversal",
                session=session,
            )
        resolved = (worktree / p).resolve()
        if worktree != resolved and worktree not in resolved.parents:
            return _Refusal(
                reason=f"{label} {p!r} resolves outside the session worktree",
                session=session,
            )
        if not resolved.exists():
            return _Refusal(
                reason=f"{label} {p!r} does not exist in worktree",
                session=session,
            )

    repo_relative = initial_path.lstrip("/")
    if is_protected_path(repo_relative):
        return _Refusal(
            reason=(
                f"initial_path {repo_relative!r} is protected "
                f"(TIER_IMMUTABLE or consciousness layer); evolution "
                f"refused — modifying it requires Tier-3 amendment, "
                f"not a coding-session diff"
            ),
            session=session,
        )

    return None


def _get_session(session_id: str, manager: object | None) -> object | None:
    if manager is None:
        from app.coding_session.runtime import get_manager
        manager = get_manager()
    try:
        return manager.get(session_id)  # type: ignore[attr-defined]
    except KeyError:
        return None


def _compute_diff(initial: Path, best: Path, repo_relative: str) -> str:
    try:
        initial_text = initial.read_text()
    except OSError:
        initial_text = ""
    try:
        best_text = best.read_text()
    except OSError:
        best_text = ""
    return "".join(difflib.unified_diff(
        initial_text.splitlines(keepends=True),
        best_text.splitlines(keepends=True),
        fromfile=f"a/{repo_relative}",
        tofile=f"b/{repo_relative}",
        n=3,
    ))


def _default_runner_factory(
    *,
    initial_path: Path,
    evaluate_path: Path,
    results_dir: Path,
    num_generations: int,
    num_islands: int,
    max_cost_usd: float,
) -> RunnerOutput:
    """Real ShinkaEvolveRunner invocation. Tests inject a fake."""
    try:
        from shinka.core import ShinkaEvolveRunner, EvolutionConfig
        from shinka.launch import LocalJobConfig
        from shinka.database import DatabaseConfig
    except ImportError as exc:  # pragma: no cover — env-dependent
        return RunnerOutput(
            best_score=0.0, baseline_score=0.0, best_program_path=None,
            generations_run=0, variants_evaluated=0,
            error=f"shinka not installed: {exc}",
        )

    evo_config = EvolutionConfig(
        num_generations=num_generations,
        init_program_path=str(initial_path),
        results_dir=str(results_dir),
        language="python",
        max_api_costs=max_cost_usd,
    )
    job_config = LocalJobConfig(eval_program_path=str(evaluate_path))
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
        verbose=False,
    )

    try:
        runner.run()
    except Exception as exc:  # pragma: no cover — runtime-dependent
        return RunnerOutput(
            best_score=0.0, baseline_score=0.0, best_program_path=None,
            generations_run=0, variants_evaluated=0,
            error=f"shinka runner crashed: {exc}",
        )

    best_score, best_path = _extract_best(results_dir)
    baseline_score = _read_baseline(results_dir)
    return RunnerOutput(
        best_score=best_score,
        baseline_score=baseline_score,
        best_program_path=best_path,
        generations_run=num_generations,  # shinka doesn't expose actual count
        variants_evaluated=0,  # ditto
        error="",
    )


def _extract_best(results_dir: Path) -> tuple[float, Path | None]:
    """Read the best variant's score + path from the shinka results dir.

    Mirrors ``app.shinka_engine._extract_best_result`` semantics
    without importing it (that module is TIER_IMMUTABLE).
    """
    best_score = 0.0
    best_path: Path | None = None
    for variant in results_dir.glob("**/program.py"):
        score_file = variant.parent / "score.txt"
        if not score_file.exists():
            continue
        try:
            score = float(score_file.read_text().strip())
        except (ValueError, OSError):
            continue
        if score > best_score:
            best_score = score
            best_path = variant
    return best_score, best_path


def _read_baseline(results_dir: Path) -> float:
    baseline_file = results_dir / "baseline_score.txt"
    if not baseline_file.exists():
        return 0.0
    try:
        return float(baseline_file.read_text().strip())
    except (ValueError, OSError):
        return 0.0
