"""Companion cycle — one ideation pass through Creative MAS.

Composes context from WorkspaceKB + the workspace seed prompt, runs the
3-phase Creative MAS pipeline (Initiation / Discussion / Convergence),
persists the lineage (fragments → developed → converged) to the idea
store, scores the converged output (novelty + quality + transferability),
and returns a CycleResult.

Scores are computed against the workspace's PRIOR history, before any of
this cycle's outputs are persisted, so an idea is never compared against
its own siblings. Persistence is best-effort end-to-end: failures of
ChromaDB / scoring LLM degrade gracefully without crashing the cycle.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.companion import idea_store as _idea_store
from app.companion import reflexion as _reflexion
from app.companion import scoring as _scoring
from app.companion import surfacing as _surfacing
from app.companion import workspace_kb
from app.companion.config import CompanionConfig

if TYPE_CHECKING:
    from app.crews.creative_crew import CreativeRunResult

logger = logging.getLogger(__name__)


@dataclass
class CycleResult:
    """Outcome of one cycle."""
    workspace_id: str
    cycle_id: str = ""
    phase_1_count: int = 0
    phase_2_count: int = 0
    final_output: str = ""
    final_output_chars: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    aborted_reason: str | None = None
    creative_scores: dict | None = None
    # Phase 3 persistence outcomes
    converged_idea_id: str | None = None
    fragment_ids: list[str] = field(default_factory=list)
    developed_ids: list[str] = field(default_factory=list)
    novelty: float = 0.0
    quality: float = 0.0
    transferability: float = 0.0
    # Phase 4 surfacing
    surfaced: bool = False
    surface_reason: str = ""


def run_cycle(workspace_id: str, config: CompanionConfig) -> CycleResult:
    """Execute one Companion cycle for the given workspace.

    Returns a CycleResult with ``aborted_reason="no_seed_prompt"`` if the
    workspace has no seed and no synthesised grand task yet (grand-task
    synthesis lands in Phase 11). Failures inside Creative MAS are caught
    and surfaced via ``aborted_reason``; the caller still records a tick
    so the fairness scheduler advances.
    """
    started = time.monotonic()
    cycle_id = f"cyc_{uuid.uuid4().hex[:12]}"
    seed = (config.seed_prompt or "").strip()
    if not seed:
        return CycleResult(
            workspace_id=workspace_id,
            cycle_id=cycle_id,
            aborted_reason="no_seed_prompt",
            duration_s=time.monotonic() - started,
        )

    snippets = workspace_kb.compose(
        workspace_id=workspace_id,
        query=seed,
        top_k=workspace_kb.DEFAULT_KB_TOP_K,
    )
    prompt = _compose_prompt(seed, snippets, workspace_id=workspace_id)

    try:
        result: "CreativeRunResult" = _invoke_creative_crew(prompt)
    except Exception as exc:
        logger.warning("companion.cycle: creative_crew failed for %s: %s",
                       workspace_id, exc)
        return CycleResult(
            workspace_id=workspace_id,
            cycle_id=cycle_id,
            aborted_reason=f"creative_crew_failed:{type(exc).__name__}",
            duration_s=time.monotonic() - started,
        )

    final = getattr(result, "final_output", "") or ""
    aborted = getattr(result, "aborted_reason", None)
    phase_1 = list(getattr(result, "phase_1_outputs", []) or [])
    phase_2 = list(getattr(result, "phase_2_outputs", []) or [])

    # Persist + score only on successful completion with non-empty output.
    fragment_ids: list[str] = []
    developed_ids: list[str] = []
    converged_id: str | None = None
    novelty = quality = transferability = 0.0

    surfaced = False
    surface_reason = "not_attempted"

    if aborted is None and final.strip():
        novelty = _scoring.compute_novelty(final, workspace_id)
        quality = _scoring.compute_quality(final)
        transferability = _scoring.compute_transferability(final)
        fragment_ids, developed_ids, converged_id = _persist_lineage(
            workspace_id=workspace_id,
            cycle_id=cycle_id,
            phase_1=phase_1,
            phase_2=phase_2,
            final=final,
            novelty=novelty,
            quality=quality,
            transferability=transferability,
        )
        if converged_id:
            surfaced, surface_reason = _maybe_surface(
                workspace_id=workspace_id,
                idea_id=converged_id,
                final=final,
                novelty=novelty,
                quality=quality,
                transferability=transferability,
                config=config,
            )

    return CycleResult(
        workspace_id=workspace_id,
        cycle_id=cycle_id,
        phase_1_count=len(phase_1),
        phase_2_count=len(phase_2),
        final_output=final,
        final_output_chars=len(final),
        cost_usd=float(getattr(result, "cost_usd", 0.0) or 0.0),
        duration_s=time.monotonic() - started,
        aborted_reason=aborted,
        creative_scores=getattr(result, "scores", None),
        converged_idea_id=converged_id,
        fragment_ids=fragment_ids,
        developed_ids=developed_ids,
        novelty=novelty,
        quality=quality,
        transferability=transferability,
        surfaced=surfaced,
        surface_reason=surface_reason,
    )


def _maybe_surface(*, workspace_id: str, idea_id: str, final: str,
                    novelty: float, quality: float, transferability: float,
                    config: CompanionConfig) -> tuple[bool, str]:
    """Build a transient IdeaRecord and run the surfacing pipeline."""
    rec = _idea_store.IdeaRecord(
        idea_id=idea_id,
        workspace_id=workspace_id,
        text=final,
        state=_idea_store.IdeaState.CONVERGED,
        novelty=novelty,
        quality=quality,
        transferability=transferability,
    )
    try:
        decision = _surfacing.should_surface(rec, config)
    except Exception as exc:
        logger.debug("companion.cycle: should_surface raised: %s", exc)
        return False, "decision_failed"
    if not decision.eligible:
        return False, decision.reason
    try:
        sent = _surfacing.surface(rec, config)
    except Exception as exc:
        logger.warning("companion.cycle: surface raised: %s", exc)
        return False, f"surface_failed:{type(exc).__name__}"
    return sent, "ok" if sent else "send_failed"


def _persist_lineage(
    *, workspace_id: str, cycle_id: str,
    phase_1, phase_2, final: str,
    novelty: float, quality: float, transferability: float,
) -> tuple[list[str], list[str], str | None]:
    """Persist fragments → developed → converged with parent edges.

    Each phase 1 output is a fragment with no parents; each phase 2 output
    is developed and lists ALL phase 1 ids as parents (the discussion sees
    every initiation output as context); the converged output lists all
    phase 2 ids as parents. Returns ``(fragment_ids, developed_ids,
    converged_id)``. Persistence failures per-record are logged and absorbed.
    """
    fragment_ids: list[str] = []
    for o in phase_1:
        text = (getattr(o, "text", "") or "").strip()
        if not text:
            continue
        rec = _idea_store.IdeaRecord(
            workspace_id=workspace_id, cycle_id=cycle_id,
            text=text, role=getattr(o, "role", ""),
            state=_idea_store.IdeaState.FRAGMENT,
        )
        try:
            _idea_store.persist(rec)
            fragment_ids.append(rec.idea_id)
        except Exception as exc:
            logger.debug("companion.cycle: fragment persist failed: %s", exc)

    developed_ids: list[str] = []
    for o in phase_2:
        text = (getattr(o, "text", "") or "").strip()
        if not text:
            continue
        rec = _idea_store.IdeaRecord(
            workspace_id=workspace_id, cycle_id=cycle_id,
            text=text, role=getattr(o, "role", ""),
            state=_idea_store.IdeaState.DEVELOPED,
            lineage_parents=list(fragment_ids),
        )
        try:
            _idea_store.persist(rec)
            developed_ids.append(rec.idea_id)
        except Exception as exc:
            logger.debug("companion.cycle: developed persist failed: %s", exc)

    converged_id: str | None = None
    converged = _idea_store.IdeaRecord(
        workspace_id=workspace_id, cycle_id=cycle_id,
        text=final, role="commander (converge)",
        state=_idea_store.IdeaState.CONVERGED,
        lineage_parents=list(developed_ids) or list(fragment_ids),
        novelty=novelty, quality=quality, transferability=transferability,
    )
    try:
        converged_id = _idea_store.persist(converged)
    except Exception as exc:
        logger.warning("companion.cycle: converged persist failed: %s", exc)

    return fragment_ids, developed_ids, converged_id


def _invoke_creative_crew(task_description: str):
    """Indirection over ``app.crews.creative_crew.run_creative_crew``.

    Keeps the cycle testable without dragging the full CrewAI / LLM stack
    into the test environment.
    """
    from app.crews.creative_crew import run_creative_crew
    return run_creative_crew(task_description=task_description, creativity="high")


def _compose_prompt(seed: str, snippets: list[workspace_kb.KBSnippet],
                    *, workspace_id: str | None = None) -> str:
    """Assemble the prompt fed to Creative MAS phase 1."""
    lines: list[str] = [
        "You are exploring an open-ended idea space for a workspace.",
        "",
        f"## Workspace seed\n{seed}",
        "",
    ]
    body_snippets = [s for s in snippets if s.text]
    if body_snippets:
        lines.append("## Context")
        for s in body_snippets:
            lines.append(s.to_prompt_line())
        lines.append("")
    if workspace_id:
        try:
            block = _reflexion.build_block(workspace_id)
        except Exception as exc:
            logger.debug("companion.cycle: reflexion.build_block raised: %s",
                         exc)
            block = ""
        if block:
            lines.append(block)
    lines.append(
        "## Task\n"
        "Generate fresh, surprising ideas that bear on the workspace seed. "
        "Lateral thinking, analogies, and conceptual blends are welcome. "
        "Stay on the workspace's topic — do not wander into unrelated domains."
    )
    return "\n".join(lines)
