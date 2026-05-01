"""Companion cycle — one ideation pass through Creative MAS.

Composes context from WorkspaceKB + the workspace seed prompt, runs the
3-phase Creative MAS pipeline (Initiation / Discussion / Convergence),
and returns a CycleResult. Phase 2: fragments are returned in-memory and
logged. Phase 3 adds persistence + scoring.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.companion import workspace_kb
from app.companion.config import CompanionConfig

if TYPE_CHECKING:
    from app.crews.creative_crew import CreativeRunResult

logger = logging.getLogger(__name__)


@dataclass
class CycleResult:
    """Outcome of one cycle. Phase 3 will extend with idea_ids."""
    workspace_id: str
    phase_1_count: int = 0
    phase_2_count: int = 0
    final_output: str = ""
    final_output_chars: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    aborted_reason: str | None = None
    creative_scores: dict | None = None


def run_cycle(workspace_id: str, config: CompanionConfig) -> CycleResult:
    """Execute one Companion cycle for the given workspace.

    Returns a CycleResult with ``aborted_reason="no_seed_prompt"`` if the
    workspace has no seed and no synthesised grand task yet (grand-task
    synthesis lands in Phase 11). Failures inside Creative MAS are caught
    and surfaced via ``aborted_reason``; the caller still records a tick
    so the fairness scheduler advances.
    """
    started = time.monotonic()
    seed = (config.seed_prompt or "").strip()
    if not seed:
        return CycleResult(
            workspace_id=workspace_id,
            aborted_reason="no_seed_prompt",
            duration_s=time.monotonic() - started,
        )

    snippets = workspace_kb.compose(
        workspace_id=workspace_id,
        query=seed,
        top_k=workspace_kb.DEFAULT_KB_TOP_K,
    )
    prompt = _compose_prompt(seed, snippets)

    try:
        result: "CreativeRunResult" = _invoke_creative_crew(prompt)
    except Exception as exc:
        logger.warning("companion.cycle: creative_crew failed for %s: %s",
                       workspace_id, exc)
        return CycleResult(
            workspace_id=workspace_id,
            aborted_reason=f"creative_crew_failed:{type(exc).__name__}",
            duration_s=time.monotonic() - started,
        )

    final = getattr(result, "final_output", "") or ""
    return CycleResult(
        workspace_id=workspace_id,
        phase_1_count=len(getattr(result, "phase_1_outputs", []) or []),
        phase_2_count=len(getattr(result, "phase_2_outputs", []) or []),
        final_output=final,
        final_output_chars=len(final),
        cost_usd=float(getattr(result, "cost_usd", 0.0) or 0.0),
        duration_s=time.monotonic() - started,
        aborted_reason=getattr(result, "aborted_reason", None),
        creative_scores=getattr(result, "scores", None),
    )


def _invoke_creative_crew(task_description: str):
    """Indirection over ``app.crews.creative_crew.run_creative_crew``.

    Keeps the cycle testable without dragging the full CrewAI / LLM stack
    into the test environment.
    """
    from app.crews.creative_crew import run_creative_crew
    return run_creative_crew(task_description=task_description, creativity="high")


def _compose_prompt(seed: str, snippets: list[workspace_kb.KBSnippet]) -> str:
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
    lines.append(
        "## Task\n"
        "Generate fresh, surprising ideas that bear on the workspace seed. "
        "Lateral thinking, analogies, and conceptual blends are welcome. "
        "Stay on the workspace's topic — do not wander into unrelated domains."
    )
    return "\n".join(lines)
