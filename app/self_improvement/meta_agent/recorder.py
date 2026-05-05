"""
app.self_improvement.meta_agent.recorder — outcome capture.

The recorder is called from base_crew.run_single_agent_crew right after
the lifecycle envelope finishes. It writes a RecipeOutcome row, bumps
the recipe's denormalised counters, and (when the recipe was new and
this is its first observation) seeds the recipe's task_signature so
future similarity searches can find it.

Design constraints:
    - Best-effort. Outcome recording must NEVER raise into the
      run_single_agent_crew path; a meta-agent failure cannot break
      the user's task.
    - No new persistence layer. The recorder reuses store.record_outcome
      (Postgres) and store.upsert_recipe (Postgres + Chroma).
    - Confidence is read from the existing app.human_gate.classify_confidence
      result when available, else "" (no synthesis here).

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import logging
import uuid

from app.self_improvement.meta_agent.types import (
    AgentRecipe, RecipeOutcome, RecipeSelection,
)
from app.self_improvement.meta_agent.store import record_outcome as store_outcome
from app.self_improvement.meta_agent.store import upsert_recipe, get_recipe

logger = logging.getLogger(__name__)


def record_outcome(
    *,
    selection: RecipeSelection,
    crew_name: str,
    task_id: str,
    task_description: str,
    success: bool,
    duration_s: float = 0.0,
    confidence: str = "",
    error_signature: str = "",
    user_feedback: str = "",
    cost_estimate: float = 0.0,
) -> bool:
    """Persist an outcome for a single dispatch.

    Idempotent on (task_id, recipe_id) — re-recording the same outcome
    (e.g. on a retry) is a no-op at the DB level via the outcome row's
    ON CONFLICT DO NOTHING.

    Args:
        selection: Whatever the selector returned.
        crew_name: e.g. "coding".
        task_id: The crew_lifecycle's task_id (firebase id) — used as
            the join key for cross-system audit trails.
        task_description: For seeding the recipe.task_signature on
            first observation. Truncated server-side.
        success: True iff the crew finished without raising.
        duration_s: Wall-clock seconds of the dispatch.
        confidence: Optional ConfidenceTier value from human_gate.
        error_signature: For failure clustering — typically the
            exception type name. Empty on success.
        user_feedback: "👍" / "👎" / "" if the React reactor surfaced
            an explicit reaction.
        cost_estimate: Token-based cost estimate (advisory).

    Returns:
        True iff the outcome row was persisted.
    """
    recipe = selection.chosen
    # Lazy-resolve recipe.task_signature on first observation. The
    # selector matches future tasks against this string, so seeding
    # it once a recipe has actually been used is the cleanest way to
    # propagate "this recipe is good for tasks like X" without a
    # separate migration step.
    if not recipe.is_null:
        try:
            persisted = get_recipe(recipe.id)
        except Exception:
            persisted = None
        if persisted is not None and not persisted.task_signature and task_description:
            persisted.task_signature = _truncate_signature(task_description)
            try:
                upsert_recipe(persisted)
                # Refresh the in-memory copy so similarity_search picks
                # up the seeded signature on the very next dispatch.
                recipe.task_signature = persisted.task_signature
            except Exception:
                logger.debug(
                    "meta_agent.recorder: signature seed upsert failed",
                    exc_info=True,
                )

    outcome = RecipeOutcome(
        id=f"out_{uuid.uuid4().hex[:14]}",
        recipe_id=recipe.id,
        crew_name=crew_name,
        task_id=task_id or "",
        success=bool(success),
        confidence=confidence or "",
        duration_s=float(duration_s),
        cost_estimate=float(cost_estimate),
        error_signature=error_signature or "",
        user_feedback=user_feedback or "",
        task_signature=_truncate_signature(task_description),
    )

    try:
        ok = store_outcome(outcome)
    except Exception as exc:
        logger.debug(f"meta_agent.recorder: store_outcome raised: {exc}",
                     exc_info=True)
        return False

    if ok:
        logger.debug(
            "meta_agent.recorder: %s recipe=%s task=%s success=%s "
            "(%.1fs, sim=%.2f, succ=%.2f, %s)",
            crew_name, recipe.id, task_id, success,
            duration_s, selection.similarity, selection.smoothed_success_rate,
            "explore" if selection.explored else "exploit",
        )
    return ok


# ── Helpers ──────────────────────────────────────────────────────────────────

_MAX_SIGNATURE_LEN = 600  # ChromaDB document length cap is generous; ours is
                          # tighter to keep recipe rows small in Postgres.


def _truncate_signature(text: str) -> str:
    """Trim a task description down to the size we store as a signature.

    Stripped + collapsed whitespace — preserves the semantic content
    Chroma's embedding model cares about while keeping the recipe row
    a predictable size.
    """
    if not text:
        return ""
    norm = " ".join(text.split())
    return norm[:_MAX_SIGNATURE_LEN]
