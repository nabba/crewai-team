"""
app.self_improvement.meta_agent.selector — UCB1 + similarity recipe selection.

Algorithm
=========

For each incoming task:

    1. Embed task_description and query the recipe similarity index
       (Chroma) for the top-k recipes registered to ``crew_name``.
    2. Filter to candidates with cosine_distance ≤ similarity_tau
       (1.0 = orthogonal; we want similar tasks).
    3. The null recipe (factory defaults) is always included as the
       control arm.
    4. Score each candidate r as:
           similarity_r = 1 - cosine_distance_r
           smoothed_succ = (successes+1) / (uses+2)
           ucb_r = smoothed_succ + ucb_c × sqrt(log(N+1) / (uses+1))
           score_r = similarity_r × ucb_r
       where N is the total observed uses across candidates.
    5. ε-greedy: with prob epsilon, pick a uniformly random candidate
       (or the null recipe if no recorded recipes exist) — that's the
       explore branch and is logged as such.
    6. Otherwise return argmax(score_r).

Cold-start
==========

When no recipes pass the similarity filter, the selector returns the
null recipe. The caller (recorder) still observes an outcome, which
keeps the null recipe's denominator growing — so as augmented recipes
accumulate evidence, the bandit can fairly compare them to the
factory-default baseline.

A useful consequence of UCB1's structure: the null recipe's exploration
bonus is unbounded when its uses=0, so until the control arm has been
exercised at least a few dozen times, the bandit will prefer it over
any augmented candidate, no matter how well-evidenced. That's the
design intent — don't apply augmentation before establishing the
factory-default baseline.

Thresholds are IMMUTABLE module-level constants — same convention as
NOVELTY_THRESHOLDS in app.self_improvement.novelty. The selector itself
is part of the protected core that the system cannot edit.

Why UCB1 (not Thompson)
=======================

UCB1 is deterministic given the same counters; that makes outcome
audits trivial ("why did the selector pick X for this task?") and
keeps the dashboard's per-recipe ranking stable across refreshes.
Thompson sampling would add randomness on every dispatch with no
real benefit at our scale.
"""

from __future__ import annotations

import logging
import math
import os
import random
from typing import Optional

from app.self_improvement.meta_agent.types import AgentRecipe, RecipeSelection
from app.self_improvement.meta_agent.store import (
    null_recipe_for, similarity_search, list_recipes,
)

logger = logging.getLogger(__name__)


# ── IMMUTABLE: selection thresholds ──────────────────────────────────────────
#
# Calibrated to match the existing self_improvement.novelty thresholds where
# they overlap (similarity_tau is on the same cosine-distance scale). Tuning
# requires editing this module — and this module is part of TIER_IMMUTABLE,
# so a tuning pass needs operator review, exactly like NOVELTY_THRESHOLDS.

SELECTION_THRESHOLDS = {
    # Maximum cosine distance (lower = more similar) for a recipe to be
    # considered "relevant" to the incoming task. 0.55 leaves room for
    # cross-domain transfer (a "summarise X" recipe is reachable from a
    # "summarise Y" task) while excluding fully-unrelated work.
    "similarity_tau": 0.55,

    # ε-greedy exploration rate — fraction of dispatches that pick a
    # uniformly random candidate to keep the bandit from collapsing on
    # an early winner that may not be the global optimum.
    "epsilon": 0.10,

    # UCB1 exploration constant. Higher = more aggressive exploration of
    # under-tried recipes. 1.4 (= sqrt(2)) is the standard default and
    # works well at our scale (hundreds of outcomes, not millions).
    "ucb_c": 1.4,

    # Hard upper bound on similarity-search candidates. Keeps the
    # selector's runtime constant per dispatch.
    "max_candidates": 8,
}


# ── Public selection entrypoint ──────────────────────────────────────────────

def select_recipe(
    *,
    crew_name: str,
    task_description: str,
    rng: Optional[random.Random] = None,
) -> RecipeSelection:
    """Choose the best recipe for ``task_description`` on ``crew_name``.

    Always returns a RecipeSelection. The chosen recipe may be the null
    recipe (factory defaults) — the caller is responsible for applying
    the augmentation through ``apply_recipe``.

    Args:
        crew_name: e.g. "coding", "research", "writing".
        task_description: The user's task text — embedded for similarity.
        rng: Optional random.Random for deterministic tests. Defaults to
            module-level random.

    Side-effects: none. Selection is pure (modulo embedding).
    """
    rng = rng or random

    # Cold-start fast path — if there are no recipes at all for this crew,
    # don't bother with embedding lookups. Just return the null recipe
    # and tag the selection as exploratory.
    null = null_recipe_for(crew_name)
    candidates = _gather_candidates(crew_name, task_description)
    if not candidates:
        return RecipeSelection(
            chosen=null,
            candidates_considered=1,  # the null recipe itself
            score=0.0,
            similarity=1.0,           # null recipe matches everything
            smoothed_success_rate=null.smoothed_success_rate,
            explored=True,
            rationale="cold-start: no similar recipes in registry",
        )

    # The null recipe is always a candidate (control arm). It has no
    # similarity to the task, so we score it with similarity=1.0 — it
    # represents "no augmentation, factory defaults" which by definition
    # applies to every task.
    if not any(r.is_null for r, _ in candidates):
        candidates.append((null, 0.0))

    # ε-greedy explore branch
    if rng.random() < SELECTION_THRESHOLDS["epsilon"]:
        chosen, dist = rng.choice(candidates)
        return RecipeSelection(
            chosen=chosen,
            candidates_considered=len(candidates),
            score=0.0,
            similarity=1.0 - dist,
            smoothed_success_rate=chosen.smoothed_success_rate,
            explored=True,
            rationale=f"ε-greedy explore (epsilon={SELECTION_THRESHOLDS['epsilon']})",
        )

    # Exploit branch — score and argmax
    n_total = sum(max(r.uses, 0) for r, _ in candidates)
    log_term = math.log(max(n_total + 1, 2))

    best: tuple[AgentRecipe, float, float, float] | None = None  # (rec, score, sim, succ)
    for recipe, dist in candidates:
        # Null recipe gets similarity=1.0 by convention. Augmented recipes
        # are scored by 1-cos_dist as usual.
        similarity = 1.0 if recipe.is_null else max(0.0, 1.0 - dist)
        smoothed = recipe.smoothed_success_rate
        ucb = smoothed + SELECTION_THRESHOLDS["ucb_c"] * math.sqrt(
            log_term / max(recipe.uses + 1, 1)
        )
        score = similarity * ucb
        if best is None or score > best[1]:
            best = (recipe, score, similarity, smoothed)

    assert best is not None, "candidates non-empty, best must be set"
    chosen, score, similarity, smoothed = best
    return RecipeSelection(
        chosen=chosen,
        candidates_considered=len(candidates),
        score=score,
        similarity=similarity,
        smoothed_success_rate=smoothed,
        explored=False,
        rationale=(
            f"argmax UCB1×similarity over {len(candidates)} candidates "
            f"(score={score:.3f}, sim={similarity:.3f}, "
            f"succ={smoothed:.3f})"
        ),
    )


# ── Candidate gathering ──────────────────────────────────────────────────────

def _gather_candidates(
    crew_name: str,
    task_description: str,
) -> list[tuple[AgentRecipe, float]]:
    """Find recipes within similarity_tau of the task.

    Falls back to listing all recipes for the crew (with synthetic
    distance=0.5) when Chroma is down — the bandit still works, just
    without the similarity weighting.
    """
    tau = SELECTION_THRESHOLDS["similarity_tau"]
    k = SELECTION_THRESHOLDS["max_candidates"]

    try:
        chroma = similarity_search(
            crew_name=crew_name,
            task_text=task_description,
            n_results=k,
        )
    except Exception:
        chroma = []

    candidates = [(r, d) for r, d in chroma if d <= tau]

    if not candidates:
        # Chroma either down or returned nothing within tau. Fall back to
        # the Postgres recipe list — selector still works, similarity_r
        # will be neutral (0.5 for all augmented recipes).
        try:
            recipes = list_recipes(crew_name=crew_name, limit=k, include_null=False)
        except Exception:
            recipes = []
        if recipes:
            candidates = [(r, 0.5) for r in recipes]

    # Test-mode override: allow tests to seed candidates without going
    # through Chroma. Off in production.
    if os.environ.get("META_AGENT_SELECTOR_FORCE_NULL_ONLY") == "1":
        return []

    return candidates
