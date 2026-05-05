"""
app.self_improvement.meta_agent — bounded meta-agent layer.

Inspired by Hyperagents (arXiv:2603.19461) but deliberately bounded to
respect BotArmy's safety invariant: "evaluation functions and safety
constraints live at INFRASTRUCTURE level, never in agent-modifiable
code paths" (CLAUDE.md).

What this layer does
====================

Optimises the *creation and configuration* of sub-agents at task
dispatch time, learning from historical success across runs:

    Task arrives  →  embed(task)  →  retrieve nearest historical recipes
                  →  UCB-rank by (similarity × success rate)
                  →  apply recipe as bounded augmentation onto the
                     existing agent factory's output
                  →  record outcome (success, confidence, duration)
                  →  feed next round of selection

The "recipe" is a bounded augmentation, NOT a replacement of the
agent factory. It can adjust:

    * force_tier            — LLM tier override (already a factory parameter)
    * extra_tool_names      — additional tools (already supported by
                              run_single_agent_crew)
    * task_hint             — short prefix injected into the task template
    * max_execution_time    — advisory ceiling

It cannot edit:

    * The agent factory itself (create_coder, create_researcher, ...)
    * backstory or goal text (those live in TIER_GATED souls/)
    * The LLM model selection rules (in llm_factory)
    * Anything in TIER_IMMUTABLE

What this layer does NOT do
===========================

The Hyperagents paper proposes that the meta-level modification
procedure should itself be editable. That directly conflicts with
BotArmy's "Self-Improver cannot modify its own evaluation criteria"
invariant. Instead, when this layer observes that good recipes are
systematically blocked by an immutable rule, it emits a TIER_IMMUTABLE
*amendment proposal* (a downloadable .md) for operator review via
the existing app.proposals pipeline. Operator action is required to
apply — no code in this package can edit auto_deployer.py.

Opt-in via env flags
====================

    META_AGENT=1                    — master switch (default OFF)
    META_AGENT_<CREW>=1 / =0        — per-crew override (e.g. META_AGENT_CODING=1)

Mirrors the LOADABLE_AGENT_EXPERIMENTAL pattern. Failsafe falls back
to factory defaults on any error in the meta_agent path.

IMMUTABLE selection thresholds
==============================

The selector's exploration constants (epsilon, ucb_c, similarity_tau)
live as module-level constants in selector.py — same convention as
NOVELTY_THRESHOLDS. The selector itself is part of the protected core
that the system cannot edit.

Module layout
=============

    types.py              — AgentRecipe, RecipeOutcome, RecipeSelection
    feature_flag.py       — is_meta_agent_enabled(crew_name)
    store.py              — Postgres + ChromaDB persistence (recipe registry,
                            outcome ledger, similarity index)
    selector.py           — UCB1 + similarity-weighted recipe selection
    apply.py              — apply_recipe(crew_name, recipe) returning the
                            augmentation kwargs run_single_agent_crew uses
    recorder.py           — record_outcome from the lifecycle envelope
    policy_gap.py         — detect "this recipe was good but immutable
                            constraints prevented it from running"
    amendment.py          — render TIER_IMMUTABLE amendment proposals

IMMUTABLE — infrastructure-level package.
"""

from __future__ import annotations

from app.self_improvement.meta_agent.types import (
    AgentRecipe,
    RecipeOutcome,
    RecipeSelection,
)
from app.self_improvement.meta_agent.feature_flag import (
    is_meta_agent_enabled, is_master_on, explicit_flag_for,
)
from app.self_improvement.meta_agent import meta_agent_settings
from app.self_improvement.meta_agent.store import (
    upsert_recipe,
    get_recipe,
    list_recipes,
    record_outcome as store_outcome,
    list_outcomes,
    similarity_search,
    ensure_schema,
    null_recipe_for,
)
from app.self_improvement.meta_agent.selector import (
    select_recipe,
    SELECTION_THRESHOLDS,
)
from app.self_improvement.meta_agent.apply import (
    apply_recipe,
    RecipeAugmentation,
)
from app.self_improvement.meta_agent.recorder import record_outcome
from app.self_improvement.meta_agent.policy_gap import (
    scan_for_policy_gaps,
    PolicyGap,
)
from app.self_improvement.meta_agent.amendment import (
    propose_immutable_amendment,
    render_amendment_md,
)

__all__ = [
    # types
    "AgentRecipe", "RecipeOutcome", "RecipeSelection",
    # feature flag
    "is_meta_agent_enabled", "is_master_on", "explicit_flag_for",
    # settings (dashboard toggle layer)
    "meta_agent_settings",
    # store
    "upsert_recipe", "get_recipe", "list_recipes",
    "store_outcome", "list_outcomes", "similarity_search",
    "ensure_schema", "null_recipe_for",
    # selector
    "select_recipe", "SELECTION_THRESHOLDS",
    # apply
    "apply_recipe", "RecipeAugmentation",
    # recorder
    "record_outcome",
    # policy gap
    "scan_for_policy_gaps", "PolicyGap",
    # amendment
    "propose_immutable_amendment", "render_amendment_md",
]
