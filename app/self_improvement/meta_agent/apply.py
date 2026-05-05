"""
app.self_improvement.meta_agent.apply — bounded recipe → factory augmentation.

apply_recipe(crew_name, recipe) returns the kwargs run_single_agent_crew
will use, with the recipe's bounded augmentation merged in. The recipe
NEVER replaces the agent factory's output — it only adjusts the channels
the factory already exposes:

    force_tier         — already a factory parameter
    extra_tools        — already a run_single_agent_crew parameter
    task_template      — non-destructively prefixed with task_hint
    max_execution_time — applied post-construction if set

Tool resolution
===============

Recipes carry tool *names*, not tool instances. At apply time we look
the names up in the global tool catalog (Phase 2 tool_registry) and
resolve them to instances. Any name that doesn't resolve is dropped
with a debug log — recipes are ground truth about *intent*, not
guarantees about availability.

Why a separate "augmentation" return type
=========================================

run_single_agent_crew already accepts force_tier (via difficulty), an
extra_tools list, and a task_template string. Returning a structured
augmentation lets the caller weave the recipe in cleanly without us
needing to take over the dispatch loop. The integration point is
~6 lines in run_single_agent_crew (see base_crew.py wiring).

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.self_improvement.meta_agent.types import AgentRecipe

logger = logging.getLogger(__name__)


@dataclass
class RecipeAugmentation:
    """How a recipe modifies the run_single_agent_crew dispatch.

    All fields are optional — if a recipe is null or partial, only the
    fields it sets are present, and run_single_agent_crew uses its
    existing defaults for the rest.
    """

    force_tier_override: Optional[str] = None
    task_template_prefix: str = ""
    extra_tools: list = field(default_factory=list)
    max_execution_time: Optional[int] = None

    # Names that didn't resolve at apply time. Logged but not fatal.
    unresolved_tool_names: list[str] = field(default_factory=list)

    @property
    def is_noop(self) -> bool:
        return (
            self.force_tier_override is None
            and not self.task_template_prefix
            and not self.extra_tools
            and self.max_execution_time is None
        )


def apply_recipe(
    *,
    crew_name: str,
    recipe: AgentRecipe,
) -> RecipeAugmentation:
    """Translate a recipe into the augmentation kwargs run_single_agent_crew uses.

    Pure: no side-effects. Tool name resolution is best-effort — any
    name that doesn't resolve falls into ``unresolved_tool_names`` and
    is logged at debug level. The caller can still proceed; partial
    augmentation is better than no augmentation.

    Args:
        crew_name: For logging context only — the recipe already knows
            its crew, but this lets the dispatcher's log line tag which
            crew was being dispatched.
        recipe: The recipe selected by selector.select_recipe.

    Returns:
        A RecipeAugmentation. If the recipe is null, returns a no-op
        augmentation (every field empty / None) — the dispatcher then
        falls through to its normal factory defaults.
    """
    if recipe.is_null:
        return RecipeAugmentation()

    aug = RecipeAugmentation(
        force_tier_override=recipe.force_tier,
        task_template_prefix=_format_task_hint(recipe.task_hint),
        max_execution_time=recipe.max_execution_time,
    )

    if recipe.extra_tool_names:
        resolved, unresolved = _resolve_tool_names(recipe.extra_tool_names)
        aug.extra_tools = resolved
        aug.unresolved_tool_names = unresolved
        if unresolved:
            logger.debug(
                "meta_agent.apply: %s/%s tools unresolved for crew=%s recipe=%s: %s",
                len(unresolved), len(recipe.extra_tool_names),
                crew_name, recipe.id, unresolved,
            )

    return aug


# ── Internal helpers ─────────────────────────────────────────────────────────

def _format_task_hint(hint: str) -> str:
    """Format a recipe task_hint as a non-destructive task-template prefix.

    Empty hint → empty string (no prefix). Otherwise wraps in a clearly
    delimited block so the agent can see it's an additional hint, not
    part of the user's actual task description.
    """
    hint = (hint or "").strip()
    if not hint:
        return ""
    return (
        "## Recipe-suggested approach\n"
        f"{hint}\n\n"
        "(Apply where it improves the result — the user's task below is "
        "the source of truth.)\n\n"
    )


def _resolve_tool_names(names: list[str]) -> tuple[list, list[str]]:
    """Look up ``names`` in the tool catalog and return (instances, unresolved).

    Uses the Phase 2 tool_registry singleton. The registry's
    ``build_instance(name)`` raises KeyError for unknown names and
    RuntimeError when a tool's guard() fails (missing env, unreachable
    dep). Both are treated as "unresolved" — the recipe's other knobs
    still apply.
    """
    try:
        from app.tool_registry.registry import ToolRegistry
    except Exception:
        return [], list(names)

    try:
        registry = ToolRegistry.instance()
    except Exception:
        return [], list(names)

    resolved: list = []
    unresolved: list[str] = []
    for name in names:
        try:
            tool = registry.build_instance(name)
        except (KeyError, RuntimeError):
            unresolved.append(name)
            continue
        except Exception:
            unresolved.append(name)
            continue
        if tool is None:
            unresolved.append(name)
        else:
            resolved.append(tool)
    return resolved, unresolved
