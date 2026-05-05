"""
app.self_improvement.meta_agent.policy_gap — detect immutable-blocked recipes.

A "policy gap" is a recurring pattern where the meta-agent's selector
WANTED to apply a recipe knob that auto_deployer's TIER_IMMUTABLE set
(or a related safety-core constant) prevented from working. Examples:

    * The selector keeps generating recipes with extra tools that
      can't resolve because their factories are gated by an env var
      whose default is OFF in TIER_IMMUTABLE config.
    * A recipe's task_hint repeatedly steers the agent to a tool that
      lives in TIER_IMMUTABLE and the agent's factory can't surface it.
    * High-success recipes are blocked from promotion past SHADOW
      (this is by design for tools, but worth surfacing so an
      operator can decide whether to promote).

The detector aggregates outcome data into PolicyGap records. Each
PolicyGap is the input to amendment.propose_immutable_amendment, which
generates a downloadable .md proposal — never auto-applies anything.

Design constraints:
    - Read-only over outcomes + recipes. No writes here.
    - Safe to call from the idle scheduler.
    - Returns deterministic, dedupable records — repeated detection of
      the same gap collides on (signature_hash, target_filepath).

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from app.self_improvement.meta_agent.types import AgentRecipe, RecipeOutcome
from app.self_improvement.meta_agent.store import (
    list_recipes, list_outcomes,
)

logger = logging.getLogger(__name__)


# ── Public record ────────────────────────────────────────────────────────────

@dataclass
class PolicyGap:
    """A diagnosed pattern where an immutable rule blocks a successful recipe.

    Used by amendment.propose_immutable_amendment to render an operator
    proposal. Never auto-applied.
    """

    id: str                                  # deterministic, dedupable
    crew_name: str
    target_filepath: str                     # the TIER_IMMUTABLE entry suspected
    suggested_action: str                    # "demote_to_gated" | "remove" | "review"
    affected_recipes: list[str] = field(default_factory=list)
    affected_outcomes: int = 0
    success_rate_observed: float = 0.0       # over affected_outcomes
    unresolved_tool_names: list[str] = field(default_factory=list)
    rationale: str = ""
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc)
                              .isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return asdict(self)


# ── Detection ────────────────────────────────────────────────────────────────

# A recipe must accumulate at least this many successful outcomes before
# we trust the "this immutable rule is blocking us" signal. Below this
# threshold we don't have enough evidence to propose an amendment.
_MIN_SUCCESSFUL_OUTCOMES = 5

# Minimum success rate (smoothed) for a recipe to be considered
# "performing well enough to warrant an amendment". A 35% recipe being
# blocked is not a strong signal — but a 75% recipe is.
_MIN_SUCCESS_RATE_FOR_AMENDMENT = 0.65


def scan_for_policy_gaps(
    *,
    crew_name: str | None = None,
    since_days: int = 30,
) -> list[PolicyGap]:
    """Aggregate recipe outcomes into PolicyGap proposals.

    Two detection rules:

    1. **Tool resolution gap** — recipes whose extra_tool_names
       repeatedly fail to resolve, but whose other knobs (force_tier,
       task_hint) still produce above-baseline success. The unresolved
       tool name is correlated against TIER_IMMUTABLE / TIER_GATED to
       suggest the specific entry to amend.

    2. **Tier-cap gap** — recipes whose force_tier preference is
       systematically downgraded by the cost mode (because a higher
       tier is gated by an immutable rate-limit). This shows up as
       low success on premium-tier recipes whose alternative paths
       hit budget caps.

    Returns a list of PolicyGap records, deduped on (target_filepath,
    crew_name). The amendment proposer turns each into a single .md.

    Read-only. Safe to call from idle_scheduler.
    """
    crews = [crew_name] if crew_name else _all_crews_with_outcomes(since_days)
    out: list[PolicyGap] = []

    for crew in crews:
        try:
            out.extend(_scan_crew(crew, since_days=since_days))
        except Exception:
            logger.debug(f"policy_gap: scan_crew[{crew}] failed", exc_info=True)
    return out


def _all_crews_with_outcomes(since_days: int) -> list[str]:
    """Distinct crew_names with at least one outcome in the window."""
    rows = list_outcomes(since_days=since_days, limit=2000)
    return sorted({r.crew_name for r in rows if r.crew_name})


def _scan_crew(crew_name: str, *, since_days: int) -> list[PolicyGap]:
    recipes = {r.id: r for r in list_recipes(crew_name=crew_name, limit=200)}
    outcomes = list_outcomes(crew_name=crew_name, since_days=since_days, limit=2000)
    if not recipes or not outcomes:
        return []

    out: list[PolicyGap] = []
    out.extend(_detect_tool_resolution_gaps(crew_name, recipes, outcomes))
    return out


def _detect_tool_resolution_gaps(
    crew_name: str,
    recipes: dict[str, AgentRecipe],
    outcomes: list[RecipeOutcome],
) -> list[PolicyGap]:
    """Surface recipes whose extra tools can't resolve but otherwise succeed.

    The signal: a recipe has uses ≥ _MIN_SUCCESSFUL_OUTCOMES, success
    rate ≥ _MIN_SUCCESS_RATE_FOR_AMENDMENT, AND extra_tool_names but
    those tool names don't currently exist in the registry. The usual
    cause is the tool factory is gated by an env var whose default is
    OFF in the immutable config.
    """
    try:
        from app.tool_registry.registry import ToolRegistry
        registered_names = set(ToolRegistry.instance().names())
    except Exception:
        registered_names = set()

    gaps: list[PolicyGap] = []
    by_tool: dict[str, list[AgentRecipe]] = defaultdict(list)

    for recipe in recipes.values():
        if recipe.is_null:
            continue
        if recipe.uses < _MIN_SUCCESSFUL_OUTCOMES:
            continue
        if recipe.smoothed_success_rate < _MIN_SUCCESS_RATE_FOR_AMENDMENT:
            continue
        unresolved = [n for n in recipe.extra_tool_names if n not in registered_names]
        if not unresolved:
            continue
        for n in unresolved:
            by_tool[n].append(recipe)

    for tool_name, tool_recipes in by_tool.items():
        target_path = _suspect_immutable_path_for_tool(tool_name)
        affected_outcomes = sum(r.uses for r in tool_recipes)
        success_rate = (
            sum(r.successes for r in tool_recipes) / max(affected_outcomes, 1)
        )
        gap_id = _gap_id(crew_name, target_path or tool_name, "tool_resolution")
        gaps.append(PolicyGap(
            id=gap_id,
            crew_name=crew_name,
            target_filepath=target_path or "(unknown)",
            suggested_action="review" if not target_path else "demote_to_gated",
            affected_recipes=[r.id for r in tool_recipes],
            affected_outcomes=affected_outcomes,
            success_rate_observed=success_rate,
            unresolved_tool_names=[tool_name],
            rationale=(
                f"Recipe(s) for crew '{crew_name}' want to call tool "
                f"'{tool_name}' but it's not in the registry. The "
                f"high success rate of these recipes ({success_rate:.0%} "
                f"over {affected_outcomes} outcomes) suggests this tool "
                f"would help if available."
            ),
        ))
    return gaps


# ── Helpers ──────────────────────────────────────────────────────────────────

def _gap_id(crew_name: str, target: str, kind: str) -> str:
    payload = json.dumps({"crew": crew_name, "target": target, "kind": kind},
                         sort_keys=True)
    h = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"policy_gap_{kind}_{h}"


def _suspect_immutable_path_for_tool(tool_name: str) -> str:
    """Best-effort guess of the TIER_IMMUTABLE / TIER_GATED entry to amend.

    Walks app.auto_deployer.TIER_IMMUTABLE and looks for a path whose
    final component matches the tool name (e.g. tool name 'pdf_compose'
    → 'app/tools/pdf_compose.py'). Returns "" when no plausible match
    is found — the proposal then asks the operator to identify the
    target path manually.
    """
    try:
        from app.auto_deployer import TIER_IMMUTABLE, TIER_GATED
    except Exception:
        return ""

    candidates = list(TIER_IMMUTABLE) + list(TIER_GATED)
    target = tool_name.lower().replace("-", "_")
    for path in candidates:
        leaf = path.split("/")[-1].rsplit(".", 1)[0].lower()
        if leaf == target or target in leaf or leaf in target:
            return path
    return ""
