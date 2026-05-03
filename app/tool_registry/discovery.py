"""Tool discovery — capability + ChromaDB ranking with 4-layer defense.

Phase 1b component. This is the brain behind ``tool_search``: given an
intent (free text) and optional structured constraints (capability
tags, workspace, agent tier), return a ranked list of tools.

Why "4-layer defense" matters
-----------------------------
The May 2026 incident where a stale "Weather Forecast" skill matched
on a subject-less "execute the plan" query and hijacked an Estonia
deforestation request was a surface-keyword failure. The skills-
retrieval layer hardened against it with four independent gates;
this module ports the same discipline to tool retrieval. See
``tests/test_skill_retrieval_contamination.py`` for the original
regression panel; this module's tests mirror it for tools.

The 4 layers (in order)
-----------------------
1. **Quarantine filter** — tools listed in
   ``workspace/tool_registry/quarantine.json`` are filtered out
   entirely. No score, no rank — they don't appear at all. Mirrors
   ``workspace/skills/_quarantine/``.
2. **Tier gate** — tools above ``agent_tier`` are filtered out.
   PRODUCTION-only crews never see SHADOW tools; SHADOW crews see
   everything (including SHADOW + CANARY + PRODUCTION). Filter,
   not down-rank: a SHADOW tool seen by a PRODUCTION crew would be
   unloadable anyway.
3. **Workspace gate** — the tool's ``workspace_scope`` must include
   ``*`` or the active workspace ID. Same pattern as agent
   workspace allowlists — a forest-research crew shouldn't get
   ECB-finance tools suggested.
4. **Distance gate** — ChromaDB cosine distance must be below
   ``_DISTANCE_CEILING`` (default 0.55, matching the skill ceiling).
   Above the ceiling means "weak orthogonal match" — exactly the
   Weather/Estonia failure mode.

Plus three soft signals that affect ranking, not membership:
  * **Capability exact-match boost** — tools declaring at least one
    of the requested capability tags get a fixed score bonus.
  * **Tier rank** — within the authorized tier, PRODUCTION outranks
    CANARY outranks SHADOW.
  * **Loadability** — tools whose ``guard()`` currently returns
    False are visible (so the agent knows they exist) but ranked
    below loadable equivalents.

The intent string itself goes through one preprocessing step:
**subjectless detection**. Bare queries like "execute the plan",
"run it", "go" return empty results unless capability tags are also
provided — surfacing arbitrary tools on a subjectless query is the
exact failure mode the skills-retrieval layer was hardened against.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

from app.tool_registry.indexer import query_index
from app.tool_registry.quarantine import quarantined_names
from app.tool_registry.types import Tier, ToolSpec

logger = logging.getLogger(__name__)


# Tunable: cosine-distance ceiling. Above this, match is rejected.
# 0.55 matches the skills-retrieval ceiling (see app/agents/commander/
# context.py:24); same data-collection pipeline + same nomic-embed
# model, so the threshold transfers.
_DISTANCE_CEILING = 0.55

# Capability exact-match adds this much to the inverse-distance score.
# Tuned so a perfect tag match outranks a 0.40-distance prose-only
# match: 0.40 → score 0.60; +0.30 boost = 0.90 — wins.
_CAPABILITY_BOOST = 0.30

# Tier ranking — within authorized tier, prefer higher tier.
_TIER_RANK = {Tier.SHADOW: 0.0, Tier.CANARY: 0.1, Tier.PRODUCTION: 0.2, Tier.IMMUTABLE: 0.25}

# Subjectless query detection — these tokens, when comprising the
# WHOLE intent, are treated as "no signal". Pulled from
# app/agents/commander/context.py:53-74.
_SUBJECTLESS_TOKENS: frozenset[str] = frozenset({
    "ok", "go", "do", "run", "execute", "start", "begin",
    "the", "plan", "task", "this", "that", "it",
    "now", "please", "yes", "no", "fix", "report",
})


@dataclass(frozen=True)
class ToolMatch:
    """One ranked candidate from ``search_tools``.

    ``score`` is normalized to [0, 1.5] roughly — distance-derived
    base ([0, 1]) plus boosts. Higher = better; sort descending.
    ``reason`` is a one-line human-readable explanation of why this
    tool ranked where it did, surfaced to the LLM in the
    ``tool_search`` output so it can decide whether to load.
    """
    name: str
    score: float
    reason: str
    spec: ToolSpec

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score": round(self.score, 3),
            "reason": self.reason,
            "tier": self.spec.tier.value,
            "capabilities": list(self.spec.capabilities),
            "is_loadable": self.spec.is_loadable,
            "description": self.spec.description,
        }


def _is_subjectless(intent: str) -> bool:
    """True iff the intent is essentially noise — bare imperative
    tokens with no domain content. Mirrors
    ``app.agents.commander.context._is_subjectless_message``.
    """
    if not intent:
        return True
    tokens = re.findall(r"[a-z]+", intent.lower())
    if not tokens:
        return True
    if len(tokens) > 8:
        return False  # long messages have signal somewhere
    non_filler = [t for t in tokens if t not in _SUBJECTLESS_TOKENS]
    return len(non_filler) == 0


def _tier_passes_gate(spec: ToolSpec, agent_tier: Tier) -> bool:
    """Tier gate — tool's trust tier must be ≥ the agent's authorized
    tier. Tier ranking: SHADOW < CANARY < PRODUCTION < IMMUTABLE.

    Mental model: the agent's tier is the LOWEST trust level it's
    allowed to use. A PRODUCTION agent sees PRODUCTION + IMMUTABLE
    (proven-safe tools). A SHADOW agent (testing crew) sees
    everything because it's authorized for the riskiest tools.
    """
    return _TIER_RANK[spec.tier] >= _TIER_RANK[agent_tier]


def _workspace_passes_gate(spec: ToolSpec, workspace: str | None) -> bool:
    """Workspace gate — tool's workspace_scope must include ``*``
    or the active workspace ID."""
    if workspace is None:
        return True  # caller didn't pin a workspace; allow any
    if "*" in spec.workspace_scope:
        return True
    return workspace in spec.workspace_scope


# ── Public API ──────────────────────────────────────────────────────


def search_tools(
    intent: str = "",
    *,
    capabilities: Iterable[str] | None = None,
    workspace: str | None = None,
    agent_tier: Tier = Tier.PRODUCTION,
    limit: int = 5,
    distance_ceiling: float | None = None,
) -> list[ToolMatch]:
    """Rank tools for an agent looking for a capability.

    Args:
        intent: Natural-language description of what the agent wants
            to do. May be empty if ``capabilities`` is provided.
        capabilities: Optional list of capability tags. If provided,
            tools declaring at least one of them get an exact-match
            score boost AND tools declaring NONE of them are dropped
            (when ``intent`` is empty or subjectless — forces an
            either/or signal).
        workspace: Active workspace ID. Tools not scoped to this
            workspace are filtered out.
        agent_tier: Agent's authorized tier. Tools above it filtered.
            Default PRODUCTION (the safest assumption).
        limit: Max number of matches to return.
        distance_ceiling: Override the default 0.55 distance gate.
            Mostly for tests; production should use the default.

    Returns:
        Ranked list of ``ToolMatch``, highest score first. Empty list
        on subjectless intent + no capabilities specified, on
        infrastructure failure, or when nothing passes the gates.
    """
    from app.tool_registry.registry import ToolRegistry

    registry = ToolRegistry.instance()
    cap_set = set(capabilities) if capabilities else set()
    ceiling = distance_ceiling if distance_ceiling is not None else _DISTANCE_CEILING
    quarantined = quarantined_names()

    # ── Subjectless gate ────────────────────────────────────────
    # If the agent passed a pure-noise intent and no capability
    # tags, refuse to surface anything. Same defense the skills
    # layer uses against "execute the plan"-style hijacking.
    if _is_subjectless(intent) and not cap_set:
        logger.info(
            "tool_search: subjectless intent %r with no capabilities — "
            "returning empty (the 4-layer defense, layer 0).",
            intent,
        )
        return []

    # ── Candidate set ───────────────────────────────────────────
    # Two paths that get unioned:
    #   A. Capability tag exact match → registry.by_capability for each.
    #   B. ChromaDB semantic match on intent (if non-empty).
    #
    # Each path produces (name, source_score, source_label).
    candidate_scores: dict[str, tuple[float, list[str]]] = {}

    # Path A: capability exact-match candidates
    for tag in cap_set:
        for spec in registry.by_capability(tag):
            existing = candidate_scores.get(spec.name, (0.0, []))
            new_score = existing[0] + _CAPABILITY_BOOST
            existing[1].append(f"matches capability {tag}")
            candidate_scores[spec.name] = (new_score, existing[1])

    # Path B: semantic match (only if intent has signal)
    if intent and not _is_subjectless(intent):
        # Pull a few extra so we have headroom after gates.
        results = query_index(intent, limit=max(limit * 3, 10))
        for row in results:
            distance = float(row.get("distance", 1.0))
            if distance > ceiling:
                continue  # Layer 4: distance gate
            score = max(0.0, 1.0 - distance)  # closer = higher score
            existing = candidate_scores.get(row["name"], (0.0, []))
            new_score = existing[0] + score
            existing[1].append(f"semantic match (d={distance:.2f})")
            candidate_scores[row["name"]] = (new_score, existing[1])

    if not candidate_scores:
        return []

    # ── Apply Layers 1-3 (quarantine, tier, workspace) ──────────
    matches: list[ToolMatch] = []
    for name, (base_score, reasons) in candidate_scores.items():
        # Layer 1: quarantine
        if name in quarantined:
            continue
        spec = registry.get(name)
        if spec is None:
            # Stale ChromaDB entry — registry was reset but index lagged.
            continue
        # Layer 2: tier gate
        if not _tier_passes_gate(spec, agent_tier):
            continue
        # Layer 3: workspace gate
        if not _workspace_passes_gate(spec, workspace):
            continue

        # Soft signals: tier rank + loadability
        score = base_score + _TIER_RANK[spec.tier]
        if not spec.is_loadable:
            # Visible but ranked low — the agent knows it exists,
            # but a loadable equivalent will outrank.
            score *= 0.5
            reasons.append("not loadable in this deployment")

        matches.append(ToolMatch(
            name=name, score=score,
            reason="; ".join(reasons),
            spec=spec,
        ))

    matches.sort(key=lambda m: -m.score)
    return matches[:limit]
