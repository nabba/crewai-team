"""
llm_catalog.py — Runtime-built LLM registry.

Architecture:
  - ``_BOOTSTRAP_CATALOG``: 3 survival entries (Sonnet, DeepSeek, qwen3.5:35b-a3b-q4_K_M)
    hand-coded so the system can boot with no network and no snapshot cache.
  - ``CATALOG``: the live mutable dict everything else imports. At startup
    it's a copy of ``_BOOTSTRAP_CATALOG``; ``app.llm_catalog_builder.refresh``
    mutates it in place with entries derived from Artificial Analysis +
    OpenRouter + Ollama.
  - ``resolve_role_default(role, cost_mode)``: score-based selection that
    replaces the old static ``ROLE_DEFAULTS`` dict. Candidates are filtered
    by tier floor / multimodal / tool-support requirements, then ranked by a
    blend of internal telemetry, catalog strengths, tool-use reliability and
    a cost-mode-dependent price penalty.

Hand-curated surface area (deliberately tiny):
  * 3 bootstrap entries (for degraded-mode operation).
  * Two policy tables: ``_COST_MODE_CEILING`` (soft penalty weight) and
    ``_ROLE_TIER_FLOOR`` (hard tier requirement per role).
  * The ``canonical_task_type`` classifier keywords.

Everything else — model_id, cost, context, multimodal, tool_calling,
strengths, tool_use_reliability — is populated by the builder from live
sources every 24h (Signal ``refresh catalog`` command for instant refresh).
"""

from __future__ import annotations

import re


# ── Canonical task-type taxonomy ──────────────────────────────────────────
# Nine strength-column keys used across CATALOG entries. Stable contract —
# llm_catalog_builder.derive_strengths populates every entry with all nine.

CANONICAL_TASK_TYPES: tuple[str, ...] = (
    "coding", "debugging", "architecture", "research", "writing",
    "reasoning", "multimodal", "vetting", "general",
)

TASK_ALIASES: dict[str, str] = {
    "code": "coding", "implement": "coding", "program": "coding",
    "fix": "debugging", "debug": "debugging",
    "review": "architecture", "architect": "architecture",
    "design": "architecture", "plan": "architecture",
    "write": "writing", "summarize": "writing", "document": "writing",
    "report": "writing",
    "research": "research", "search": "research", "find": "research",
    "learn": "research",
    "reason": "reasoning", "analyze": "reasoning", "think": "reasoning",
}

# ── Canonical role registry ──────────────────────────────────────────────
# Single source of truth for every *pinnable* role the system exposes —
# the union of:
#   (a) dispatched crew names (from app.crews.registry.install_defaults),
#   (b) specialist LLM roles (argument to llm_factory.create_specialist_llm),
#   (c) the "default" fallback role.
#
# Publishers (firebase.publish.report_catalog, dashboard_routes_llms.llm_catalog)
# and the React pin dialog all derive their role lists from here so
# adding a new crew or specialist only requires updating this tuple.

CREW_ROLES: tuple[str, ...] = (
    # Dispatched via app.crews.registry.install_defaults
    "research",
    "coding",
    "writing",
    "media",
    "creative",
    "pim",
    "financial",
    "desktop",
    "repo_analysis",
    "devops",
    # No registered crew — the commander answers directly without dispatch.
    # Kept here because it's a legitimate routing target the resolver can pin.
    "direct",
)

SPECIALIST_ROLES: tuple[str, ...] = (
    # Distinct LLM personas used inside crews / orchestrator gates,
    # none of which correspond to a dispatched crew. Each is an
    # acceptable ``role=`` argument to create_specialist_llm() and a
    # pin target.
    "commander",
    "critic",
    "vetting",
    "synthesis",
    "introspector",
    "self_improve",
    "planner",
    "evo_critic",
)

# All roles the resolver knows how to pick for + the user can pin.
PUBLIC_ROLES: tuple[str, ...] = CREW_ROLES + SPECIALIST_ROLES + ("default",)

# Unified runtime-mode vocabulary. A single axis controls both the
# candidate pool (which tiers / providers are reachable) and the cost
# stance (how aggressively to prefer cheaper models). Previously split
# across two concepts (``llm_mode`` runtime modes + ``cost_mode``); the
# split was conceptually redundant because the extremes collapsed (free
# implied budget, insane implied quality).
#
# Monotonic cost gradient: free < budget < balanced < quality < insane.
# ``anthropic`` is the special single-provider lock outside the gradient.
RUNTIME_MODES: tuple[str, ...] = (
    "free", "budget", "balanced", "quality", "insane", "anthropic",
)

# Deprecated alias — kept so legacy imports continue to work for one
# release. New code should read ``RUNTIME_MODES``. The tuple contents
# differ from the old ``COST_MODES`` (which had only 3 values), but every
# legacy value (``budget``/``balanced``/``quality``) is still present.
COST_MODES: tuple[str, ...] = RUNTIME_MODES

# Legacy-name → new-name normalisation. ``hybrid``/``local``/``cloud``
# were runtime modes; the unified vocabulary collapses them into the
# new set so callers don't have to know the history.
_MODE_ALIASES: dict[str, str] = {
    "hybrid": "balanced",
    "local":  "free",
    "cloud":  "balanced",
}


def _normalize_mode(mode: str) -> str:
    """Normalise a mode string to the current 6-value vocabulary.

    Accepts the legacy ``hybrid``/``local``/``cloud`` names and maps
    them onto their modern equivalents. Unknown inputs fall through to
    ``balanced`` (the default) so callers never crash on stale data.
    """
    m = (mode or "").strip().lower()
    m = _MODE_ALIASES.get(m, m)
    return m if m in RUNTIME_MODES else "balanced"


# Crew/role → canonical task type.
_ROLE_TO_TASK: dict[str, str] = {
    # Crew names (from orchestrator._run_crew)
    "coding":        "coding",
    "research":      "research",
    "writing":       "writing",
    "media":         "multimodal",
    "creative":      "writing",
    "pim":           "writing",
    "financial":     "reasoning",
    "desktop":       "general",
    "repo_analysis": "architecture",
    "devops":        "architecture",
    "direct":        "general",
    # Specialist roles
    "commander":     "general",
    "critic":        "reasoning",
    "introspector":  "reasoning",
    "self_improve":  "research",
    "vetting":       "vetting",
    "synthesis":     "writing",
    "planner":       "architecture",
    "evo_critic":    "reasoning",
    "architecture":  "architecture",
    "debugging":     "debugging",
    "reasoning":     "reasoning",
    "multimodal":    "multimodal",
    "default":       "general",
}

_TASK_KEYWORDS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\b(debug|traceback|stacktrace|fix\s+bug)\b", re.I), "debugging"),
    (re.compile(r"\b(image|photo|screenshot|picture|visual|pdf|scan)\b", re.I), "multimodal"),
    (re.compile(r"\b(video|audio|podcast|youtube|camera|media|voice|music|mp[34])\b", re.I), "multimodal"),
    (re.compile(r"\b(architect|design|system\s+design|review|plan)\b", re.I), "architecture"),
    (re.compile(r"\b(code|implement|function|class|module|script|program)\b", re.I), "coding"),
    (re.compile(r"\b(research|search|find|learn|investigate)\b", re.I), "research"),
    (re.compile(r"\b(write|summarize|document|report|explain|describe)\b", re.I), "writing"),
    (re.compile(r"\b(reason|think|logic|proof|math|analyze)\b", re.I), "reasoning"),
)


def canonical_task_type(
    role: str = "",
    task_hint: str = "",
    crew_name: str = "",
) -> str:
    """Resolve the canonical task_type for telemetry and strength lookups.

    Resolution order (most specific wins):
      1. Keyword match inside ``task_hint``.
      2. Direct lookup in ``TASK_ALIASES`` for ``task_hint`` or ``role``.
      3. Crew name lookup in ``_ROLE_TO_TASK``.
      4. Role lookup in ``_ROLE_TO_TASK``.
      5. Fallback: ``"general"``.
    """
    if task_hint:
        hint = task_hint.strip()
        for pattern, task_type in _TASK_KEYWORDS:
            if pattern.search(hint):
                return task_type
        aliased = TASK_ALIASES.get(hint.lower())
        if aliased in CANONICAL_TASK_TYPES:
            return aliased

    if crew_name:
        mapped = _ROLE_TO_TASK.get(crew_name.lower())
        if mapped:
            return mapped

    if role:
        role_lower = role.lower()
        mapped = _ROLE_TO_TASK.get(role_lower)
        if mapped:
            return mapped
        aliased = TASK_ALIASES.get(role_lower)
        if aliased in CANONICAL_TASK_TYPES:
            return aliased
        if role_lower in CANONICAL_TASK_TYPES:
            return role_lower

    return "general"


# ── Bootstrap catalog (hand-curated survival minimum) ────────────────────
# These three entries exist so the system boots when every external API is
# down and no snapshot is on disk. llm_catalog_builder refreshes their
# derived fields (cost, strengths, tool_use_reliability) from live sources
# but never removes them.

_BOOTSTRAP_CATALOG: dict[str, dict] = {
    "claude-sonnet-4.6": {
        "tier": "premium", "provider": "anthropic",
        "model_id": "anthropic/claude-sonnet-4-6",
        "context": 1_000_000, "multimodal": True,
        "max_output_tokens": 64_000,
        "cost_input_per_m": 1.00, "cost_output_per_m": 5.00,
        "tool_use_reliability": 0.95,
        "supports_tools": True,
        "description": "Claude Sonnet 4.6 — survival bootstrap (Anthropic fallback).",
        "strengths": {
            "coding": 0.91, "debugging": 0.88, "architecture": 0.88,
            "research": 0.90, "writing": 0.93, "reasoning": 0.90,
            "multimodal": 1.0, "vetting": 0.92, "general": 0.92,
        },
    },
    "deepseek-v3.2": {
        "tier": "budget", "provider": "openrouter",
        "model_id": "openrouter/deepseek/deepseek-chat",
        "context": 128_000, "multimodal": False,
        "max_output_tokens": 8_192,
        "cost_input_per_m": 0.28, "cost_output_per_m": 0.42,
        "tool_use_reliability": 0.82,
        "supports_tools": True,
        # Per DeepSeek V3 model card. Live builder will override from
        # OpenRouter ``created_at`` once the snapshot refreshes.
        "knowledge_cutoff": "2024-07-01",
        "description": "DeepSeek V3.2 — survival bootstrap (budget API fallback).",
        "strengths": {
            "coding": 0.87, "debugging": 0.87, "architecture": 0.85,
            "research": 0.85, "writing": 0.82, "reasoning": 0.90,
            "multimodal": 0.0, "vetting": 0.78, "general": 0.85,
        },
    },
    "qwen3.5:35b-a3b-q4_K_M": {
        "tier": "local", "provider": "ollama",
        "model_id": "ollama_chat/qwen3.5:35b-a3b-q4_K_M",
        # 256k native context window (Qwen3.5 family).  We cap usage via
        # num_ctx at runtime; the catalog records the model ceiling.
        "context": 262_144, "multimodal": True,
        "max_output_tokens": 8_192,
        "cost_input_per_m": 0.0, "cost_output_per_m": 0.0,
        # Bumped from 0.70 (qwen3:30b-a3b) — Qwen3.5 has stronger tools
        # support advertised on the model card; mem0 function-calling
        # error on the older 30b-a3b is the specific gap this addresses.
        "tool_use_reliability": 0.82,
        "supports_tools": True,
        "size_gb": 20, "ram_gb": 28, "speed": "very_fast",
        # Per Qwen3.5 model card.
        "knowledge_cutoff": "2024-09-01",
        "description": "Qwen3.5 35B-A3B (MoE Q4_K_M) — local survival "
                       "bootstrap. 35B total / 3B active per token. Vision "
                       "+ tools + thinking-mode capable. Replaces "
                       "qwen3:30b-a3b.",
        "strengths": {
            "coding": 0.85, "debugging": 0.80, "architecture": 0.80,
            "research": 0.80, "writing": 0.80, "reasoning": 0.82,
            "multimodal": 0.65, "vetting": 0.62, "general": 0.82,
        },
    },
}

# Live mutable catalog. ``app.llm_catalog_builder.refresh()`` adds or updates
# entries here in place. Code across the app imports ``CATALOG`` by reference;
# mutating it keeps every caller's view current without any listener plumbing.
CATALOG: dict[str, dict] = {name: dict(entry) for name, entry in _BOOTSTRAP_CATALOG.items()}


# ── Persistent overrides (governance-approved) ──────────────────────────
# When a ``model_id_remap`` or ``model_retired`` governance request is
# approved, the change is recorded in workspace/llm_catalog_overrides.json
# so it survives container restarts.  Without this, the llm_discovery
# detector would re-find the same dead ID on every restart and create a
# duplicate governance request.
#
# File schema (all fields optional):
#   {
#     "model_id_remaps": {"<catalog_key>": "<new_model_id>", ...},
#     "retired":          ["<catalog_key>", ...]
#   }

import json as _json
import os as _os
import threading as _threading
from pathlib import Path as _Path

_OVERRIDES_PATH = _Path(_os.environ.get(
    "LLM_CATALOG_OVERRIDES",
    "/app/workspace/llm_catalog_overrides.json",
))
_OVERRIDES_LOCK = _threading.Lock()


def _load_overrides() -> dict:
    """Read the overrides file. Returns {} on any error or missing file."""
    try:
        if _OVERRIDES_PATH.exists():
            return _json.loads(_OVERRIDES_PATH.read_text())
    except Exception:
        pass
    return {}


def _apply_overrides_to_catalog(catalog: dict[str, dict]) -> None:
    """Mutate *catalog* in place with any governance-approved changes.

    Called once at module import, and again from llm_catalog_builder.refresh()
    after it rebuilds the catalog so builder-driven updates don't stomp on
    approved remaps/retirements.
    """
    data = _load_overrides()
    remaps = (data.get("model_id_remaps") or {}) if isinstance(data, dict) else {}
    retired = set((data.get("retired") or [])) if isinstance(data, dict) else set()

    for catalog_key, new_model_id in remaps.items():
        entry = catalog.get(catalog_key)
        if entry and isinstance(new_model_id, str) and new_model_id:
            entry["model_id"] = new_model_id
    for catalog_key in retired:
        entry = catalog.get(catalog_key)
        if entry:
            entry["_retired"] = True


def persist_model_id_remap(catalog_key: str, new_model_id: str) -> bool:
    """Record an approved model_id_remap to the overrides file AND apply it
    to the live CATALOG in place.  Thread-safe.

    Returns True if the file was written successfully.
    """
    if not catalog_key or not new_model_id:
        return False
    with _OVERRIDES_LOCK:
        try:
            data = _load_overrides() or {}
            remaps = data.setdefault("model_id_remaps", {})
            remaps[catalog_key] = new_model_id
            _OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
            _OVERRIDES_PATH.write_text(_json.dumps(data, indent=2, sort_keys=True))
            # Apply immediately so running processes don't need a restart
            entry = CATALOG.get(catalog_key)
            if entry:
                entry["model_id"] = new_model_id
                # Clear any _retired flag since we just proved this key works
                entry.pop("_retired", None)
            return True
        except Exception:
            return False


def persist_model_retired(catalog_key: str) -> bool:
    """Record an approved model_retired to the overrides file AND mark the
    live CATALOG entry retired.  Thread-safe.
    """
    if not catalog_key:
        return False
    with _OVERRIDES_LOCK:
        try:
            data = _load_overrides() or {}
            retired = data.setdefault("retired", [])
            if catalog_key not in retired:
                retired.append(catalog_key)
            _OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
            _OVERRIDES_PATH.write_text(_json.dumps(data, indent=2, sort_keys=True))
            entry = CATALOG.get(catalog_key)
            if entry:
                entry["_retired"] = True
            return True
        except Exception:
            return False


# Apply overrides at import time so the catalog starts with any
# governance-approved changes already baked in.  llm_catalog_builder.refresh()
# calls _apply_overrides_to_catalog() again after rebuilding to ensure
# freshly-fetched entries don't overwrite approved overrides.
_apply_overrides_to_catalog(CATALOG)


# ── Policy (tiny, rarely changed) ────────────────────────────────────────

# Cost-penalty weight in the scoring function. Applied as
# ``quality - weight * (cost / max_cost_in_candidates)`` so a 5% quality
# improvement isn't worth a 20× cost increase under budget. Higher values
# make the resolver more aggressively prefer cheaper models.
_MODE_WEIGHT: dict[str, float] = {
    "free":      0.50,   # aggressively prefer free (everything else is irrelevant anyway)
    "budget":    0.35,   # strongly prefer cheap
    "balanced":  0.15,   # mild cost sensitivity
    "quality":   0.05,   # very mild — prefer the best
    "insane":    0.00,   # money no object
    "anthropic": 0.15,   # mild — pick the best Anthropic within reason
}

# Which catalog tiers a mode is allowed to pick from. This is the single
# biggest lever — e.g. ``insane`` is premium-only, ``free`` only sees
# no-cost tiers. Applied as a hard filter before scoring.
_MODE_TIER_WHITELIST: dict[str, frozenset[str]] = {
    "free":      frozenset({"local", "free"}),
    "budget":    frozenset({"local", "free", "budget"}),
    "balanced":  frozenset({"local", "free", "budget", "mid", "premium"}),
    "quality":   frozenset({"local", "free", "budget", "mid", "premium"}),
    "insane":    frozenset({"premium"}),
    "anthropic": frozenset({"mid", "premium"}),  # includes Haiku-tier entries
}

# Provider whitelist per mode. ``None`` = every provider allowed.
# Only ``anthropic`` mode restricts by provider today.
_MODE_PROVIDER_WHITELIST: dict[str, frozenset[str] | None] = {
    "free":      None,
    "budget":    None,
    "balanced":  None,
    "quality":   None,
    "insane":    None,
    "anthropic": frozenset({"anthropic"}),
}

# Modes in which a local-preferred role actually gets its local pick.
# Quality/insane/anthropic explicitly opt out — user wants premium even
# for background roles.
_MODE_PREFER_LOCAL: frozenset[str] = frozenset({"free", "budget", "balanced"})

# Back-compat alias — old code reads ``_COST_MODE_WEIGHT``. Keep the
# reference so ROLE_DEFAULTS shim and any external module continues to
# work unchanged.
_COST_MODE_WEIGHT = _MODE_WEIGHT

# Hard tier requirements per role — commander/vetting/critic must be premium
# regardless of mode. Everything else accepts budget or above.
_ROLE_TIER_FLOOR: dict[str, str] = {
    "commander":  "premium",
    "vetting":    "premium",
    "critic":     "premium",
    "default":    "budget",
}

# Roles that can opt into local tier when one is available and the mode
# allows it (self-reflection, background jobs — cost sensitivity high).
_ROLE_LOCAL_PREFERRED: set[str] = {
    "introspector", "self_improve", "planner", "evo_critic",
}

# Roles whose crews use tool-calling — must pick models with tool support.
_ROLES_NEEDING_TOOLS: set[str] = {
    "coding", "research", "writing", "media", "self_improve", "critic",
    "vetting", "commander", "synthesis", "introspector",
}

_TIER_RANK: dict[str, int] = {
    "local": 0, "free": 1, "budget": 2, "mid": 3, "premium": 4,
}


def _tier_meets_floor(tier: str, floor: str) -> bool:
    return _TIER_RANK.get(tier, 0) >= _TIER_RANK.get(floor, 0)


def _effective_tier_floor(mode: str, role_tier_floor: str) -> str:
    """Reconcile the role's tier floor with the mode's tier whitelist.

    When the user explicitly picks a restrictive mode (e.g. ``free``), we
    honour that choice by capping the role tier floor at the highest
    tier the mode allows — otherwise the resolver would fall through to
    the bootstrap premium fallback even when the user asked for zero
    cost. The role still gets its *best* available pick within the mode.
    """
    whitelist = _MODE_TIER_WHITELIST.get(mode, _MODE_TIER_WHITELIST["balanced"])
    if not whitelist:
        return role_tier_floor
    max_allowed = max(whitelist, key=lambda t: _TIER_RANK.get(t, 0))
    if _TIER_RANK.get(role_tier_floor, 0) > _TIER_RANK.get(max_allowed, 0):
        return max_allowed
    return role_tier_floor


def _filter_candidates(
    mode: str,
    tier_floor: str,
    needs_multimodal: bool,
    prefer_local: bool,
    needs_tools: bool,
) -> list[str]:
    """Return catalog keys that satisfy the hard constraints.

    Hard constraints (all AND):
      * Mode tier whitelist (``_MODE_TIER_WHITELIST[mode]``).
      * Mode provider whitelist (``_MODE_PROVIDER_WHITELIST[mode]``).
      * Role tier floor (unless prefer_local is set and local is
        whitelisted by the mode).
      * Multimodal capability when the task requires it.
      * Tool-calling support when the role needs tools.

    Cost is NOT a hard filter — it enters via the scoring function so the
    selector can surface a cheaper model that meets the tier floor.
    """
    tier_whitelist = _MODE_TIER_WHITELIST.get(
        mode, _MODE_TIER_WHITELIST["balanced"],
    )
    provider_whitelist = _MODE_PROVIDER_WHITELIST.get(mode)

    candidates: list[str] = []
    for name, entry in CATALOG.items():
        tier = entry.get("tier", "budget")
        provider = entry.get("provider", "")
        if needs_multimodal and not entry.get("multimodal"):
            continue
        if needs_tools and entry.get("supports_tools") is False:
            continue
        if provider_whitelist is not None and provider not in provider_whitelist:
            continue
        if tier not in tier_whitelist:
            continue
        if not _tier_meets_floor(tier, tier_floor):
            if not (prefer_local and tier == "local"):
                continue
        candidates.append(name)
    return candidates


def resolve_role_default(
    role: str,
    mode: str = "balanced",
    *,
    cost_mode: str | None = None,
) -> str:
    """Three-layer role → catalog-key resolution.

    The ``mode`` parameter is the unified runtime-mode axis (see
    :data:`RUNTIME_MODES`). For back-compat, callers may still pass
    ``cost_mode=`` — the value is normalised onto the 6-mode vocabulary
    via :func:`_normalize_mode`.

    Authority (strongest first):
      1. **Hand-pin** — active overlay in ``control_plane.role_assignments``
         with ``priority ≥ HAND_PIN_PRIORITY`` (source='manual').
         Returned directly; no scoring. ``unpin_role`` removes the pin.
      2. **Promotion** — catalog entries present in
         ``control_plane.model_promotions``. If any promoted model fits
         the role's hard constraints, candidates are filtered down to
         the promoted set before scoring. ``demote`` removes the flag.
      3. **Pool** — regular blended scoring (telemetry + catalog
         strengths + tool-use reliability minus cost penalty).

    When no candidate survives the hard filters, returns
    ``"claude-sonnet-4.6"`` — the universal bootstrap fallback.
    """
    # Back-compat: accept legacy ``cost_mode=`` keyword as alias.
    if cost_mode is not None:
        mode = cost_mode
    mode = _normalize_mode(mode)

    # Layer 1: hand-pin hard override
    try:
        from app.llm_role_assignments import get_assigned_model
        pin = get_assigned_model(role, mode)
        if pin and pin in CATALOG:
            return pin
    except Exception:
        pass  # graceful degradation to layer 2/3

    task_type = canonical_task_type(role=role)
    role_tier_floor = _ROLE_TIER_FLOOR.get(role, _ROLE_TIER_FLOOR["default"])
    tier_floor = _effective_tier_floor(mode, role_tier_floor)
    needs_multimodal = task_type == "multimodal"
    prefer_local = (
        role in _ROLE_LOCAL_PREFERRED and mode in _MODE_PREFER_LOCAL
    )
    needs_tools = role in _ROLES_NEEDING_TOOLS

    candidates = _filter_candidates(
        mode, tier_floor, needs_multimodal, prefer_local, needs_tools,
    )

    if prefer_local:
        local_cands = [c for c in candidates if CATALOG[c].get("tier") == "local"]
        if local_cands:
            candidates = local_cands

    # Layer 2: promotion filter
    try:
        from app.llm_promotions import list_promoted
        promoted = list_promoted()
        if promoted:
            promoted_candidates = [c for c in candidates if c in promoted]
            if promoted_candidates:
                # At least one promoted model fits → restrict to promoted.
                candidates = promoted_candidates
    except Exception:
        pass  # missing promotions table just means layer 2 is inert

    if not candidates:
        return "claude-sonnet-4.6"

    # Blended quality signal: live benchmark if present, catalog strengths
    # otherwise. Bench gets the dominant weight because it's grounded in
    # real production outcomes.
    try:
        from app.llm_benchmarks import get_combined_scores
        bench = get_combined_scores(task_type)
    except Exception:
        bench = {}

    cost_weight = _MODE_WEIGHT.get(mode, _MODE_WEIGHT["balanced"])
    # Normalise the cost penalty against the most expensive candidate so
    # the penalty stays comparable across roles with different tier floors.
    costs = [CATALOG[n].get("cost_output_per_m", 0.0) for n in candidates]
    max_cost = max(costs) if costs else 1.0
    if max_cost <= 0:
        max_cost = 1.0

    def _score(name: str) -> float:
        entry = CATALOG[name]
        s_bench = float(bench.get(name, 0.0))
        strengths = entry.get("strengths", {})
        s_strength = float(strengths.get(task_type, strengths.get("general", 0.5)))
        s_tool = float(entry.get("tool_use_reliability", 0.7)) if needs_tools else 1.0
        quality = (
            0.60 * s_bench + 0.35 * s_strength + 0.05 * s_tool
            if s_bench > 0
            else 0.80 * s_strength + 0.20 * s_tool
        )
        cost = float(entry.get("cost_output_per_m", 0.0))
        cost_penalty = cost_weight * (cost / max_cost)
        return quality - cost_penalty

    return max(candidates, key=_score)


# ── Public API ────────────────────────────────────────────────────────────

def get_model(name: str) -> dict | None:
    return CATALOG.get(name)

def get_model_id(name: str) -> str:
    entry = CATALOG.get(name)
    if not entry:
        raise KeyError(f"Model {name!r} not in catalog")
    return entry["model_id"]

def get_tier(name: str) -> str:
    entry = CATALOG.get(name)
    return entry["tier"] if entry else "unknown"

def get_provider(name: str) -> str:
    entry = CATALOG.get(name)
    return entry["provider"] if entry else "unknown"

def is_multimodal(name: str) -> bool:
    entry = CATALOG.get(name)
    return entry.get("multimodal", False) if entry else False


def get_default_for_role(
    role: str,
    mode: str = "balanced",
    *,
    cost_mode: str | None = None,
) -> str:
    """Return the catalog key to use for a role at a given runtime mode.

    Resolution order:
      1. Runtime overlay in ``control_plane.role_assignments`` (manual
         override / governance-approved promotion — see
         :mod:`app.llm_role_assignments`).
      2. :func:`resolve_role_default` — score-based selection against the
         live ``CATALOG``.

    ``cost_mode=`` is accepted as a legacy alias for ``mode=``.
    """
    if cost_mode is not None:
        mode = cost_mode
    mode = _normalize_mode(mode)
    try:
        from app.llm_role_assignments import get_assigned_model
        override = get_assigned_model(role, mode)
        if override and override in CATALOG:
            return override
    except Exception:
        pass
    return resolve_role_default(role, mode)


def get_candidates(task_type: str) -> list[tuple[str, float]]:
    task_type = TASK_ALIASES.get(task_type, task_type)
    scored = []
    for name, info in CATALOG.items():
        strengths = info.get("strengths", {})
        score = strengths.get(task_type, strengths.get("general", 0.5))
        scored.append((name, score))
    scored.sort(key=lambda x: -x[1])
    return scored


def get_candidates_by_tier(
    task_type: str, tiers: list[str] | None = None,
) -> list[tuple[str, float]]:
    task_type = TASK_ALIASES.get(task_type, task_type)
    scored = []
    for name, info in CATALOG.items():
        if tiers and info.get("tier") not in tiers:
            continue
        strengths = info.get("strengths", {})
        score = strengths.get(task_type, strengths.get("general", 0.5))
        scored.append((name, score))
    scored.sort(key=lambda x: -x[1])
    return scored


def get_smallest_model() -> str:
    local = {k: v for k, v in CATALOG.items() if v.get("tier") == "local"}
    if not local:
        return "deepseek-v3.2"
    return min(local, key=lambda m: local[m].get("ram_gb", 99))


def get_ram_requirement(model: str) -> float:
    info = CATALOG.get(model)
    return info["ram_gb"] if info and "ram_gb" in info else 20.0


def estimate_task_cost(
    model_name: str, input_tokens: int = 2000, output_tokens: int = 2000,
) -> float:
    entry = CATALOG.get(model_name)
    if not entry:
        return 0.0
    return (
        (input_tokens / 1_000_000) * entry.get("cost_input_per_m", 0)
        + (output_tokens / 1_000_000) * entry.get("cost_output_per_m", 0)
    )


def format_catalog() -> str:
    """Human-readable catalog snapshot for Signal output."""
    lines = ["LLM Model Catalog:\n"]
    for tier_name in ("local", "free", "budget", "mid", "premium"):
        tier_models = {k: v for k, v in CATALOG.items() if v.get("tier") == tier_name}
        if not tier_models:
            continue
        lines.append(f"\n  [{tier_name.upper()}]")

        def _top_strength(info: dict) -> float:
            s = info.get("strengths", {})
            return max(s.values()) if s else 0.0

        for name, info in sorted(tier_models.items(), key=lambda x: -_top_strength(x[1])):
            s = info.get("strengths", {})
            top = max(s, key=s.get) if s else "?"
            cost = info.get("cost_output_per_m", 0)
            cost_str = "free" if cost == 0 else f"${cost:.2f}/Mo"
            marker = "·" if info.get("_auto") else "★"  # ★ = bootstrap
            lines.append(
                f"  {marker} {name}  ({cost_str}, "
                f"tool:{info.get('tool_use_reliability', 0):.0%}) — best: {top}"
            )
    return "\n".join(lines)


def format_role_assignments(
    mode: str = "balanced",
    *,
    cost_mode: str | None = None,
) -> str:
    """Show the resolver's current pick for every role at a given mode.

    ``cost_mode=`` accepted as a legacy alias for ``mode=``.
    """
    if cost_mode is not None:
        mode = cost_mode
    mode = _normalize_mode(mode)
    roles_to_show = [
        "commander", "coding", "research", "writing", "media", "critic",
        "introspector", "self_improve", "vetting", "synthesis", "planner",
        "evo_critic", "default",
    ]
    lines = [f"Role Assignments [{mode}] (resolved):\n"]
    for role in roles_to_show:
        picked = get_default_for_role(role, mode)
        entry = CATALOG.get(picked, {})
        cost = entry.get("cost_output_per_m", 0)
        cost_str = "free" if cost == 0 else f"${cost:.2f}/Mo"
        lines.append(f"  {role:<14} → {picked} ({cost_str})")
    return "\n".join(lines)


# ── Back-compat: ROLE_DEFAULTS as a lazy view over the resolver ────────────
# A handful of call sites (``firebase/publish.py``, older ``llm_discovery``
# paths) read ``ROLE_DEFAULTS[mode][role]`` directly. Rather than rewrite
# each, expose a dict-like wrapper that computes entries on access via
# ``resolve_role_default``. Always reflects the live CATALOG.

class _ResolvedRoleMap:
    """Dict-like view: ``m[role]`` → ``resolve_role_default(role, mode)``."""

    __slots__ = ("_mode",)

    def __init__(self, mode: str) -> None:
        self._mode = _normalize_mode(mode)

    def __contains__(self, role: object) -> bool:
        return isinstance(role, str)

    def __getitem__(self, role: str) -> str:
        return resolve_role_default(role, self._mode)

    def get(self, role: str, default=None):
        try:
            return resolve_role_default(role, self._mode)
        except Exception:
            return default

    def items(self):
        for role in _ROLE_TO_TASK:
            yield role, resolve_role_default(role, self._mode)

    def keys(self):
        return list(_ROLE_TO_TASK.keys())


class _RoleDefaultsView:
    """Back-compat shim for the ``ROLE_DEFAULTS[mode][role]`` idiom.

    Accepts any mode name the resolver understands (the 6 canonical
    values plus the legacy ``hybrid``/``local``/``cloud`` aliases via
    ``_normalize_mode``).
    """

    __slots__ = ()

    def _accept(self, mode: object) -> bool:
        if not isinstance(mode, str):
            return False
        # Accept both canonical and alias names. Normalisation maps aliases
        # onto canonical names inside _ResolvedRoleMap.
        if mode in _MODE_WEIGHT:
            return True
        if mode.strip().lower() in _MODE_ALIASES:
            return True
        return False

    def __contains__(self, mode: object) -> bool:
        return self._accept(mode)

    def __getitem__(self, mode: str) -> _ResolvedRoleMap:
        if not self._accept(mode):
            raise KeyError(mode)
        return _ResolvedRoleMap(mode)

    def get(self, mode: str, default=None):
        if self._accept(mode):
            return _ResolvedRoleMap(mode)
        return default


# Exposed for legacy callers; new code should use ``get_default_for_role``.
ROLE_DEFAULTS: _RoleDefaultsView = _RoleDefaultsView()
