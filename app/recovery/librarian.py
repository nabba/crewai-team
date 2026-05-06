"""
librarian.py — Read-only inventory of what the system can do.

When a refusal fires, the librarian answers: "given this task and this
refusal category, what alternative routes exist?" It reads from the
existing registries (crews, tools, LLM catalog, adapters) without
mutating state, so it's cheap to call repeatedly and safe to
short-circuit on errors.

The output is a ranked list of ``Alternative`` objects, cheapest first.
The recovery loop walks the list within a budget; the FIRST one that
succeeds wins.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Alternative:
    """One concrete recovery attempt the loop could try.

    Fields are deliberately specific — no opaque "config" dict — so
    each strategy has typed inputs.
    """
    strategy: str               # "re_route" / "escalate_tier" / "direct_tool" / "forge_queue"
    rationale: str              # human-readable why this might work
    est_cost_usd: float         # rough estimate, cents — for budget ordering
    est_latency_s: float        # rough wall-clock estimate
    sync: bool                  # True = run inside the user's request; False = async
    crew: str | None = None     # for re_route: target crew name
    tier: str | None = None     # for escalate_tier: target tier ('premium', 'mid')
    tool: str | None = None     # for direct_tool: tool name
    extra: dict = field(default_factory=dict)


# ── Crew/tool capability map ───────────────────────────────────────────
#
# Hand-curated mapping of which crew has tools relevant to which
# refusal-category. Rationale: crews register their tools at runtime
# and we COULD walk that registry, but the mapping is stable enough
# that a small static table is more debuggable + faster.
#
# This is the table that tells the librarian "the user asked about
# email and the research crew refused — try PIM."

# Tools wired into the direct_tool recipe table (see
# app/recovery/strategies/direct_tool.py:_TOOL_RECIPES). Consulted by
# both the keyword path below and the registry-bridge fallback in
# `_registry_alternatives` — no point suggesting a tool we don't know
# how to call directly.
_DIRECT_TOOL_RECIPE_NAMES: tuple[str, ...] = (
    "email_tools.check_email",
    "calendar_tools.list_events",
)


_CAPABILITY_MAP: dict[str, dict] = {
    "email": {
        "crews": ["pim"],
        # ``email_tools.check_email`` is the actual tool name registered
        # by app/tools/email_tools.py:create_email_tools — used by the
        # direct_tool strategy's recipe table.
        "tools": ["email_tools.check_email", "email_tools.send_email"],
        "keywords": ("email", "e-mail", "inbox", "mailbox", "gmail", "imap"),
    },
    "calendar": {
        "crews": ["pim"],
        "tools": ["calendar_tools.list_events", "calendar_tools.create_event"],
        "keywords": ("calendar", "meeting", "appointment", "event", "schedule"),
    },
    "tasks": {
        "crews": ["pim"],
        "tools": ["task_tools.list_tasks"],
        "keywords": ("task", "todo", "to-do"),
    },
    "code_execute": {
        "crews": ["coding"],
        "tools": ["code_executor", "sandbox.run"],
        "keywords": ("execute", "run code", "stdout", "output of"),
    },
    "research_matrix": {
        "crews": ["research"],
        "tools": ["research_orchestrator"],
        "keywords": ("for these", "for each", "list of", "compile a", "table of"),
    },
    "web": {
        "crews": ["research"],
        "tools": ["web_search", "browser_fetch", "firecrawl"],
        "keywords": ("search the web", "look up", "find online", "google"),
    },
    "files": {
        "crews": ["desktop", "research"],
        "tools": ["file_manager", "read_attachment"],
        "keywords": ("file", "attachment", "document", "pdf"),
    },
}


def _infer_capabilities(task: str) -> list[str]:
    """Return capability keys whose keywords appear in ``task``.

    A task can map to multiple capabilities — "send my colleague the
    file we discussed yesterday" hits ``email`` + ``files``.
    """
    if not task:
        return []
    text = task.lower()
    hits = []
    for cap_key, cap_def in _CAPABILITY_MAP.items():
        if any(kw in text for kw in cap_def["keywords"]):
            hits.append(cap_key)
    return hits


# ── Tier escalation ─────────────────────────────────────────────────────

def _current_tier_for_role(role: str) -> str | None:
    """Best-effort lookup of the tier the system would normally pick
    for ``role`` in the active cost mode. Used to decide whether
    escalation is even possible (no point escalating from premium to premium)."""
    try:
        from app.llm_selector import select_model
        from app.llm_catalog import CATALOG
        m = select_model(role=role)
        if m:
            entry = CATALOG.get(m, {})
            return entry.get("tier")
    except Exception:
        pass
    return None


# ── Tool-registry bridge ────────────────────────────────────────────────

def _registry_alternatives(task: str, used_crew: str) -> list[Alternative]:
    """Augment the keyword-curated _CAPABILITY_MAP with semantic search
    over the tool registry's ChromaDB index.

    Closes the gap where a `@register_tool`-annotated tool exists but
    the user's phrasing doesn't hit any keyword in the map. Reuses
    `tool_registry.discovery.search_tools` so the 4-layer contamination
    defense (subjectless guard, quarantine, tier, workspace, distance
    ceiling 0.55) applies uniformly to recovery suggestions.

    Today the only actionable bridge is to `direct_tool` — the registry
    doesn't expose a source_module → crew map, so `re_route` from a
    registry hit isn't yet derivable. When that lands, emit re_route
    here too.

    Falls back to empty list on any infrastructure failure — registry
    blips must never break recovery.
    """
    try:
        from app.tool_registry.discovery import search_tools
        matches = search_tools(intent=task, limit=5)
    except Exception:
        logger.debug("librarian: registry search failed", exc_info=True)
        return []

    out: list[Alternative] = []
    for m in matches:
        if m.name in _DIRECT_TOOL_RECIPE_NAMES:
            out.append(Alternative(
                strategy="direct_tool",
                tool=m.name,
                rationale=f"Tool registry semantic match ({m.reason}).",
                est_cost_usd=0.0,
                est_latency_s=5.0,
                sync=True,
            ))
    return out


# ── Public API ──────────────────────────────────────────────────────────

# Default ranking: cheaper + faster + more-likely-to-work strategies first.
# When two alternatives have similar cost, prefer the one with stronger
# evidence (specific tool match > generic crew swap > tier escalation).

def find_alternatives(
    task: str,
    refusal_category: str,
    used_crew: str,
    used_tier: str | None = None,
    response_text: str = "",
) -> list[Alternative]:
    """Return ranked alternatives for the loop to try.

    Args:
        task: the user's original request
        refusal_category: from RefusalSignal.category
        used_crew: the crew that produced the refusal
        used_tier: the tier the LLM was at (for tier-escalation logic)
        response_text: the refused response (used by sandbox_execute to
            decide whether to extract a code block)
    """
    out: list[Alternative] = []

    inferred = _infer_capabilities(task)

    # ── Strategy: direct_tool — bypass LLM, call tool directly ─────
    # CHEAPEST sync path. Use when the refusal categorically maps to a
    # specific tool we have wired up (email, calendar, etc.). No LLM
    # call, no crew spin-up — just regex + tool invocation.
    for cap_key in inferred:
        cap = _CAPABILITY_MAP[cap_key]
        for tool in cap.get("tools", []):
            # Only for tools that have a recipe in direct_tool's table.
            # Hard-coded list keeps the librarian from suggesting tools
            # we don't actually know how to call directly.
            if tool in _DIRECT_TOOL_RECIPE_NAMES:
                out.append(Alternative(
                    strategy="direct_tool",
                    tool=tool,
                    rationale=(
                        f"Detected '{cap_key}' capability; calling "
                        f"{tool} directly (no agent layer)."
                    ),
                    est_cost_usd=0.0,
                    est_latency_s=5.0,
                    sync=True,
                ))

    # ── Strategy: sandbox_execute — run code from coding-crew dump ──
    # Only when the response actually contains a Python code block.
    if response_text and "```" in response_text:
        import re as _re
        if _re.search(r"```(?:python|py)", response_text):
            out.append(Alternative(
                strategy="sandbox_execute",
                rationale=(
                    "Response contains executable Python — run it in "
                    "the sandbox instead of dumping the raw script."
                ),
                est_cost_usd=0.005,
                est_latency_s=20.0,
                sync=True,
            ))

    # ── Strategy: re-route to a crew with relevant tools ───────────
    seen_crews: set[str] = set()
    for cap_key in inferred:
        cap = _CAPABILITY_MAP[cap_key]
        for crew in cap["crews"]:
            if crew == used_crew or crew in seen_crews:
                continue
            seen_crews.add(crew)
            out.append(Alternative(
                strategy="re_route",
                crew=crew,
                rationale=(
                    f"Detected '{cap_key}' capability in task; "
                    f"{crew!r} crew has matching tools "
                    f"({', '.join(cap['tools'][:2])})."
                ),
                est_cost_usd=0.02,
                est_latency_s=30.0,
                sync=True,
            ))

    # ── Strategy: skill_chain — invoke matching library skill ──────
    # Useful regardless of category — skills are domain-agnostic.
    out.append(Alternative(
        strategy="skill_chain",
        rationale=(
            "Search the skills library for a known approach to this "
            "kind of task. Useful even when no crew has the tools."
        ),
        est_cost_usd=0.01,
        est_latency_s=10.0,
        sync=True,
    ))

    # ── Strategy: escalate model tier (same crew, stronger LLM) ────
    # Only useful for `generic` refusals — the model gave up, but a
    # stronger one might persist. Don't escalate for missing_tool /
    # auth (no model can fix a missing API key).
    if refusal_category in ("generic", "data_unavailable"):
        if used_tier in (None, "budget", "mid", "free"):
            out.append(Alternative(
                strategy="escalate_tier",
                tier="premium",
                rationale=(
                    f"Generic refusal at tier={used_tier!r} — a premium "
                    f"model may persist through the obstacle."
                ),
                est_cost_usd=0.10,
                est_latency_s=60.0,
                sync=True,
            ))

    # ── Augment with semantic search over the tool registry ───────
    # Catches tools whose phrasing doesn't hit any _CAPABILITY_MAP
    # keyword. Dedup by (strategy, tool, crew) so a registry hit that
    # already came in via the keyword path doesn't get double-emitted.
    existing_keys = {(a.strategy, a.tool, a.crew) for a in out}
    for alt in _registry_alternatives(task, used_crew):
        key = (alt.strategy, alt.tool, alt.crew)
        if key not in existing_keys:
            out.append(alt)
            existing_keys.add(key)

    # Sort the runtime strategies by cost (cheapest first) so the
    # loop's budget is spent on most-likely-to-recover paths before
    # expensive ones. forge_queue is appended unconditionally at the
    # end below — it's the always-available fallback.
    out.sort(key=lambda a: (a.est_cost_usd, a.est_latency_s))

    # ── Strategy N (always last): forge_queue ─────────────────────
    # Even when no other strategy applies, file the refusal for the
    # offline skill-forge so the gap eventually closes. This is what
    # makes the system self-improving rather than fail-and-forget.
    out.append(Alternative(
        strategy="forge_queue",
        rationale=(
            "File this gap for the offline skill-forge — same gap "
            "3+ times/week auto-spawns an experiment."
        ),
        est_cost_usd=0.0,
        est_latency_s=0.0,
        sync=False,
    ))

    return out
