"""Capability vocabulary — the bounded namespace of "things tools do".

This file is **GOVERNANCE-GRADE** infrastructure. It is listed in
``TIER_IMMUTABLE`` (see ``app/auto_deployer.py``) and treated with the
same review discipline as ``app/souls/``. The Self-Improver agent
cannot auto-modify this vocabulary; expansion requires human approval.

Why it's bounded
----------------
Surface-keyword matching is the failure mode that produced the May
2026 "Weather Forecast" skill hijacking the Estonia deforestation
request: an unbounded synonym space lets a stale entry steal a
relevant query. A bounded, hand-curated capability vocabulary makes
the discovery problem typed instead of stringly. ``tool_search``
exact-matches against tags first, falls back to embedding similarity
second; the exact-match layer can't drift because the vocabulary is
finite and reviewed.

Why it's separate from tool descriptions
----------------------------------------
A description is for the *LLM* — it explains *how* and *when* to use
a tool. A capability tag is for the *registry* — it answers "is there
a tool for X?" without reading prose. Tags also let multiple tools
declare the same capability (e.g. both ``pdf_compose`` and
``generate_pdf`` declare ``renders-pdf``), which is how the registry
ranks alternatives.

How to add a capability
-----------------------
1. Open a PR. Touching this file requires an explicit human review —
   it is in TIER_IMMUTABLE.
2. Capability names are kebab-case verbs-or-noun-verbs:
   ``renders-pdf`` (not ``pdf`` or ``pdf_renderer``).
3. Place it under the right category. If no category fits, propose a
   new category in the same PR.
4. Add a one-line description. The description is what
   ``/api/cp/tools`` and ``tool_search`` show users.
5. After merge, tools can declare the new tag in their
   ``@register_tool(capabilities=[...])`` decorator.

Stability promise
-----------------
Once a capability tag is in this file, **do not rename or remove it**.
Tools across the codebase reference these strings; renaming silently
breaks discovery. If you must deprecate one, mark it as deprecated
(see ``DEPRECATED_CAPABILITIES``) and migrate tools in a separate PR.
"""
from __future__ import annotations

from typing import Final


# ── Categories ──────────────────────────────────────────────────────
# Each top-level key is a category; values map capability tag → human-
# readable description shown in /api/cp/tools and tool_search results.

CAPABILITIES: Final[dict[str, dict[str, str]]] = {
    # ── data: pulling information from outside the agent ────────────
    "data": {
        "reads-file": "Read content from a file in the workspace.",
        "writes-file": "Write content to a file in the workspace.",
        "reads-attachment": "Read user-provided attachments / uploads.",
        "reads-host-file": "Read files on the host (outside the container).",
        "writes-host-file": "Write files on the host (outside the container).",
        "searches-web": "Search the public web (Google / Tavily / equivalent).",
        "reads-satellite": "Pull satellite imagery / geospatial data (GEE, Sentinel, MODIS).",
        "fetches-geodata": "Fetch curated geospatial datasets via HTTP API.",
        "fetches-finance": "Pull stock / forex / ECB / OECD financial data.",
    },

    # ── knowledge: institutional / curated knowledge ────────────────
    "knowledge": {
        "reads-knowledge-base": "Search the enterprise knowledge base (Mem0/RAG).",
        "reads-philosophy": "Query the philosophy RAG corpus.",
        "reads-fiction": "Query the fiction-inspiration corpus.",
        "reads-wiki": "Read internal wiki pages.",
        "writes-wiki": "Create / edit internal wiki pages.",
        "reads-journal": "Read the experiential journal.",
        "searches-aesthetic-patterns": "Search the aesthetic-pattern catalog.",
        "searches-tensions": "Search the recorded tension graph.",
    },

    # ── memory: agent-scoped + shared belief state ──────────────────
    "memory": {
        "reads-agent-memory": "Read the agent's own memory store.",
        "writes-agent-memory": "Write to the agent's own memory store.",
        "reads-team-belief": "Read the cross-agent shared belief store.",
        "writes-team-belief": "Update the cross-agent shared belief store.",
    },

    # ── compute: running computation in a sandbox ───────────────────
    "compute": {
        "executes-code": "Run Python in the in-process sandbox.",
        "executes-on-host": "Run a command on the host via the bridge.",
        "executes-earth-engine": "Run server-side aggregation against Google Earth Engine.",
        "renders-pdf": "Compose a PDF document from data.",
        "renders-chart": "Render charts / figures (matplotlib, plotly).",
        "renders-document": "Compose prose-shaped documents (DOCX, HTML).",
        "blends-concepts": "Operationalise philosophy+fiction analogical blending.",
        "finds-counter-argument": "Generate a counter-argument via the dialectics graph.",
        "converts-currency": "Convert between currencies / units.",
    },

    # ── delivery: getting artifacts to the user ─────────────────────
    "delivery": {
        "sends-signal": "Send a Signal message (text + optional attachments).",
        "delivers-attachment": "Deliver a file from /app/workspace/output/ to the user.",
    },

    # ── governance: modifying the system itself ─────────────────────
    "governance": {
        "registers-tool": "Forge — register a new sandboxed tool through the audit pipeline.",
        "records-tension": "Record a creative / structural tension in the tension graph.",
        "flags-aesthetic-pattern": "Flag a code/text pattern for the aesthetic library.",
    },
}


# ── Deprecated tags (kept for backward compat, hidden from search) ──
# Move a tag here BEFORE removing it from CAPABILITIES. Tools using it
# get a warning at registry boot and a 1-release window to migrate.
DEPRECATED_CAPABILITIES: Final[dict[str, str]] = {
    # "old-tag": "use 'new-tag' instead",
}


# ── Flat lookup helpers ─────────────────────────────────────────────

def all_capability_tags() -> set[str]:
    """Return the flat set of all valid capability tags."""
    out: set[str] = set()
    for cat in CAPABILITIES.values():
        out.update(cat.keys())
    return out


def category_for(tag: str) -> str | None:
    """Return the category name for a capability tag, or None if unknown."""
    for cat_name, cat in CAPABILITIES.items():
        if tag in cat:
            return cat_name
    return None


def is_known(tag: str) -> bool:
    """True if ``tag`` is in CAPABILITIES (not deprecated)."""
    return tag in all_capability_tags()


def description_for(tag: str) -> str | None:
    """Return the human description for a capability tag."""
    for cat in CAPABILITIES.values():
        if tag in cat:
            return cat[tag]
    return DEPRECATED_CAPABILITIES.get(tag)
