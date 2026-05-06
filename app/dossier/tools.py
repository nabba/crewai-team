"""tools — CrewAI tool wrappers + registry registration.

Exposes :func:`build_dossier` to agents through two paths:

1. **Direct tool**: ``DossierBuildTool`` — a CrewAI ``BaseTool`` an
   agent can call inline.  Use case: the financial_analyst is mid-
   conversation and the user says "do a quick dossier on $SPOT" — the
   agent invokes the tool with the company query.

2. **Registry**: ``@register_tool`` decoration in
   :func:`_register_with_tool_registry` — surfaces under
   ``tool_search`` and the ``/api/cp/tools`` panel.

The crew (``app.crews.dossier_crew.DossierCrew``) is the higher-level
entry point invoked by the commander when the request is *primarily*
a dossier build; this tool is for agents that want to incorporate a
dossier as a sub-step of a larger reasoning chain.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Type

logger = logging.getLogger(__name__)


def create_dossier_tools(agent_id: str = "dossier") -> list:
    """Factory matching the existing tool factory shape (e.g.
    ``create_financial_tools``, ``create_pdf_tools``).

    Returns a list of CrewAI tools — currently just the build tool,
    but stubbed as a list so future additions (e.g. a tool that
    inspects an existing dossier in the ledger) slot in cleanly.
    """
    try:
        from crewai.tools import BaseTool
        from pydantic import BaseModel, Field
    except ImportError:
        return []

    class _BuildInput(BaseModel):
        query: str = Field(
            description=(
                "Natural-language description of the company. "
                "Example: 'Spotify Technology (SPOT)' or "
                "'Build a dossier for Tony's Chocolonely'."
            ),
        )
        include_peers: bool = Field(
            default=True,
            description=(
                "When True (default), build a peer set and include the "
                "comparator section.  Set False for a faster build "
                "without peer comparison."
            ),
        )
        max_peers: int = Field(
            default=5,
            description="Cap on peer dossiers collected (default 5).",
        )

    class DossierBuildTool(BaseTool):
        name: str = "build_company_dossier"
        description: str = (
            "Build an investment-grade company dossier (10-15 page PDF) "
            "with structured fields, source citations, and a comparator "
            "section.  USE THIS instead of trying to assemble company "
            "research with web_search/firecrawl — this tool runs the "
            "full collection pipeline (SEC EDGAR, Wikidata, Wikipedia, "
            "yfinance, Companies House when configured), reconciles "
            "conflicts, and produces a PDF in /app/workspace/output/.\n\n"
            "Input: company query string.\n"
            "Output: JSON with the PDF path, coverage stats, and a "
            "human-readable summary.  Pair with `signal_send_attachment` "
            "to deliver the PDF over Signal."
        )
        args_schema: Type[BaseModel] = _BuildInput

        def _run(self, query: str, include_peers: bool = True,
                 max_peers: int = 5) -> str:
            from app.dossier.pipeline import build_dossier
            try:
                build = build_dossier(
                    query=query,
                    include_peers=include_peers,
                    max_peers=max_peers,
                )
            except ValueError as exc:
                return json.dumps({"ok": False, "error": str(exc)})
            except Exception as exc:
                logger.exception("DossierBuildTool: build crashed")
                return json.dumps({
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            return json.dumps({
                "ok": True,
                "pdf_path": str(build.pdf_path),
                "company": build.ref.name,
                "ticker": build.ref.ticker,
                "fields_filled": build.dossier.known_field_count(),
                "fields_total": build.dossier.total_field_count(),
                "coverage_pct": round(build.dossier.coverage_pct() * 100, 1),
                "peers_count": len(build.peers),
                "fact_check_warnings": build.warnings_total,
                "sections": [s.title for s in build.report.sections],
                "summary": build.summary(),
            })

    return [DossierBuildTool()]


# ── Tool-registry registration (Phase 1b/4 pattern) ──────────────────


_DOSSIER_DESCRIPTION = (
    "Build an investment-grade company dossier (10-15 page PDF). "
    "Runs the deterministic dossier pipeline: identity resolution, "
    "parallel multi-source data collection (SEC EDGAR, Wikidata, "
    "yfinance, Companies House), reconciliation across sources, "
    "section-by-section composition with strict citation discipline, "
    "fact-check pass on numeric claims, and ReportLab Platypus "
    "typesetting.\n\n"
    "Use for: company research where the deliverable is a structured "
    "report (due diligence, M&A targets, portfolio reviews).\n"
    "Don't use for: ad-hoc analyst chat (use the financial crew); "
    "free-form research matrices (use research_orchestrator)."
)


def _register_with_tool_registry() -> None:
    """Idempotent registration via ``@register_tool``.

    Mirrors the pattern from :mod:`app.tools.pdf_compose` /
    :mod:`app.tools.financial_tools`: the decorator runs once at
    module import time; subsequent imports are cached by ``sys.modules``.
    """
    try:
        from app.tool_registry import Lifecycle, Tier, register_tool
    except ImportError:
        return

    try:
        from pydantic import BaseModel, Field
    except ImportError:
        return

    class _DossierToolArgs(BaseModel):
        query: str = Field(
            description="Natural-language company identity (name + ticker).",
        )
        include_peers: bool = Field(default=True)
        max_peers: int = Field(default=5)

    @register_tool(
        name="build_company_dossier",
        capabilities=["renders-pdf", "renders-document", "fetches-finance",
                      "searches-web"],
        description=_DOSSIER_DESCRIPTION,
        args_schema=_DossierToolArgs,
        tier=Tier.PRODUCTION,
        lifecycle=Lifecycle.SINGLETON,
        guard=lambda: True,
    )
    def _dossier_factory(agent_id: str = "researcher"):
        # Defer instantiation until the registry asks (matches
        # pdf_compose's _pdf_compose_registry_factory shape).
        return create_dossier_tools(agent_id=agent_id)[0]


# Run the registration at module import time so any process that
# imports ``app.dossier`` gets the tool registered.  ``app.dossier``'s
# package-init does NOT pull this module on its own — the registration
# is opt-in via importing ``app.dossier.tools`` (or via the crew
# registry, which depends on the tool registry being independent).
_register_with_tool_registry()
