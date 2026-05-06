"""dossier — Investment-grade company dossier subsystem.

The dossier produces a structured, source-attributed view of a company
suitable for investment-grade reporting (10-15 page PDF).  Every field
is typed, every value carries provenance + confidence, and the prose
composition layer can only reproduce facts that are in the dossier.

Pipeline:

    company_ref  ─►  collect (parallel adapters, source priority)
                ─►  CompanyDossier (typed + provenance per field)
                ─►  peer selection (SIC/NAICS + sector intersect)
                ─►  compose (LLM, section-by-section, strict citation)
                ─►  fact-check pass (numeric regex vs. dossier)
                ─►  typeset (ReportLab Platypus, multi-page)
                ─►  PDF artifact + source appendix

Reuses:

* :mod:`app.tools.research_orchestrator` — circuit breakers, parallel
  dispatch, per-call timeouts, partial-streaming progress
* :mod:`app.epistemic.ledger` — every field becomes a Claim with
  Evidence; the source appendix is rendered from the ledger
* :mod:`app.tools.pdf_compose` — primitive matplotlib + reportlab
  imports (we don't reinvent the heavy-import machinery)
* :mod:`app.tool_registry` — every dossier tool registers via
  ``@register_tool`` so it's visible to ``tool_search``
* :mod:`app.crews.registry` — registers as the ``company_dossier`` crew
  so the commander dispatches via the same mechanism as every other crew

The dossier subsystem is **business logic, not safety code** — it is
not in TIER_IMMUTABLE.  Adapters can be added without governance review.
"""
from __future__ import annotations

from app.dossier.schema import (
    Confidence,
    CompanyDossier,
    CompanyRef,
    DossierField,
    FieldStatus,
    FundingRound,
    Source,
    Owner,
    PeerEntry,
)

__all__ = [
    "Confidence",
    "CompanyDossier",
    "CompanyRef",
    "DossierField",
    "FieldStatus",
    "FundingRound",
    "Source",
    "Owner",
    "PeerEntry",
]


# ── Tool-registry side-effect import ─────────────────────────────────
#
# Importing ``app.dossier.tools`` triggers the ``@register_tool``
# decorator that lands ``build_company_dossier`` in the global
# tool registry.  We do it here (rather than in ``main.py``) so the
# tool is visible whenever any part of the system imports the dossier
# package — matching the pattern used by ``app.tools.pdf_compose``
# and ``app.tools.financial_tools``.
#
# Defensive: a partial install (no crewai, no pydantic) must not break
# the import of the schema types above.  The tools module's
# registration call is itself wrapped — but we also catch here in case
# of any other unforeseen error.

import logging as _logging

try:
    from app.dossier import tools as _tools_module  # noqa: F401
except Exception as _exc:  # noqa: BLE001
    _logging.getLogger(__name__).debug(
        "app.dossier.tools side-effect import failed: %s — "
        "build_company_dossier will not appear in the tool registry "
        "this session", _exc,
    )
