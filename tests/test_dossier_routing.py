"""Tests for the commander's fast-path routing of dossier queries.

Why this test exists
====================
The fast-path regex chain in ``app.agents.commander.routing`` matches in
ORDER and returns on first match.  Before the dossier work, the
``financial`` rule's ``\\binvestment\\b`` would capture every
"investment-grade" / "investor report" query and steer it to the
financial crew (which can't produce structured dossiers).

These tests guard against the regression of "investment-grade overview
of X" being stolen by the financial rule.  They also assert that
unambiguous financial queries still route to the financial crew.
"""
from __future__ import annotations

import pytest

from app.agents.commander.routing import _try_fast_route


# ══════════════════════════════════════════════════════════════════════
# Dossier-shaped queries → company_dossier
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("query", [
    "Build me an investment-grade overview of Spotify (SPOT)",
    "Investment grade report on Apple",
    "Investment-grade review of Tesla",
    "Build a dossier for Stripe",
    "Spotify dossier please",
    "Due diligence on Tesla",
    "Due-diligence pack on Wolt",
    "Company profile for Nvidia",
    "Company review of Spotify",
    "Company report on Microsoft",
    "Investor brief on Apple",
    "Investor report on Stripe",
    "Investment overview of Wolt",
])
def test_dossier_query_routes_to_company_dossier(query):
    result = _try_fast_route(query, has_attachments=False)
    assert result is not None, f"no fast-path match for {query!r}"
    assert result[0]["crew"] == "company_dossier", (
        f"expected company_dossier for {query!r}, got {result[0]['crew']!r}"
    )


# ══════════════════════════════════════════════════════════════════════
# Pure-financial queries → financial (must not be stolen by dossier)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("query", [
    "Compute DCF for Tesla",
    "Stock price of MSFT",
    "Show me Spotify's earnings last quarter",
    "Review my investment portfolio",
    "What's MSFT's market cap?",
    "Show SEC filings for AAPL",
    "Valuation of Tesla using DCF",
])
def test_financial_query_still_routes_to_financial(query):
    result = _try_fast_route(query, has_attachments=False)
    # Either matches financial directly, or falls through to LLM
    # routing (no_match).  Critically: must NOT route to
    # company_dossier — that would be a routing regression.
    if result is not None:
        assert result[0]["crew"] != "company_dossier", (
            f"financial query incorrectly captured by company_dossier: {query!r}"
        )
        # Most should match financial directly.
        assert result[0]["crew"] == "financial", (
            f"expected financial (or no-match), got {result[0]['crew']!r} for {query!r}"
        )
