"""sec_edgar — SEC EDGAR company-facts adapter.

For public US companies, SEC EDGAR is the highest-confidence financial
source on the open internet.  It exposes XBRL-tagged facts via the
``data.sec.gov/api/xbrl/companyfacts`` endpoint (no key, no rate-limit
auth — only User-Agent identification required).

What this adapter fills
=======================
* ``ticker``          (via the company-tickers index, ref enrichment)
* ``legal_name``       — entity name from companyfacts
* ``revenue_usd``      — most recent fiscal-year Revenues (us-gaap:Revenues)
* ``net_income_usd``   — most recent fiscal-year NetIncomeLoss
* ``ebitda_usd``       — derived from net income + interest + taxes +
                          depreciation when components are present;
                          marked ESTIMATED if any leg is missing
* ``employee_count``   — EntityCommonStockSharesOutstanding-adjacent
                          ``EntityNumberOfEmployees`` annual
* ``fiscal_year_end``  — ``EntityFiscalYearEndDate``
* ``industry_codes``   — SIC code via /submissions/

Comparison with ``financial_tools.SECFilingsTool``
==================================================
The existing tool returns links + filing dates; this adapter parses
structured facts.  Coexists rather than replaces — the LLM-driven
``financial_tools`` is for ad-hoc filing inspection during analyst
chat; the dossier adapter is for structured one-shot collection.
The two tools share the User-Agent helper from ``financial_tools``
to avoid divergent SEC contact strings.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from app.dossier.adapters._base import (
    DossierAdapterResult,
    FieldUpdate,
    cache_lookup,
    cache_store,
    http_get_json,
    register_adapter,
)
from app.dossier.schema import CompanyRef, Confidence

logger = logging.getLogger(__name__)


_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_BASE = "https://data.sec.gov/submissions/"
_FACTS_BASE = "https://data.sec.gov/api/xbrl/companyfacts/"
_TIMEOUT_SECS = 15

# US state and territory codes used by SEC's ``stateOfIncorporation``.
# SEC also uses non-US issuer codes (N4 = Luxembourg, G4 = Guernsey,
# X1 = United Kingdom, …); we treat those as "foreign issuer" rather
# than appending ", United States".
_US_STATE_CODES: frozenset[str] = frozenset({
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "GU", "VI", "AS", "MP",  # territories
})

# Cache the ticker → CIK map for the lifetime of the process.  Refreshed
# on the first dossier run; SEC publishes daily but we don't need to be
# that fresh inside a single dossier session.
_TICKER_INDEX: dict[str, str] | None = None


def _user_agent() -> str:
    """Reuse the existing ``financial_tools`` SEC user-agent helper.

    The SEC requires the UA to identify the requester; the existing
    settings field ``sec_edgar_user_agent`` is the canonical source.
    Fall back to a default if the settings module isn't wired up
    (tests, slim envs).
    """
    try:
        from app.config import get_settings
        return get_settings().sec_edgar_user_agent  # type: ignore[attr-defined]
    except Exception:
        return "BotArmy-Dossier/1.0 (contact@example.com)"


def _load_ticker_index() -> dict[str, str]:
    """ticker (uppercase) → 10-digit zero-padded CIK string."""
    global _TICKER_INDEX
    if _TICKER_INDEX is not None:
        return _TICKER_INDEX
    body = http_get_json(
        _TICKERS_URL,
        headers={"User-Agent": _user_agent()},
        timeout=_TIMEOUT_SECS,
    )
    out: dict[str, str] = {}
    if isinstance(body, dict):
        for entry in body.values():
            if not isinstance(entry, dict):
                continue
            ticker = (entry.get("ticker") or "").upper()
            cik = entry.get("cik_str") or entry.get("cik")
            if ticker and cik is not None:
                out[ticker] = str(cik).zfill(10)
    _TICKER_INDEX = out
    return out


def _resolve_cik(ref: CompanyRef) -> str:
    """Pick a CIK from the ref's identifiers.  Empty string when none."""
    if ref.ticker:
        idx = _load_ticker_index()
        return idx.get(ref.ticker.upper(), "")
    return ""


def _most_recent_annual(units: dict, prefer_unit: str = "USD") -> dict | None:
    """Pick the most recent annual (FY) datapoint from a companyfacts
    ``units`` block.

    SEC reports both 10-K (annual) and 10-Q (quarterly) facts under
    the same concept.  We filter to ``fp == "FY"`` and ``form ==
    "10-K"`` so we get the audited annual.  Falls back to the most
    recent FY in any form if no 10-K is present.
    """
    series = units.get(prefer_unit) or []
    if not series:
        return None
    fy_only = [d for d in series if d.get("fp") == "FY"]
    if not fy_only:
        return None
    audited = [d for d in fy_only if d.get("form") == "10-K"]
    candidates = audited or fy_only
    candidates.sort(
        key=lambda d: (d.get("end") or "", d.get("filed") or ""),
        reverse=True,
    )
    return candidates[0]


def _parse_date(s: str) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).date()
    except ValueError:
        try:
            return date.fromisoformat(s[:10])
        except ValueError:
            return None


class SecEdgarAdapter:
    """Public-company financials via SEC EDGAR companyfacts."""

    name = "sec_edgar"
    priority = 100  # regulator-grade, highest priority

    def can_collect(self, ref: CompanyRef) -> bool:
        # Need a ticker (or eventually a direct CIK).  We only support
        # ticker-based identity today — extending to name-based search
        # is possible via /search-index but is noisier.
        return bool(ref.ticker)

    def is_configured(self) -> bool:
        try:
            import requests  # noqa: F401
            return True
        except ImportError:
            return False

    def collect(self, ref: CompanyRef) -> DossierAdapterResult:
        cache_key = ref.ticker.upper() or ref.name
        cached = cache_lookup(self.name, cache_key)
        if cached is not None:
            return cached
        result = self._collect_uncached(ref)
        cache_store(self.name, cache_key, result)
        return result

    def _collect_uncached(self, ref: CompanyRef) -> DossierAdapterResult:
        cik = _resolve_cik(ref)
        if not cik:
            return DossierAdapterResult(
                adapter_name=self.name,
                error=f"no CIK for ticker {ref.ticker!r}",
            )

        ua_headers = {"User-Agent": _user_agent(),
                      "Accept": "application/json"}

        # ── companyfacts (XBRL) ─────────────────────────────────────
        facts_body = http_get_json(
            f"{_FACTS_BASE}CIK{cik}.json",
            headers=ua_headers,
            timeout=_TIMEOUT_SECS,
        )
        if not isinstance(facts_body, dict):
            return DossierAdapterResult(
                adapter_name=self.name,
                error=f"companyfacts fetch failed for CIK {cik}",
            )

        cite_url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar?"
            f"action=getcompany&CIK={cik}&type=10-K"
        )
        result = DossierAdapterResult(
            adapter_name=self.name, base_url=cite_url,
        )

        # ── identity ────────────────────────────────────────────────
        entity_name = (facts_body.get("entityName") or "").strip()
        if entity_name:
            result.fields.append(FieldUpdate(
                field_name="legal_name", value=entity_name,
                confidence=Confidence.EXACT,
                note=f"SEC EDGAR CIK {cik}",
            ))

        us_gaap = (facts_body.get("facts") or {}).get("us-gaap") or {}
        dei = (facts_body.get("facts") or {}).get("dei") or {}

        # ── revenue ─────────────────────────────────────────────────
        # Try Revenues, then RevenueFromContractWithCustomerExcludingAssessedTax
        # (the post-ASC-606 disclosure), then SalesRevenueNet for older
        # filings.  All in USD.
        for concept in (
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
        ):
            block = us_gaap.get(concept)
            if not block:
                continue
            datum = _most_recent_annual(block.get("units") or {})
            if not datum:
                continue
            result.fields.append(FieldUpdate(
                field_name="revenue_usd", value=float(datum.get("val") or 0),
                confidence=Confidence.EXACT,
                as_of=_parse_date(datum.get("end")),
                note=f"{concept}, FY ending {datum.get('end')}, {datum.get('form')}",
            ))
            break

        # ── net income ──────────────────────────────────────────────
        for concept in ("NetIncomeLoss", "ProfitLoss"):
            block = us_gaap.get(concept)
            if not block:
                continue
            datum = _most_recent_annual(block.get("units") or {})
            if not datum:
                continue
            result.fields.append(FieldUpdate(
                field_name="net_income_usd",
                value=float(datum.get("val") or 0),
                confidence=Confidence.EXACT,
                as_of=_parse_date(datum.get("end")),
                note=f"{concept}, FY ending {datum.get('end')}",
            ))
            break

        # ── EBITDA (derived) ────────────────────────────────────────
        # Companies don't report EBITDA in GAAP — we derive from
        # NetIncome + InterestExpense + IncomeTaxExpense +
        # DepreciationAndAmortization when all components are present.
        ebitda = self._derive_ebitda(us_gaap)
        if ebitda is not None:
            result.fields.append(FieldUpdate(
                field_name="ebitda_usd", value=ebitda["value"],
                confidence=Confidence.HIGH,  # derived but auditable
                as_of=ebitda.get("as_of"),
                note=("derived: NetIncome + InterestExpense + "
                      "IncomeTaxExpense + DepreciationAndAmortization"),
            ))

        # ── employees ───────────────────────────────────────────────
        emp_block = dei.get("EntityCommonStockSharesOutstanding") and None
        emp_block = dei.get("EntityNumberOfEmployees")
        if emp_block:
            datum = _most_recent_annual(emp_block.get("units") or {},
                                        prefer_unit="pure")
            if datum:
                result.fields.append(FieldUpdate(
                    field_name="employee_count", value=int(datum.get("val") or 0),
                    confidence=Confidence.EXACT,
                    as_of=_parse_date(datum.get("end")),
                    note="EntityNumberOfEmployees, 10-K disclosure",
                ))

        # ── fiscal year end ─────────────────────────────────────────
        fy_block = dei.get("CurrentFiscalYearEndDate")
        if fy_block:
            datum = _most_recent_annual(fy_block.get("units") or {},
                                        prefer_unit="USD")
            # CurrentFiscalYearEndDate is reported in MM-DD format under
            # the 'pure' unit historically.  Try both.
            if not datum:
                datum = _most_recent_annual(fy_block.get("units") or {},
                                            prefer_unit="pure")
            if datum and datum.get("val"):
                result.fields.append(FieldUpdate(
                    field_name="fiscal_year_end", value=str(datum.get("val")),
                    confidence=Confidence.EXACT,
                ))

        # ── industry (SIC) via submissions ──────────────────────────
        submissions = http_get_json(
            f"{_SUBMISSIONS_BASE}CIK{cik}.json",
            headers=ua_headers, timeout=_TIMEOUT_SECS,
        )
        if isinstance(submissions, dict):
            sic = submissions.get("sic")
            sic_desc = submissions.get("sicDescription") or ""
            if sic:
                code = f"SIC {sic}"
                if sic_desc:
                    code = f"{code} ({sic_desc})"
                result.fields.append(FieldUpdate(
                    field_name="industry_codes", value=(code,),
                    confidence=Confidence.EXACT,
                ))
            # State of incorporation.  SEC's ``stateOfIncorporation``
            # uses two-letter codes for US states (CA, NY, DE, …) but
            # also issuer-coded values for foreign filers (N4 =
            # Luxembourg, G4 = Guernsey, 1A = Switzerland, etc).
            #
            # We only emit a value when the code is a real US state —
            # for foreign issuers we defer to Wikidata's country field,
            # which renders to a real country name like "Luxembourg".
            # SEC has higher priority than Wikidata, so writing a junk
            # foreign-code string would overwrite the cleaner Wikidata
            # answer and force the composer to render it.
            state = submissions.get("stateOfIncorporation")
            if state and state in _US_STATE_CODES:
                result.fields.append(FieldUpdate(
                    field_name="incorporated_in",
                    value=f"{state}, United States",
                    confidence=Confidence.EXACT,
                ))

        return result

    def _derive_ebitda(self, us_gaap: dict) -> dict | None:
        """Compute EBITDA from GAAP components.

        Returns ``{"value": float, "as_of": date | None}`` when all
        four legs are available for the same fiscal year, else None.
        """
        legs = (
            ("NetIncomeLoss",), ("InterestExpense",),
            ("IncomeTaxExpenseBenefit",),
            ("DepreciationAndAmortization", "DepreciationDepletionAndAmortization"),
        )
        components: list[float] = []
        as_of: date | None = None
        for alts in legs:
            datum = None
            for concept in alts:
                block = us_gaap.get(concept)
                if not block:
                    continue
                datum = _most_recent_annual(block.get("units") or {})
                if datum:
                    break
            if not datum:
                return None  # missing leg → can't derive
            components.append(float(datum.get("val") or 0))
            d = _parse_date(datum.get("end"))
            # Components must agree on fiscal year end.  If they don't,
            # bail out — different years would mix periods.
            if as_of and d and as_of != d:
                return None
            as_of = as_of or d
        return {"value": sum(components), "as_of": as_of}


def install() -> None:
    register_adapter(SecEdgarAdapter())
