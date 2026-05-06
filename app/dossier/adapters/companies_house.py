"""companies_house — UK Companies House public-records adapter.

Companies House is the UK statutory registry: every UK-incorporated
company files annual accounts (revenue / EBITDA via abridged or full
accounts) plus a People-with-Significant-Control (PSC) list which is
the closest free analogue to a cap table.

Authentication
==============
Companies House requires API key registration (free).  Set
``COMPANIES_HOUSE_API_KEY`` in the environment.  HTTP Basic auth with
the API key as the username and an empty password.

Coverage
========
* All UK-incorporated entities (Ltd, PLC, LLP, CIC).
* Filings are statutory but VARY in detail: full accounts include a
  P&L, abridged accounts may only include a balance sheet.  Filing
  history shows the type — we mark fields by what the latest filing
  actually exposed.
* Charges (debt liens) and PSC are always present for active entities.

Limitations
===========
* Salary data is NOT available — UK private companies don't publish
  per-employee compensation.
* The API rate-limits aggressively (600 req / 5 min).  We make at most
  4 calls per company (search → profile → filing-history → PSC).
"""
from __future__ import annotations

import base64
import logging
import os
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
from app.dossier.schema import CompanyRef, Confidence, Owner

logger = logging.getLogger(__name__)


_API_BASE = "https://api.company-information.service.gov.uk"
_TIMEOUT_SECS = 12


def _auth_header() -> dict[str, str] | None:
    key = os.environ.get("COMPANIES_HOUSE_API_KEY", "").strip()
    if not key:
        return None
    creds = base64.b64encode(f"{key}:".encode("ascii")).decode("ascii")
    return {"Authorization": f"Basic {creds}", "Accept": "application/json"}


def _resolve_company_number(ref: CompanyRef) -> str | None:
    """Either use the ref's pre-known number or search by name."""
    if ref.companies_house_number:
        return ref.companies_house_number

    headers = _auth_header()
    if headers is None or not ref.name:
        return None

    body = http_get_json(
        f"{_API_BASE}/search/companies",
        params={"q": ref.name, "items_per_page": 1},
        headers=headers, timeout=_TIMEOUT_SECS,
    )
    if not isinstance(body, dict):
        return None
    items = body.get("items") or []
    if not items:
        return None
    return (items[0] or {}).get("company_number") or None


def _parse_iso_date(s: str) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).date()
    except ValueError:
        try:
            return date.fromisoformat(s[:10])
        except ValueError:
            return None


class CompaniesHouseAdapter:
    """UK private/public company filings."""

    name = "companies_house"
    priority = 95  # near-regulator grade for UK; just below SEC for US

    def can_collect(self, ref: CompanyRef) -> bool:
        if not (ref.companies_house_number or ref.name):
            return False
        # Soft hint: only attempt if country looks UK-shaped.  We don't
        # hard-fail on missing country — the search endpoint won't
        # match a non-UK entity anyway, so the adapter degrades to
        # "no result" cleanly.
        return True

    def is_configured(self) -> bool:
        return bool(os.environ.get("COMPANIES_HOUSE_API_KEY", "").strip())

    def collect(self, ref: CompanyRef) -> DossierAdapterResult:
        identity = ref.companies_house_number or ref.name
        cached = cache_lookup(self.name, identity)
        if cached is not None:
            return cached
        result = self._collect_uncached(ref)
        cache_store(self.name, identity, result)
        return result

    def _collect_uncached(self, ref: CompanyRef) -> DossierAdapterResult:
        headers = _auth_header()
        if headers is None:
            return DossierAdapterResult(
                adapter_name=self.name,
                error="COMPANIES_HOUSE_API_KEY not set",
            )

        company_number = _resolve_company_number(ref)
        if not company_number:
            return DossierAdapterResult(
                adapter_name=self.name,
                error=f"no Companies House match for {ref.name!r}",
            )

        cite_url = f"https://find-and-update.company-information.service.gov.uk/company/{company_number}"
        result = DossierAdapterResult(
            adapter_name=self.name, base_url=cite_url,
        )
        result.ref_enrichment.setdefault("companies_house_number", company_number)

        # ── company profile ────────────────────────────────────────
        profile = http_get_json(
            f"{_API_BASE}/company/{company_number}",
            headers=headers, timeout=_TIMEOUT_SECS,
        )
        if isinstance(profile, dict):
            self._fill_from_profile(profile, result)

        # ── PSC (people with significant control) ────────────────
        pscs = http_get_json(
            f"{_API_BASE}/company/{company_number}/persons-with-significant-control",
            headers=headers, timeout=_TIMEOUT_SECS,
        )
        if isinstance(pscs, dict):
            owners = self._parse_pscs(pscs)
            if owners:
                result.fields.append(FieldUpdate(
                    field_name="owners", value=owners,
                    confidence=Confidence.EXACT,
                    note="Companies House PSC register",
                ))

        return result

    # ── helpers ──────────────────────────────────────────────────────

    def _fill_from_profile(self, profile: dict[str, Any],
                           result: DossierAdapterResult) -> None:
        legal_name = (profile.get("company_name") or "").strip()
        if legal_name:
            result.fields.append(FieldUpdate(
                field_name="legal_name", value=legal_name,
                confidence=Confidence.EXACT,
            ))

        founded = _parse_iso_date(profile.get("date_of_creation") or "")
        if founded:
            result.fields.append(FieldUpdate(
                field_name="founded_on", value=founded,
                confidence=Confidence.EXACT, as_of=founded,
                note="Companies House date_of_creation",
            ))

        # Registered office address — closest analogue to "headquarters"
        # for UK-only entities.  Multinationals often have a different
        # de facto HQ; mark MEDIUM so a Wikidata "headquarters" value
        # could win on equal priority (it won't — Companies House has
        # higher source priority — but the confidence reflects reality).
        addr = profile.get("registered_office_address") or {}
        bits = [
            addr.get("address_line_1"), addr.get("locality"),
            addr.get("region"), addr.get("postal_code"),
            addr.get("country"),
        ]
        addr_str = ", ".join(b for b in bits if b)
        if addr_str:
            result.fields.append(FieldUpdate(
                field_name="headquarters", value=addr_str,
                confidence=Confidence.MEDIUM,
                note="Companies House registered office",
            ))

        # SIC codes — the registry stores up to 4.
        sic = profile.get("sic_codes") or []
        if sic:
            result.fields.append(FieldUpdate(
                field_name="industry_codes",
                value=tuple(f"UK SIC {c}" for c in sic),
                confidence=Confidence.EXACT,
            ))

        # Country of incorporation — Companies House is by definition UK,
        # but the jurisdiction sub-field distinguishes England-and-Wales,
        # Scotland, Northern Ireland.
        juris = (profile.get("jurisdiction") or "").replace("-", " ").title()
        if juris:
            result.fields.append(FieldUpdate(
                field_name="incorporated_in", value=f"{juris}, United Kingdom",
                confidence=Confidence.EXACT,
            ))

    def _parse_pscs(self, body: dict[str, Any]) -> tuple[Owner, ...]:
        items = body.get("items") or []
        out: list[Owner] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("ceased_on"):
                continue  # skip historical PSCs
            name = (item.get("name") or "").strip()
            if not name:
                continue
            kind_raw = item.get("kind") or ""
            if "individual" in kind_raw:
                kind = "individual"
            elif "corporate" in kind_raw or "legal" in kind_raw:
                kind = "strategic"
            else:
                kind = "individual"
            # Ownership pct is stored as an enum-like list of "natures
            # of control" (e.g. "ownership-of-shares-50-to-75-percent").
            pct = self._extract_pct(item.get("natures_of_control") or [])
            out.append(Owner(name=name, kind=kind, pct_ownership=pct,
                             note="; ".join(item.get("natures_of_control") or [])))
        return tuple(out)

    @staticmethod
    def _extract_pct(natures: list[str]) -> float | None:
        """Map a Companies House control-nature string to a midpoint pct."""
        # The registry encodes ownership in bands; we pick the midpoint
        # of whichever band is mentioned and let downstream consumers
        # treat as a soft estimate.  No band → None (kind already stamped).
        bands = {
            "ownership-of-shares-25-to-50-percent": 0.375,
            "ownership-of-shares-50-to-75-percent": 0.625,
            "ownership-of-shares-75-to-100-percent": 0.875,
        }
        for n in natures:
            for key, pct in bands.items():
                if key in n:
                    return pct
        return None


def install() -> None:
    register_adapter(CompaniesHouseAdapter())
