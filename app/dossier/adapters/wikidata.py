"""wikidata — Wikidata SPARQL adapter.

Wikidata is the dossier's identity-resolution backbone: given just a
company name, it returns ticker, ISIN, country of HQ, founding date,
founders, official website, industry codes, and links to Wikipedia.
The IDs unlock downstream adapters (SEC EDGAR via ticker, Companies
House via the registry-number property, etc.).

No API key, no rate-limit auth.  Wikidata SPARQL has a soft 60s
per-query budget — we use very narrow queries that return in <2s.

Limitations
-----------
* Coverage favours large / publicly listed / culturally prominent
  companies.  Long-tail private SMBs are typically absent.
* Property values can be slightly stale (community-edited).  We mark
  Wikidata-sourced data as MEDIUM confidence so the merge layer
  prefers regulatory sources when available.
* Founder lists may include co-founders who later left; Wikidata
  doesn't flag exits, so the dossier inherits that limitation.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any
from urllib.parse import urlparse

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


_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
_SEARCH_ENDPOINT = "https://www.wikidata.org/w/api.php"
_USER_AGENT = "BotArmy-Dossier/1.0 (https://github.com/anthropics/claude-code; contact@example.com)"
_TIMEOUT_SECS = 15


# Wikidata properties we extract.  Documented inline so future-me can
# add fields without re-reading Wikidata's schema docs.
#
#   P31   instance of                    (used to detect business_entity)
#   P112  founder
#   P571  inception
#   P159  headquarters location
#   P17   country
#   P452  industry
#   P249  ticker symbol
#   P946  ISIN
#   P856  official website
#   P1320 OpenCorporates ID              (used to resolve OpenCorporates)
#   P5531 SEC Central Index Key (CIK)
#   P2581 BabelNet ID — ignored
#   P5446 Companies House company number
#   P2284 market capitalization (often stale; we don't pull this)


_QUERY_BY_QID = """
SELECT DISTINCT ?company ?companyLabel ?inception ?countryLabel ?hqLabel
       ?industryLabel ?ticker ?isin ?website ?cik ?ch_number ?occ_id
WHERE {
  VALUES ?company { wd:%(qid)s }
  OPTIONAL { ?company wdt:P571 ?inception . }
  OPTIONAL { ?company wdt:P17 ?country . }
  OPTIONAL { ?company wdt:P159 ?hq . }
  OPTIONAL { ?company wdt:P452 ?industry . }
  OPTIONAL { ?company wdt:P249 ?ticker . }
  OPTIONAL { ?company wdt:P946 ?isin . }
  OPTIONAL { ?company wdt:P856 ?website . }
  OPTIONAL { ?company wdt:P5531 ?cik . }
  OPTIONAL { ?company wdt:P5446 ?ch_number . }
  OPTIONAL { ?company wdt:P1320 ?occ_id . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 1
"""


_QUERY_FOUNDERS = """
SELECT ?founderLabel WHERE {
  wd:%(qid)s wdt:P112 ?founder .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""


def _resolve_qid(name: str) -> str:
    """Find the most likely company QID for ``name``.

    Uses Wikidata's ``wbsearchentities`` action — designed for exactly
    this use case (aliases, redirects, fuzzy match).  We pick the
    first hit whose description suggests a company / business / firm
    to filter out homonyms (the city of Spotify, anyone? — but
    extending the search returns clearer top hits when we filter).
    """
    if not name:
        return ""
    body = http_get_json(
        _SEARCH_ENDPOINT,
        params={
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "type": "item",
            "format": "json",
            "limit": 5,
        },
        headers={"User-Agent": _USER_AGENT},
        timeout=_TIMEOUT_SECS,
    )
    if not isinstance(body, dict):
        return ""
    candidates = body.get("search") or []
    # Heuristic: prefer descriptions mentioning "company", "business",
    # "corporation", "firm", "service" — falls back to the top hit.
    company_keywords = (
        "company", "corporation", "business", "firm", "enterprise",
        "service", "platform", "publisher", "manufacturer", "bank",
        "subsidiary",
    )
    for c in candidates:
        desc = (c.get("description") or "").lower()
        if any(k in desc for k in company_keywords):
            return c.get("id") or ""
    if candidates:
        return candidates[0].get("id") or ""
    return ""


def _run_sparql(query: str) -> list[dict[str, Any]]:
    """Run a SPARQL query and return the bindings.  Empty list on failure."""
    body = http_get_json(
        _SPARQL_ENDPOINT,
        params={"query": query, "format": "json"},
        headers={"User-Agent": _USER_AGENT, "Accept": "application/sparql-results+json"},
        timeout=_TIMEOUT_SECS,
    )
    if not isinstance(body, dict):
        return []
    return body.get("results", {}).get("bindings", []) or []


def _binding(b: dict, key: str) -> str:
    return ((b.get(key) or {}).get("value") or "").strip()


def _qid_from_uri(uri: str) -> str:
    """Extract a wikidata QID from an entity URI."""
    if not uri:
        return ""
    return uri.rsplit("/", 1)[-1] if "/" in uri else uri


def _parse_inception(raw: str) -> date | None:
    """Wikidata returns ISO-8601 timestamps; we keep just the date part."""
    if not raw:
        return None
    # Strip BC (negative) values — they break date parsing and aren't
    # plausible for any real company.
    if raw.startswith("-"):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            return None


def _normalise_website(url: str) -> str:
    if not url:
        return ""
    try:
        return urlparse(url).geturl()
    except Exception:
        return url


class WikidataAdapter:
    """Adapter implementation."""

    name = "wikidata"
    priority = 60  # higher than generic web; lower than regulators

    def can_collect(self, ref: CompanyRef) -> bool:
        # We can attempt a name lookup if there's any name at all.
        return bool(ref.name) or bool(ref.wikidata_id)

    def is_configured(self) -> bool:
        # Wikidata SPARQL is open; just needs ``requests`` installed.
        try:
            import requests  # noqa: F401
            return True
        except ImportError:
            return False

    def collect(self, ref: CompanyRef) -> DossierAdapterResult:
        identity = ref.wikidata_id or ref.name
        cached = cache_lookup(self.name, identity)
        if cached is not None:
            return cached

        result = self._collect_uncached(ref)
        cache_store(self.name, identity, result)
        return result

    def _collect_uncached(self, ref: CompanyRef) -> DossierAdapterResult:
        # Step 1 — resolve QID.  Either we already have one in the ref
        # (from a prior enrichment pass) or we look it up via the
        # search-entity endpoint, which handles aliases and exact-vs-
        # fuzzy match much better than direct ``rdfs:label`` matching.
        qid = ref.wikidata_id or _resolve_qid(ref.name)
        if not qid:
            return DossierAdapterResult(
                adapter_name=self.name,
                error=f"no Wikidata entity found for {ref.name!r}",
            )

        # Step 2 — fetch the structured facts for that QID.
        bindings = _run_sparql(_QUERY_BY_QID % {"qid": qid})
        if not bindings:
            return DossierAdapterResult(
                adapter_name=self.name,
                error=f"Wikidata QID {qid} returned no facts",
            )

        b = bindings[0]
        qid = _qid_from_uri(_binding(b, "company"))
        cite_url = f"https://www.wikidata.org/wiki/{qid}" if qid else ""

        result = DossierAdapterResult(
            adapter_name=self.name, base_url=cite_url,
        )

        # ── ref enrichment ───────────────────────────────────────────
        if qid and not ref.wikidata_id:
            result.ref_enrichment["wikidata_id"] = qid

        ticker = _binding(b, "ticker")
        if ticker and not ref.ticker:
            result.ref_enrichment["ticker"] = ticker

        ch = _binding(b, "ch_number")
        if ch and not ref.companies_house_number:
            result.ref_enrichment["companies_house_number"] = ch

        # ── field updates ────────────────────────────────────────────

        legal_name = _binding(b, "companyLabel")
        if legal_name:
            result.fields.append(FieldUpdate(
                field_name="legal_name", value=legal_name,
                confidence=Confidence.MEDIUM,
            ))

        inception = _parse_inception(_binding(b, "inception"))
        if inception:
            result.fields.append(FieldUpdate(
                field_name="founded_on", value=inception,
                confidence=Confidence.MEDIUM, as_of=inception,
            ))

        country = _binding(b, "countryLabel")
        if country:
            result.fields.append(FieldUpdate(
                field_name="incorporated_in", value=country,
                confidence=Confidence.MEDIUM,
            ))

        hq = _binding(b, "hqLabel")
        if hq:
            result.fields.append(FieldUpdate(
                field_name="headquarters", value=hq,
                confidence=Confidence.MEDIUM,
            ))

        website = _normalise_website(_binding(b, "website"))
        if website:
            result.fields.append(FieldUpdate(
                field_name="website_url", value=website,
                confidence=Confidence.HIGH,  # website is canonical
            ))
            # Enrich the ref with the domain for downstream adapters.
            host = urlparse(website).netloc.replace("www.", "")
            if host and not ref.website_domain:
                result.ref_enrichment["website_domain"] = host

        # Industry — Wikidata's classification is not SIC/NAICS but it
        # is enough for the prose composer to mention the sector.
        industry = _binding(b, "industryLabel")
        if industry:
            result.fields.append(FieldUpdate(
                field_name="industry_codes", value=(industry,),
                confidence=Confidence.LOW,  # not standardised
                note="Wikidata industry classification (not SIC/NAICS)",
            ))

        # ── founders (separate query) ────────────────────────────────
        if qid:
            founder_bindings = _run_sparql(_QUERY_FOUNDERS % {"qid": qid})
            founders = tuple(
                _binding(fb, "founderLabel")
                for fb in founder_bindings
                if _binding(fb, "founderLabel")
            )
            if founders:
                result.fields.append(FieldUpdate(
                    field_name="founders", value=founders,
                    confidence=Confidence.MEDIUM,
                ))

        return result


def install() -> None:
    """Register the Wikidata adapter."""
    register_adapter(WikidataAdapter())
