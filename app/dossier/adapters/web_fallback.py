"""web_fallback — last-resort adapter using the existing web_search.

When no specialised adapter (Wikidata, Wikipedia, SEC EDGAR, Companies
House) returns useful data, this adapter does targeted web searches
for the most commonly-missed fields and extracts plausible values from
the result snippets.

What it tries to fill
=====================
Only fields where free-text search has acceptable signal-to-noise:

* ``website_url``      — top organic result for ``"<company>" official``
* ``description``      — meta-description-shaped snippet from the same

That's it.  We don't try to scrape revenue or employee counts from web
search snippets — those are the fields where invented numbers do real
damage to investment-grade output.

Why ``LOW`` confidence is correct
=================================
Web snippets are unverified.  The composer treats ``LOW`` confidence as
"may include in narrative; do not anchor any claim on this alone."
The merge layer's source priority (``web_fallback`` is the lowest)
ensures any other adapter's value wins automatically.

Reuse with the existing infrastructure
======================================
We call ``app.tools.web_search.search_brave`` directly — same fast
deterministic path the research_orchestrator uses for its ``search``
adapter.  No LLM in this adapter; the LLM in the dossier pipeline is
strictly composition-only.
"""
from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from app.dossier.adapters._base import (
    DossierAdapterResult,
    FieldUpdate,
    cache_lookup,
    cache_store,
    register_adapter,
)
from app.dossier.schema import CompanyRef, Confidence

logger = logging.getLogger(__name__)


def _looks_like_corporate_url(url: str, company_name: str) -> bool:
    """Heuristic: does this URL look like the company's own site?"""
    if not url or not company_name:
        return False
    try:
        host = urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return False
    if not host:
        return False
    # Reject obvious aggregators / encyclopedias / news.
    blacklist = (
        "wikipedia.org", "wikidata.org", "linkedin.com", "crunchbase.com",
        "bloomberg.com", "reuters.com", "ft.com", "yahoo.com", "google.com",
        "facebook.com", "twitter.com", "instagram.com", "tiktok.com",
        "youtube.com", "indeed.com", "glassdoor.com", "owler.com",
    )
    if any(b in host for b in blacklist):
        return False
    # Approximate match: at least one token from the company name should
    # appear in the host.  E.g. "Tony's Chocolonely" → "tonyschocolonely".
    tokens = re.findall(r"[A-Za-z]{3,}", company_name.lower())
    flat_host = re.sub(r"[^a-z]", "", host)
    return any(tok in flat_host for tok in tokens) if tokens else False


class WebFallbackAdapter:
    """Last-resort adapter for website + description."""

    name = "web_fallback"
    priority = 20  # the lowest non-trivial priority — anything else wins

    def can_collect(self, ref: CompanyRef) -> bool:
        return bool(ref.name)

    def is_configured(self) -> bool:
        # Requires the existing web_search infrastructure to be wired up.
        try:
            from app.tools.web_search import search_brave  # noqa: F401
            return True
        except Exception:
            return False

    def collect(self, ref: CompanyRef) -> DossierAdapterResult:
        cached = cache_lookup(self.name, ref.name)
        if cached is not None:
            return cached
        result = self._collect_uncached(ref)
        cache_store(self.name, ref.name, result)
        return result

    def _collect_uncached(self, ref: CompanyRef) -> DossierAdapterResult:
        try:
            from app.tools.web_search import search_brave
        except Exception as exc:
            return DossierAdapterResult(
                adapter_name=self.name, error=f"web_search unavailable: {exc}",
            )

        query = f'"{ref.name}" official site'
        try:
            results = search_brave(query, count=5)
        except Exception as exc:
            return DossierAdapterResult(
                adapter_name=self.name,
                error=f"search_brave failed: {type(exc).__name__}: {exc}",
            )

        result = DossierAdapterResult(adapter_name=self.name)

        # Find the first plausible corporate URL.
        for r in results or []:
            url = (r or {}).get("url") or ""
            if not _looks_like_corporate_url(url, ref.name):
                continue
            host = urlparse(url).netloc.replace("www.", "")
            result.base_url = url
            if not ref.website_domain and host:
                result.ref_enrichment["website_domain"] = host
            result.fields.append(FieldUpdate(
                field_name="website_url", value=url,
                confidence=Confidence.LOW,
                note=f"web search: {query!r} top result",
            ))
            # If the snippet exists and is sentence-shaped, use as desc.
            snippet = (r.get("description") or "").strip()
            if snippet and len(snippet) > 60:
                result.fields.append(FieldUpdate(
                    field_name="description", value=snippet[:1000],
                    confidence=Confidence.LOW,
                    note="web search snippet",
                ))
            break

        if not result.fields and not result.ref_enrichment:
            result.error = "no plausible corporate URL in top results"
        return result


def install() -> None:
    register_adapter(WebFallbackAdapter())
