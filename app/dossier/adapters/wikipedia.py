"""wikipedia — Wikipedia REST adapter.

Wikipedia provides the narrative tissue Wikidata's structured data
lacks: the company description, milestone narrative, and product/
service prose used in the History and Business Model sections.

We use the Wikipedia REST API (no auth required), pulling:

* page summary (intro paragraph) → ``description``
* page sections heading list → ``milestones`` (best-effort —
  composer treats this as low-confidence prompts, not facts)

The description is always tagged ``LOW`` confidence in the dossier:
Wikipedia is community-edited and slightly stale; the adapter exists
to give the prose composer something to anchor on, not to be the
source of truth for any number.
"""
from __future__ import annotations

import logging
import re
from urllib.parse import quote

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


_USER_AGENT = "BotArmy-Dossier/1.0 (https://github.com/anthropics/claude-code; contact@example.com)"
_SUMMARY_BASE = "https://en.wikipedia.org/api/rest_v1/page/summary"
_SECTIONS_BASE = "https://en.wikipedia.org/w/api.php"


def _resolve_title(ref: CompanyRef) -> str:
    """Pick a Wikipedia title to query.

    Priority:
      * explicit Wikidata-resolved title (future enrichment),
      * company legal name,
      * company short name.
    """
    if ref.name:
        return ref.name.strip()
    return ""


def _fetch_summary(title: str) -> dict | None:
    body = http_get_json(
        f"{_SUMMARY_BASE}/{quote(title)}",
        headers={"User-Agent": _USER_AGENT},
    )
    if not isinstance(body, dict):
        return None
    # Detect Wikipedia's "no such page" 404-shaped 200 — the API
    # returns a JSON body with "type": "https://mediawiki.org/wiki/HyperSwitch/errors/not_found".
    if body.get("type", "").endswith("not_found"):
        return None
    return body


def _fetch_sections(title: str) -> list[str]:
    """Return a list of section headings for the page (best-effort)."""
    body = http_get_json(
        _SECTIONS_BASE,
        params={
            "action": "parse", "page": title,
            "prop": "sections", "format": "json",
        },
        headers={"User-Agent": _USER_AGENT},
    )
    if not isinstance(body, dict):
        return []
    sections = body.get("parse", {}).get("sections", []) or []
    return [s.get("line", "").strip() for s in sections if s.get("line")]


def _strip_markup(text: str) -> str:
    """Remove HTML-ish residue from Wikipedia summaries."""
    if not text:
        return ""
    # Strip ref tags and stray HTML — Wikipedia summaries are mostly clean
    # plaintext but we defensively scrub.
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


class WikipediaAdapter:

    name = "wikipedia"
    priority = 40  # narrative anchor; below Wikidata + regulators

    def can_collect(self, ref: CompanyRef) -> bool:
        return bool(ref.name)

    def is_configured(self) -> bool:
        try:
            import requests  # noqa: F401
            return True
        except ImportError:
            return False

    def collect(self, ref: CompanyRef) -> DossierAdapterResult:
        title = _resolve_title(ref)
        if not title:
            return DossierAdapterResult(
                adapter_name=self.name,
                error="no name in ref to look up",
            )
        cached = cache_lookup(self.name, title)
        if cached is not None:
            return cached

        result = self._collect_uncached(title)
        cache_store(self.name, title, result)
        return result

    def _collect_uncached(self, title: str) -> DossierAdapterResult:
        summary = _fetch_summary(title)
        if not summary:
            return DossierAdapterResult(
                adapter_name=self.name,
                error=f"no Wikipedia page found for {title!r}",
            )

        page_url = (summary.get("content_urls") or {}).get(
            "desktop", {}).get("page", "") or ""
        result = DossierAdapterResult(
            adapter_name=self.name, base_url=page_url,
        )

        extract = _strip_markup(summary.get("extract") or "")
        if extract:
            # Cap at 1000 chars — the composer wants a one-paragraph
            # anchor, not the full intro.
            description = extract[:1000]
            result.fields.append(FieldUpdate(
                field_name="description", value=description,
                confidence=Confidence.LOW,
                note="Wikipedia summary, community-edited",
            ))

        # Section headings → milestones (best-effort).  We filter to
        # plausible milestone-shaped headings (those starting with a
        # year, or containing keywords like "history", "founding").
        sections = _fetch_sections(title)
        milestones = tuple(
            s for s in sections
            if (
                re.match(r"^\d{4}\b", s)  # starts with year
                or any(k in s.lower() for k in ("history", "founding", "acquisition", "ipo", "merger"))
            )
        )
        if milestones:
            result.fields.append(FieldUpdate(
                field_name="milestones", value=milestones,
                confidence=Confidence.LOW,
                note="Wikipedia section headings (composer should re-narrate)",
            ))

        return result


def install() -> None:
    register_adapter(WikipediaAdapter())
