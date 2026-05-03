"""
web_search.py — Cascading web search with three-tier fallback.

Tier 1: Brave Search API (paid, primary; auto-backs-off on 402 quota errors).
Tier 2: Self-hosted SearXNG (free; aggregates Google/Bing/DDG via firecrawl-stack).
Tier 3: DuckDuckGo HTML scrape (free; last resort, no infra dependency).

Public API (preserved for all existing callers):
  - search_brave(query, count=5) -> list[dict]   (programmatic; cascades all tiers)
  - web_search(query) -> str                      (CrewAI @tool wrapper)
  - get_search_status() -> dict                   (for dashboard health surface)
"""
from __future__ import annotations

import logging
import os
import time

import requests
from crewai.tools import tool

from app.config import get_brave_api_key

logger = logging.getLogger(__name__)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080/search")
DDG_HTML_URL = "https://html.duckduckgo.com/html/"

# Reusable HTTP session for outbound calls
_session = requests.Session()
_session.headers.update({
    "Accept": "application/json",
    "Accept-Encoding": "gzip",
})

# ── Backend health state (module-local; per-process) ─────────────────────────
# When Brave returns 402 (quota exhausted), don't hammer the API for a day —
# the quota resets monthly so the backoff doesn't need to be precise.
_BRAVE_QUOTA_BACKOFF_S = 24 * 3600

_brave_quota_blocked_until: float = 0.0  # epoch seconds; 0 = no block
_last_backend_used: str | None = None
_last_failure_chain: list[str] = []


def _brave_blocked_now() -> bool:
    return _brave_quota_blocked_until > time.time()


def _search_brave_raw(query: str, count: int) -> list[dict] | None:
    """Returns a result list on success, None on quota-exhaustion (signal to
    skip Brave for the rest of the backoff window), [] on any other failure.
    """
    global _brave_quota_blocked_until
    if _brave_blocked_now():
        return None
    try:
        r = _session.get(
            BRAVE_SEARCH_URL,
            headers={"X-Subscription-Token": get_brave_api_key()},
            params={"q": query, "count": count},
            timeout=10,
        )
        if r.status_code == 402:
            _brave_quota_blocked_until = time.time() + _BRAVE_QUOTA_BACKOFF_S
            logger.warning(
                "web_search: Brave quota exhausted (HTTP 402) — backing off %dh",
                _BRAVE_QUOTA_BACKOFF_S // 3600,
            )
            return None
        r.raise_for_status()
        data = r.json()
        return [
            {
                "title": x.get("title", ""),
                "url": x.get("url", ""),
                "description": x.get("description", ""),
            }
            for x in data.get("web", {}).get("results", [])[:count]
        ]
    except Exception as exc:
        logger.debug("web_search: brave failed: %s", exc)
        return []


def _search_searxng(query: str, count: int) -> list[dict]:
    """Self-hosted SearXNG via the firecrawl stack. Empty list on failure."""
    try:
        r = requests.get(
            SEARXNG_URL,
            params={"q": query, "format": "json", "language": "en"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        return [
            {
                "title": x.get("title", "") or "",
                "url": x.get("url", "") or "",
                "description": x.get("content", "") or "",
            }
            for x in (data.get("results") or [])[:count]
        ]
    except Exception as exc:
        logger.debug("web_search: searxng failed: %s", exc)
        return []


def _search_duckduckgo(query: str, count: int) -> list[dict]:
    """Last-resort DDG HTML scrape. Empty list on failure."""
    try:
        from bs4 import BeautifulSoup
        r = requests.post(
            DDG_HTML_URL,
            data={"q": query},
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; AndrusAI-tech-radar)",
                "Accept": "text/html",
            },
            timeout=15,
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        out: list[dict] = []
        for div in soup.select("div.result")[: count * 2]:  # over-fetch; some are ads
            link = div.select_one("a.result__a")
            snippet = div.select_one(".result__snippet")
            if not link:
                continue
            url = link.get("href", "")
            # DDG redirector — strip wrapper if present
            if "uddg=" in url:
                from urllib.parse import unquote, urlparse, parse_qs
                qs = parse_qs(urlparse(url).query)
                url = unquote(qs.get("uddg", [url])[0])
            out.append({
                "title": link.get_text(strip=True),
                "url": url,
                "description": snippet.get_text(strip=True) if snippet else "",
            })
            if len(out) >= count:
                break
        return out
    except Exception as exc:
        logger.debug("web_search: ddg failed: %s", exc)
        return []


def search_brave(query: str, count: int = 5) -> list[dict]:
    """Cascading web search: Brave → SearXNG → DuckDuckGo.

    Function name is preserved for backwards compatibility with all existing
    callers (atlas/api_scout, atlas/learning_planner, fiction_inspiration,
    research_orchestrator, research_adapters/linkedin_data). They get the
    fallback chain transparently — no caller changes needed.

    Always returns a list (possibly empty when every tier failed).
    """
    global _last_backend_used, _last_failure_chain
    chain: list[str] = []

    res = _search_brave_raw(query, count)
    if res:
        _last_backend_used = "brave"
        _last_failure_chain = chain
        return res
    chain.append("brave:quota" if res is None else "brave:error")

    res = _search_searxng(query, count)
    if res:
        _last_backend_used = "searxng"
        _last_failure_chain = chain
        return res
    chain.append("searxng:no_results")

    res = _search_duckduckgo(query, count)
    if res:
        _last_backend_used = "ddg"
        _last_failure_chain = chain
        return res
    chain.append("ddg:no_results")

    _last_backend_used = None
    _last_failure_chain = chain
    logger.warning(
        "web_search: all backends failed for %r: %s",
        query[:60], chain,
    )
    return []


def get_search_status() -> dict:
    """Snapshot of search-backend health for the dashboard.

    Lets the React app tell the user *why* tech radar might be quiet —
    e.g. "Brave quota exhausted, falling back to SearXNG" — instead of
    silently showing an empty list.
    """
    return {
        "last_backend_used": _last_backend_used,
        "last_failure_chain": list(_last_failure_chain),
        "brave_quota_blocked_until": (
            _brave_quota_blocked_until if _brave_blocked_now() else None
        ),
    }


@tool("web_search")
def web_search(query: str) -> str:
    """
    Search the web (Brave → SearXNG → DuckDuckGo fallback chain).
    Returns top 5 results as title + URL + snippet.
    """
    results = search_brave(query, 5)
    if not results:
        return "No results found."
    lines = []
    for item in results:
        title = item.get("title") or "No title"
        url = item.get("url") or ""
        snippet = item.get("description") or "No description"
        lines.append(f"**{title}**\n{url}\n{snippet}\n")
    return "\n".join(lines)


# ── Tool registry annotation (Phase 1a, passive) ────────────────────
try:
    from app.tool_registry import Lifecycle, Tier, register_tool

    @register_tool(
        name="web_search",
        capabilities=["searches-web"],
        description=(
            "Search the public web with three-tier fallback: Brave → "
            "SearxNG → DuckDuckGo. Returns top results as a formatted "
            "markdown list (title, URL, snippet). Use this when you "
            "need information beyond your training data — current "
            "events, recent docs, freshly published research."
        ),
        tier=Tier.PRODUCTION,
        lifecycle=Lifecycle.SINGLETON,
    )
    def _web_search_registry_factory():
        return web_search
except ImportError:
    pass
