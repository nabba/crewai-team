"""Multi-source paper / standards feed adapters.

PROGRAM §46.15 (Q10.3). The original :mod:`paper_pipeline` fetches
from arXiv only. This module adds four more sources that share the
same downstream pipeline (dedup → LLM summarize → JSONL ledger →
Signal digest):

  * **OpenReview** — per-venue feeds for NeurIPS / ICML / ICLR
    accepted papers. ML conferences are where the bulk of the
    operator's interests actually land. (Each venue has its own
    "Forum" RSS-ish page; we use the public API for accepted lists.)
  * **Python PEPs** — ``https://peps.python.org/peps.rss``. Captures
    language-level changes that touch agent code.
  * **W3C TR drafts** — ``https://www.w3.org/News/atom.xml``. Useful
    for the Dashboard / PWA / WebPush surfaces (which the operator
    actively uses).
  * **Hugging Face papers** — ``https://huggingface.co/papers/atom``
    (with editorial curation; complements raw arXiv).

Every source exposes the same shape::

    fetch_recent(lookback_days, max_items) -> list[{
        id, title, abstract, published, categories, source,
    }]

so the pipeline can iterate over sources without per-source branches.
Each source is failure-isolated: a network blip on OpenReview must
not block arXiv.

Master switches (per source, default ON):

  ``PAPER_PIPELINE_OPENREVIEW_ENABLED``
  ``PAPER_PIPELINE_PEPS_ENABLED``
  ``PAPER_PIPELINE_W3C_ENABLED``
  ``PAPER_PIPELINE_HF_ENABLED``

The arXiv source remains gated by the top-level
``PAPER_PIPELINE_ENABLED`` (unchanged).
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from app.utils import feed_parser

logger = logging.getLogger(__name__)


_HTTP_TIMEOUT_S = 15.0
_USER_AGENT = "AndrusAI-Episteme/1.0"


def _enabled(env_var: str, default: bool = True) -> bool:
    raw = os.getenv(env_var, "true" if default else "false").strip().lower()
    return raw in ("true", "1", "yes", "on")


def _fetch_url(url: str) -> str:
    """Plain HTTPS GET with a uniform UA + timeout. Returns '' on
    any failure (failure-isolated by design)."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": _USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError):
        logger.debug("feed_sources: fetch failed for %s", url, exc_info=True)
        return ""


def _parse_iso_loose(s: str) -> datetime | None:
    """Accept ISO 8601 with or without ``Z``. Returns None on parse fail."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        pass
    # Some feeds use RFC 822 ("Mon, 06 May 2026 ...") — best-effort parse.
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(s)
        if dt and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
#   Generic RSS/Atom-backed source (used by PEPs + W3C + HF)
# ─────────────────────────────────────────────────────────────────────


def _generic_feed_records(
    *, url: str, source_name: str, lookback_days: int,
    max_items: int, categories: list[str] | None = None,
    kind: str = "paper",
) -> list[dict[str, Any]]:
    """Pull a feed via ``feed_parser`` and normalise to the pipeline's
    record shape. Filters by ``published >= now - lookback_days``.

    ``kind`` tags the source type so downstream consumers (daily
    briefing, JSONL ledger) can branch:

      * ``"paper"``    — research papers (arXiv, OpenReview, HF). The
                        existing experiment-proposal pipeline applies.
      * ``"standard"`` — language / web standards (PEPs, W3C TR).
                        Same pipeline, but the LLM's "experiment"
                        field is unlikely to be ``codeable``.
      * ``"news"``     — editorial commentary (Rundown, Verge, Wired).
                        Surfaces in a separate "📰 News" briefing
                        section; codeable=false expected.
    """
    body = _fetch_url(url)
    if not body:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    cats = list(categories or [])
    out: list[dict[str, Any]] = []
    for entry in feed_parser.parse(body, max_items=max_items * 3):
        pub = _parse_iso_loose(entry.get("published") or "")
        if pub is None:
            # Some feeds (Hugging Face, W3C) emit dates without
            # timezone; treat as UTC.
            continue
        if pub < cutoff:
            continue
        pid = entry.get("id") or entry.get("link") or ""
        if not pid:
            continue
        out.append({
            "id": pid,
            "title": " ".join((entry.get("title") or "").split()),
            "abstract": " ".join((entry.get("summary") or "").split()),
            "published": pub.isoformat(),
            "categories": cats,
            "source": source_name,
            "kind": kind,
        })
        if len(out) >= max_items:
            break
    return out


# ─────────────────────────────────────────────────────────────────────
#   Python PEPs
# ─────────────────────────────────────────────────────────────────────


_PEPS_FEED_URL = "https://peps.python.org/peps.rss"


def fetch_python_peps(
    *, lookback_days: int = 90, max_items: int = 10,
) -> list[dict[str, Any]]:
    """PEPs land at a low cadence (a few per month); 90-day lookback
    catches most actively-discussed proposals. Returns [] when the
    PEPs source is disabled or the fetch fails."""
    if not _enabled("PAPER_PIPELINE_PEPS_ENABLED"):
        return []
    try:
        return _generic_feed_records(
            url=_PEPS_FEED_URL,
            source_name="python_peps",
            lookback_days=lookback_days,
            max_items=max_items,
            categories=["python", "language"],
            kind="standard",
        )
    except Exception:
        logger.debug("feed_sources: PEPs fetcher raised", exc_info=True)
        return []


# ─────────────────────────────────────────────────────────────────────
#   W3C Technical Reports
# ─────────────────────────────────────────────────────────────────────


_W3C_FEED_URL = "https://www.w3.org/News/atom.xml"


def fetch_w3c_tr(
    *, lookback_days: int = 30, max_items: int = 8,
) -> list[dict[str, Any]]:
    """W3C News covers TR drafts + standards updates. 30-day lookback
    matches the dashboard/PWA cadence we typically iterate on."""
    if not _enabled("PAPER_PIPELINE_W3C_ENABLED"):
        return []
    try:
        return _generic_feed_records(
            url=_W3C_FEED_URL,
            source_name="w3c_tr",
            lookback_days=lookback_days,
            max_items=max_items,
            categories=["web", "standards"],
            kind="standard",
        )
    except Exception:
        logger.debug("feed_sources: W3C fetcher raised", exc_info=True)
        return []


# ─────────────────────────────────────────────────────────────────────
#   Hugging Face papers daily
# ─────────────────────────────────────────────────────────────────────


_HF_FEED_URL = "https://huggingface.co/papers/atom"


def fetch_huggingface_papers(
    *, lookback_days: int = 14, max_items: int = 10,
) -> list[dict[str, Any]]:
    """Editorially-curated daily paper picks. Some overlap with arXiv
    (HF surfaces a subset, often with better titles/descriptions),
    but the editorial layer is a different signal."""
    if not _enabled("PAPER_PIPELINE_HF_ENABLED"):
        return []
    try:
        return _generic_feed_records(
            url=_HF_FEED_URL,
            source_name="huggingface_papers",
            lookback_days=lookback_days,
            max_items=max_items,
            categories=["ml", "curated"],
            kind="paper",
        )
    except Exception:
        logger.debug("feed_sources: HF fetcher raised", exc_info=True)
        return []


# ─────────────────────────────────────────────────────────────────────
#   OpenReview NeurIPS / ICML / ICLR
# ─────────────────────────────────────────────────────────────────────


# OpenReview's public API. The "venues" endpoint lists active venues;
# each venue has accepted papers reachable via its notes API. For each
# of the three target venues, we query the latest year's accepted-
# papers list.
_OPENREVIEW_API = "https://api.openreview.net/notes"


_VENUE_QUERIES: dict[str, str] = {
    # The query field selects the venue. The venue ids are stable;
    # year is implicit in the venue id we pick. We use a wildcard
    # match on the venue name and let OpenReview return the latest
    # active venue for each conference.
    "neurips":  "NeurIPS.cc",
    "icml":     "ICML.cc",
    "iclr":     "ICLR.cc",
}


def _openreview_query(venue_token: str, limit: int) -> list[dict[str, Any]]:
    """Query OpenReview for notes (papers) matching a venue token.

    OpenReview's exact API shape varies by major version; this is a
    best-effort fetch — failure → [] and the source degrades gracefully.
    """
    params = urllib.parse.urlencode({
        "content.venue": venue_token,
        "limit": str(limit),
        "sort": "tcdate:desc",
    })
    url = f"{_OPENREVIEW_API}?{params}"
    body = _fetch_url(url)
    if not body:
        return []
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return []
    notes = data.get("notes") if isinstance(data, dict) else None
    if not isinstance(notes, list):
        return []
    return notes


def fetch_openreview(
    *, venues: tuple[str, ...] = ("neurips", "icml", "iclr"),
    lookback_days: int = 120, max_items_per_venue: int = 5,
) -> list[dict[str, Any]]:
    """Top conference accepted papers per venue. The per-venue cap
    keeps the total bounded so a single venue can't crowd the
    digest. 120-day lookback covers the typical post-acceptance
    publishing window."""
    if not _enabled("PAPER_PIPELINE_OPENREVIEW_ENABLED"):
        return []
    try:
        return _fetch_openreview_inner(
            venues=venues, lookback_days=lookback_days,
            max_items_per_venue=max_items_per_venue,
        )
    except Exception:
        logger.debug("feed_sources: OpenReview fetcher raised", exc_info=True)
        return []


def _fetch_openreview_inner(
    *, venues: tuple[str, ...], lookback_days: int,
    max_items_per_venue: int,
) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    out: list[dict[str, Any]] = []
    for slug, venue_token in _VENUE_QUERIES.items():
        if slug not in venues:
            continue
        notes = _openreview_query(venue_token, max_items_per_venue * 3)
        for note in notes:
            if not isinstance(note, dict):
                continue
            content = note.get("content") or {}
            if not isinstance(content, dict):
                continue
            title = _openreview_field(content, "title")
            abstract = _openreview_field(content, "abstract")
            if not title:
                continue
            tcdate = note.get("tcdate")  # epoch ms
            published = ""
            if isinstance(tcdate, (int, float)):
                pub = datetime.fromtimestamp(
                    tcdate / 1000.0, tz=timezone.utc,
                )
                if pub < cutoff:
                    continue
                published = pub.isoformat()
            pid = str(note.get("id") or note.get("forum") or "")
            if not pid:
                continue
            out.append({
                "id": f"openreview:{slug}:{pid}",
                "title": title,
                "abstract": abstract,
                "published": published or datetime.now(timezone.utc).isoformat(),
                "categories": [slug, "openreview"],
                "source": f"openreview_{slug}",
                "kind": "paper",
            })
            if sum(1 for r in out if r["source"] == f"openreview_{slug}") >= max_items_per_venue:
                break
    return out


def _openreview_field(content: dict[str, Any], field: str) -> str:
    """OpenReview's content field can be either a plain string or
    a dict with a ``value`` key (v2 API). Normalise both."""
    raw = content.get(field)
    if isinstance(raw, dict):
        val = raw.get("value") or ""
        return str(val).strip()
    if isinstance(raw, str):
        return raw.strip()
    return ""


# ─────────────────────────────────────────────────────────────────────
#   News sources (Q10 follow-up — 2026-05-16)
#
# Distinct from the paper / standard adapters above: news is editorial
# commentary about the industry, not research output. All three news
# adapters tag rows with ``kind="news"`` so:
#
#   - daily_briefing._gather_relevant_news surfaces them under the
#     dedicated "📰 News (relevant)" section, NOT mixed into the
#     "📚 Paper-to-experiment" digest
#   - the existing experiment-proposal flow runs against them too —
#     codeable=false is the expected outcome from the LLM for news
#     articles, so they don't pollute the "queued codeable ideas"
#     section in the morning briefing
# ─────────────────────────────────────────────────────────────────────


# The Rundown AI — Beehiiv-hosted daily newsletter. As of 2026-05-16
# the publisher does NOT expose a public RSS feed: ``/feed``,
# ``/feed.xml``, ``/rss``, and ``/archive/rss`` all return 404. The
# archive page at ``/archive`` is rendered HTML only (no Atom/RSS
# discovery links in the head).
#
# The fetcher defaults to OFF for that reason — turning it on without
# a working feed URL just spins on 404s. To re-enable, the operator
# needs to supply a working feed URL via ``RUNDOWN_FEED_URL``:
#
#   * a third-party RSS bridge (RSSHub, kill-the-newsletter.com,
#     rss.app, FetchRSS) that generates a feed from the newsletter
#     subscription, OR
#   * a future official Rundown feed if/when the publisher exposes
#     one (Beehiiv newsletters CAN expose RSS — The Rundown has
#     specifically opted out).
_RUNDOWN_FEED_URL = "https://www.therundown.ai/feed"


def fetch_rundown_ai(
    *, lookback_days: int = 14, max_items: int = 8,
) -> list[dict[str, Any]]:
    """The Rundown AI daily newsletter — disabled by default
    because the publisher doesn't expose a public RSS feed (see
    module-level note above the fetcher).

    Operator enables via:
      1. ``RUNDOWN_FEED_URL=<bridge_url>`` env var, AND
      2. ``PAPER_PIPELINE_RUNDOWN_ENABLED=true``.

    Both are required — the env-default OFF prevents silent 404
    spinning when the operator hasn't set up a bridge.
    """
    if not _enabled("PAPER_PIPELINE_RUNDOWN_ENABLED", default=False):
        return []
    url = os.environ.get("RUNDOWN_FEED_URL", "").strip()
    if not url:
        logger.debug(
            "feed_sources: Rundown enabled but RUNDOWN_FEED_URL unset"
        )
        return []
    try:
        return _generic_feed_records(
            url=url,
            source_name="news_rundown",
            lookback_days=lookback_days,
            max_items=max_items,
            categories=["ai", "industry-news"],
            kind="news",
        )
    except Exception:
        logger.debug("feed_sources: Rundown fetcher raised", exc_info=True)
        return []


# The Verge — subscriber-only full-feed RSS. Operator-supplied URL.
_THEVERGE_FEED_URL = (
    "https://www.theverge.com/rss/partner/subscriber-only-full-feed/rss.xml"
)


def fetch_theverge(
    *, lookback_days: int = 7, max_items: int = 8,
) -> list[dict[str, Any]]:
    """The Verge subscriber-only full feed. Tech/AI industry news,
    product launches, regulatory coverage. The subscriber-only feed
    requires the operator's RSS subscription (the public Verge feed
    is title+lede; the partner feed is full text).

    Master switch: ``PAPER_PIPELINE_VERGE_ENABLED`` (default ON).
    """
    if not _enabled("PAPER_PIPELINE_VERGE_ENABLED"):
        return []
    try:
        return _generic_feed_records(
            url=_THEVERGE_FEED_URL,
            source_name="news_theverge",
            lookback_days=lookback_days,
            max_items=max_items,
            categories=["tech", "industry-news"],
            kind="news",
        )
    except Exception:
        logger.debug("feed_sources: Verge fetcher raised", exc_info=True)
        return []


# Wired — main top-stories feed. The operator-supplied URL was the
# Wired RSS index page; this is the primary feed that captures AI,
# security, business stories that overlap with the operator's
# surfaces (Dashboard, PWA, healing, etc.).
_WIRED_FEED_URL = "https://www.wired.com/feed/rss"


def fetch_wired(
    *, lookback_days: int = 7, max_items: int = 8,
) -> list[dict[str, Any]]:
    """Wired top-stories feed. Cross-topic — AI, security, business,
    culture. Operator can switch to a topic-specific Wired feed via
    ``WIRED_FEED_URL`` env (e.g. ``/feed/category/ai/rss`` for the
    AI-only subset).

    Master switch: ``PAPER_PIPELINE_WIRED_ENABLED`` (default ON).
    """
    if not _enabled("PAPER_PIPELINE_WIRED_ENABLED"):
        return []
    url = os.environ.get("WIRED_FEED_URL", _WIRED_FEED_URL).strip() or _WIRED_FEED_URL
    try:
        return _generic_feed_records(
            url=url,
            source_name="news_wired",
            lookback_days=lookback_days,
            max_items=max_items,
            categories=["tech", "industry-news"],
            kind="news",
        )
    except Exception:
        logger.debug("feed_sources: Wired fetcher raised", exc_info=True)
        return []


# ─────────────────────────────────────────────────────────────────────
#   All sources combined
# ─────────────────────────────────────────────────────────────────────


def fetch_extra_sources(
    *, lookback_days: int = 14, max_per_source: int = 8,
) -> list[dict[str, Any]]:
    """Fetch from all enabled non-arXiv sources. Returns a single
    deduped list. arXiv is fetched separately by the existing
    pipeline code; this helper is the one-call surface for the
    additional sources."""
    out: list[dict[str, Any]] = []
    fetchers: list[Callable[[], list[dict[str, Any]]]] = [
        # Papers
        lambda: fetch_openreview(
            lookback_days=120, max_items_per_venue=max(2, max_per_source // 3),
        ),
        lambda: fetch_huggingface_papers(
            lookback_days=lookback_days, max_items=max_per_source,
        ),
        # Standards
        lambda: fetch_python_peps(
            lookback_days=90, max_items=max_per_source,
        ),
        lambda: fetch_w3c_tr(
            lookback_days=lookback_days, max_items=max_per_source,
        ),
        # News (2026-05-16 follow-up)
        lambda: fetch_rundown_ai(
            lookback_days=lookback_days, max_items=max_per_source,
        ),
        lambda: fetch_theverge(
            lookback_days=lookback_days, max_items=max_per_source,
        ),
        lambda: fetch_wired(
            lookback_days=lookback_days, max_items=max_per_source,
        ),
    ]
    seen_ids: set[str] = set()
    for fetcher in fetchers:
        try:
            rows = fetcher()
        except Exception:
            logger.debug(
                "feed_sources: a fetcher raised; "
                "continuing with remaining sources", exc_info=True,
            )
            rows = []
        for r in rows:
            pid = r.get("id") or ""
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            out.append(r)
    return out
