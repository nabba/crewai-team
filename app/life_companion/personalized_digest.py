"""Personalized weekly digest (Phase D #5, 2026-05-09).

Closes the gap from the original Wave 2 plan that the Phase B
``personalized_signal_crew`` did NOT cover. The Phase B work added
context-aware Signal output (mood/time-of-day → tone hint); this
module does the OTHER thing the user asked for: a weekly digest of
what's new in their personal information surface.

Sources, polled in parallel and stitched into one Signal message:

  1. **RSS feeds** — operator-curated list at
     ``workspace/companion/personalized_feeds.json``.
  2. **GitHub starred-repo activity** — public events feed for the
     operator's GitHub username (no auth needed for public events).
  3. **arXiv by author** — papers from a list of authors the operator
     follows, fetched the same way as Phase C's paper_pipeline.
  4. **Venture news** — Google News RSS for the operator's ventures
     (plg / archibal / kaicart per CLAUDE.md). Optional via env.

Each source contributes ≤ ``MAX_PER_SOURCE`` items (default 3).
The composed digest is sent on a fixed weekday + hour
(``DIGEST_WEEKDAY`` / ``DIGEST_HOUR``, default Friday 09:00 local).

Cadence: hourly probe via the idle scheduler; the weekday/hour gate
is checked internally so the digest fires once per week. Dedup key is
the day-of-week, so a missed window doesn't mean a missed digest —
the next pass within that day still fires.

State at ``workspace/life_companion/personalized_digest.json``::

    {"last_sent_iso_week": "2026-W19", "seen_urls": [...]}

Master switch: ``PERSONALIZED_DIGEST_ENABLED`` (default ON; LIFE_COMPANION
master also gates).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.life_companion._common import (
    audit_event,
    background_enabled,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)


_STATE_FILE = "personalized_digest.json"
_FEEDS_PATH = Path("/app/workspace/companion/personalized_feeds.json")
_RUN_CADENCE_S = 3600  # hourly probe; internal weekday/hour gate

_HTTP_TIMEOUT_S = 12.0
_MAX_PER_SOURCE = 3
_SEEN_CAP = 1000
_DEFAULT_WEEKDAY = 4   # Friday (Mon=0, Sun=6)
_DEFAULT_HOUR = 9      # 09:00 local


def _enabled() -> bool:
    return os.getenv("PERSONALIZED_DIGEST_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _digest_weekday() -> int:
    raw = os.getenv("PERSONALIZED_DIGEST_WEEKDAY", str(_DEFAULT_WEEKDAY)).strip()
    try:
        return max(0, min(6, int(raw)))
    except ValueError:
        return _DEFAULT_WEEKDAY


def _digest_hour() -> int:
    raw = os.getenv("PERSONALIZED_DIGEST_HOUR", str(_DEFAULT_HOUR)).strip()
    try:
        return max(0, min(23, int(raw)))
    except ValueError:
        return _DEFAULT_HOUR


def _now_local() -> datetime:
    return datetime.now().astimezone()


def _iso_week_str(dt: datetime) -> str:
    iso = dt.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


# ── Source loaders ────────────────────────────────────────────────────────


def _http_get(url: str, accept: str = "*/*") -> str:
    req = urllib.request.Request(
        url, headers={
            "User-Agent": "AndrusAI-Digest/1.0",
            "Accept": accept,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        logger.debug("personalized_digest: GET failed for %s", url, exc_info=True)
        return ""


def _load_feeds_config() -> dict:
    if not _FEEDS_PATH.exists():
        return {}
    try:
        return json.loads(_FEEDS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ── RSS / Atom parser ────────────────────────────────────────────────────


def _parse_feed(xml_text: str, max_items: int) -> list[dict]:
    """Return up to ``max_items`` entries from an RSS or Atom feed.

    Each entry: {title, link, summary, published}. [] on parse failure.
    """
    if not xml_text:
        return []
    out: list[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    # Strip namespaces — most feed parsers do this so callers can use
    # plain tag names. The arXiv Atom feed uses `atom:` etc.
    def _localname(tag: str) -> str:
        return tag.rsplit("}", 1)[-1] if "}" in tag else tag

    # RSS 2.0: items live under channel/item.
    for item in root.iter():
        if _localname(item.tag) != "item":
            continue
        title, link, summary, published = "", "", "", ""
        for child in item:
            local = _localname(child.tag).lower()
            text = (child.text or "").strip()
            if local == "title":
                title = text
            elif local == "link":
                link = text or (child.attrib.get("href") or "")
            elif local in ("description", "summary"):
                summary = re.sub(r"<[^>]+>", "", text)[:240]
            elif local in ("pubdate", "published", "updated"):
                published = text
        if title and link:
            out.append({"title": title[:200], "link": link,
                        "summary": summary, "published": published})
        if len(out) >= max_items:
            return out

    # Atom: entries are <entry>.
    if not out:
        for entry in root.iter():
            if _localname(entry.tag).lower() != "entry":
                continue
            title, link, summary, published = "", "", "", ""
            for child in entry:
                local = _localname(child.tag).lower()
                if local == "title":
                    title = (child.text or "").strip()[:200]
                elif local == "link":
                    link = child.attrib.get("href") or (child.text or "").strip()
                elif local in ("summary", "content"):
                    summary = re.sub(
                        r"<[^>]+>", "", (child.text or "").strip(),
                    )[:240]
                elif local in ("published", "updated"):
                    published = (child.text or "").strip()
            if title and link:
                out.append({"title": title, "link": link,
                            "summary": summary, "published": published})
            if len(out) >= max_items:
                break
    return out


def _gather_rss(feeds: list[str], seen: set[str]) -> list[dict]:
    out: list[dict] = []
    for url in feeds:
        if not url:
            continue
        xml_text = _http_get(url, accept="application/rss+xml, application/atom+xml")
        items = _parse_feed(xml_text, max_items=_MAX_PER_SOURCE)
        for it in items:
            if it["link"] in seen:
                continue
            it["source"] = "rss"
            it["feed_url"] = url
            out.append(it)
        if len(out) >= _MAX_PER_SOURCE * 5:
            break
    return out


def _gather_github_user_events(username: str, seen: set[str]) -> list[dict]:
    if not username:
        return []
    url = f"https://api.github.com/users/{urllib.parse.quote(username)}/events/public"
    raw = _http_get(url, accept="application/vnd.github+json")
    if not raw:
        return []
    try:
        events = json.loads(raw)
    except Exception:
        return []
    out: list[dict] = []
    for ev in events:
        ev_type = ev.get("type") or ""
        if ev_type not in (
            "PushEvent", "PullRequestEvent", "IssuesEvent",
            "WatchEvent", "ReleaseEvent", "ForkEvent", "CreateEvent",
        ):
            continue
        repo = (ev.get("repo") or {}).get("name") or ""
        if not repo:
            continue
        link = f"https://github.com/{repo}"
        if link in seen:
            continue
        out.append({
            "source": "github",
            "title": f"{ev_type[:-5]}: {repo}",
            "link": link,
            "summary": "",
            "published": ev.get("created_at", ""),
        })
        if len(out) >= _MAX_PER_SOURCE:
            break
    return out


def _gather_arxiv_by_author(authors: list[str], seen: set[str]) -> list[dict]:
    if not authors:
        return []
    parts = [f'au:"{a}"' for a in authors if a.strip()]
    if not parts:
        return []
    query = " OR ".join(parts)
    params = urllib.parse.urlencode({
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": _MAX_PER_SOURCE * 2,
    })
    raw = _http_get(
        f"http://export.arxiv.org/api/query?{params}",
        accept="application/atom+xml",
    )
    items = _parse_feed(raw, max_items=_MAX_PER_SOURCE * 2)
    out: list[dict] = []
    for it in items:
        if it["link"] in seen:
            continue
        it["source"] = "arxiv_author"
        out.append(it)
        if len(out) >= _MAX_PER_SOURCE:
            break
    return out


def _gather_venture_news(ventures: list[str], seen: set[str]) -> list[dict]:
    if not ventures:
        return []
    out: list[dict] = []
    for v in ventures:
        if not v.strip():
            continue
        params = urllib.parse.urlencode({"q": v, "hl": "en", "gl": "US", "ceid": "US:en"})
        url = f"https://news.google.com/rss/search?{params}"
        xml_text = _http_get(url, accept="application/rss+xml")
        items = _parse_feed(xml_text, max_items=_MAX_PER_SOURCE)
        for it in items:
            if it["link"] in seen:
                continue
            it["source"] = "venture_news"
            it["venture"] = v
            out.append(it)
        if len(out) >= _MAX_PER_SOURCE * 3:
            break
    return out


# ── Composition ──────────────────────────────────────────────────────────


def _format_digest(items_by_source: dict[str, list[dict]]) -> str:
    """Build a Signal-friendly digest body."""
    lines = ["📰 Weekly personalized digest", ""]
    if items_by_source.get("rss"):
        lines.append("RSS:")
        for it in items_by_source["rss"][:_MAX_PER_SOURCE]:
            lines.append(f"  • {it['title'][:90]}")
            lines.append(f"    {it['link']}")
        lines.append("")
    if items_by_source.get("github"):
        lines.append("GitHub:")
        for it in items_by_source["github"][:_MAX_PER_SOURCE]:
            lines.append(f"  • {it['title'][:90]}")
        lines.append("")
    if items_by_source.get("arxiv_author"):
        lines.append("arXiv (followed authors):")
        for it in items_by_source["arxiv_author"][:_MAX_PER_SOURCE]:
            arxiv_id = it["link"].rsplit("/", 1)[-1]
            lines.append(f"  • [{arxiv_id}] {it['title'][:80]}")
        lines.append("")
    if items_by_source.get("venture_news"):
        lines.append("Venture news:")
        for it in items_by_source["venture_news"][:_MAX_PER_SOURCE]:
            v = it.get("venture", "")
            tag = f"[{v}] " if v else ""
            lines.append(f"  • {tag}{it['title'][:90]}")
        lines.append("")
    if len(lines) <= 2:
        return ""
    return "\n".join(lines).rstrip()


# ── Main ──────────────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ran": False, "items_by_source": {}, "sent": False,
    }
    if not _enabled() or not background_enabled():
        return summary

    state = read_state_json(_STATE_FILE, {
        "last_run_at": 0.0, "last_sent_iso_week": "", "seen_urls": [],
    })
    now_ts = time.time()
    if now_ts - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return summary
    state["last_run_at"] = now_ts
    summary["ran"] = True

    now_local = _now_local()
    target_weekday = _digest_weekday()
    target_hour = _digest_hour()

    # Once-per-week gate. Allow firing any time on the target weekday
    # at-or-after the target hour, until end of day, deduped by ISO week.
    iso_week = _iso_week_str(now_local)
    if iso_week == state.get("last_sent_iso_week"):
        write_state_json(_STATE_FILE, state)
        return summary
    if now_local.weekday() != target_weekday:
        write_state_json(_STATE_FILE, state)
        return summary
    if now_local.hour < target_hour:
        write_state_json(_STATE_FILE, state)
        return summary

    seen = set(state.get("seen_urls") or [])
    config = _load_feeds_config()

    rss_feeds = config.get("rss") or []
    if isinstance(rss_feeds, list):
        rss_items = _gather_rss(rss_feeds, seen)
    else:
        rss_items = []

    github_user = (
        config.get("github_username")
        or os.getenv("GITHUB_FOLLOWING_USER", "").strip()
    )
    github_items = _gather_github_user_events(github_user, seen) if github_user else []

    arxiv_authors_cfg = config.get("arxiv_authors") or []
    if not arxiv_authors_cfg:
        env_authors = os.getenv("ARXIV_FOLLOWING_AUTHORS", "").strip()
        arxiv_authors_cfg = [a.strip() for a in env_authors.split(",") if a.strip()]
    arxiv_items = _gather_arxiv_by_author(arxiv_authors_cfg, seen)

    ventures_cfg = config.get("ventures") or []
    if not ventures_cfg:
        env_v = os.getenv("VENTURES", "PLG,Archibal,KaiCart").strip()
        ventures_cfg = [v.strip() for v in env_v.split(",") if v.strip()]
    venture_items = _gather_venture_news(ventures_cfg, seen)

    items_by_source = {
        "rss": rss_items,
        "github": github_items,
        "arxiv_author": arxiv_items,
        "venture_news": venture_items,
    }
    summary["items_by_source"] = {k: len(v) for k, v in items_by_source.items()}

    # Update seen-set with everything we surfaced this round.
    new_links = []
    for items in items_by_source.values():
        for it in items:
            link = it.get("link") or ""
            if link:
                new_links.append(link)
    if new_links:
        merged = list(seen) + new_links
        # Cap to bounded size to keep state file from growing forever.
        state["seen_urls"] = merged[-_SEEN_CAP:]

    body = _format_digest(items_by_source)
    if body:
        try:
            send_signal_alert(body, tag="personalized_digest")
            summary["sent"] = True
            state["last_sent_iso_week"] = iso_week
        except Exception:
            logger.debug(
                "personalized_digest: send failed", exc_info=True,
            )

    write_state_json(_STATE_FILE, state)
    audit_event(
        "personalized_digest_pass",
        sent=summary["sent"],
        items_by_source=summary["items_by_source"],
    )
    return summary
