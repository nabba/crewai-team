"""Shared minimal RSS / Atom parser (Phase E #6, 2026-05-09).

Originally lived inline in ``app/life_companion/personalized_digest.py``.
``app/episteme/paper_pipeline.py`` had a parallel regex-based parser
that was fragile under namespace prefixes and CDATA blocks. Both now
delegate here.

Stdlib-only — no feedparser/lxml deps. Returns plain dicts so callers
don't have to handle XML element types.

Each entry::

    {"id": "<feed-side id, may equal link>",
     "title": "...", "link": "...",
     "summary": "...", "published": "..."}

The parser is deliberately tolerant: garbage XML returns ``[]``,
mixed RSS+Atom feeds in the same root are both honored, namespace
prefixes are stripped before tag matching.
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

logger = logging.getLogger(__name__)


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SUMMARY_CAP = 240
_TITLE_CAP = 200


def _localname(tag: str) -> str:
    """Drop the ``{namespace}`` prefix that ET prepends to tags."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub("", text or "").strip()


def _link_from_element(elem: ET.Element) -> str:
    """RSS uses ``<link>text</link>``; Atom uses ``<link href="..."/>``."""
    text = (elem.text or "").strip()
    if text:
        return text
    return (elem.attrib.get("href") or "").strip()


def parse(xml_text: str, max_items: int = 20) -> list[dict[str, Any]]:
    """Parse RSS or Atom XML; return up to ``max_items`` entries.

    Stable, dependency-free, and preserves order from the source feed.
    Returns ``[]`` on any parse failure.
    """
    if not xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    out: list[dict[str, Any]] = []

    # RSS 2.0 entries live under channel/item.
    for item in root.iter():
        if _localname(item.tag) != "item":
            continue
        entry: dict[str, Any] = {
            "id": "", "title": "", "link": "",
            "summary": "", "published": "",
        }
        for child in item:
            local = _localname(child.tag).lower()
            if local == "title":
                entry["title"] = (child.text or "").strip()[:_TITLE_CAP]
            elif local == "link":
                entry["link"] = _link_from_element(child)
            elif local == "guid":
                entry["id"] = (child.text or "").strip()
            elif local in ("description", "summary"):
                entry["summary"] = _strip_html(child.text or "")[:_SUMMARY_CAP]
            elif local in ("pubdate", "published", "updated"):
                entry["published"] = (child.text or "").strip()
        if entry["title"] and entry["link"]:
            entry["id"] = entry["id"] or entry["link"]
            out.append(entry)
        if len(out) >= max_items:
            return out

    # Atom: entries are <entry>. We iterate even when RSS items already
    # populated `out` so a feed with both is fully covered.
    for elem in root.iter():
        if _localname(elem.tag).lower() != "entry":
            continue
        entry = {
            "id": "", "title": "", "link": "",
            "summary": "", "published": "",
        }
        for child in elem:
            local = _localname(child.tag).lower()
            if local == "title":
                entry["title"] = (child.text or "").strip()[:_TITLE_CAP]
            elif local == "id":
                entry["id"] = (child.text or "").strip()
            elif local == "link":
                # Atom may have multiple <link> children with rel attrs.
                # The "self" / "alternate" relation we want is usually
                # the first one (or the one without rel).
                if not entry["link"]:
                    entry["link"] = _link_from_element(child)
            elif local in ("summary", "content"):
                entry["summary"] = _strip_html(child.text or "")[:_SUMMARY_CAP]
            elif local in ("published", "updated"):
                if not entry["published"]:
                    entry["published"] = (child.text or "").strip()
        # Atom feeds (including arXiv) sometimes omit <link> when <id>
        # is itself a URL — use the id as link in that case.
        if not entry["link"] and entry["id"].startswith(("http://", "https://")):
            entry["link"] = entry["id"]
        if entry["title"] and entry["link"]:
            entry["id"] = entry["id"] or entry["link"]
            out.append(entry)
        if len(out) >= max_items:
            return out

    return out
