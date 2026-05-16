"""URL canonicalisation for browser-history ingestion.

The single rule: **store only ``scheme://domain/path``**. Query strings
and fragments are stripped at canonicalisation time and never persisted.

This is a privacy contract — query strings carry search terms (``?q=…``),
session tokens (``?token=…``), and user identifiers in lots of webapps.
By making the strip step structural (rather than a sanitiser that could
be bypassed), we guarantee the events.jsonl can't contain those values.

The pin test ``tests/browse/test_url_canon.py::test_query_stripped``
makes the contract observable in CI.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class CanonicalURL:
    """Parsed + canonicalised URL components."""

    scheme: str
    domain: str
    path: str

    @property
    def url(self) -> str:
        if not self.domain:
            return ""
        path = self.path or "/"
        return f"{self.scheme}://{self.domain}{path}"


def canonicalize(raw: str) -> CanonicalURL | None:
    """Parse a URL string and return its canonical form, or ``None`` if
    the URL is unparseable / non-http(s) / has no host.

    Steps:

      1. Parse with :func:`urllib.parse.urlsplit`.
      2. Lowercase the scheme; reject anything that isn't ``http`` /
         ``https``. ``file://``, ``chrome://``, ``about:`` etc. all
         filter out at this layer — internal pages aren't interest signal.
      3. Lowercase the host; strip a leading ``www.``.
      4. Drop the query string and fragment entirely.
      5. Strip a single trailing slash from non-root paths so
         ``/foo`` and ``/foo/`` canonicalise the same.
    """
    if not raw:
        return None
    try:
        parts = urlsplit(raw)
    except ValueError:
        return None

    scheme = (parts.scheme or "").lower()
    if scheme not in ("http", "https"):
        return None

    host = (parts.hostname or "").lower()
    if not host:
        return None
    if host.startswith("www."):
        host = host[4:]

    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"

    return CanonicalURL(scheme=scheme, domain=host, path=path)


def truncate_title(title: str | None, *, max_len: int = 200) -> str | None:
    """Truncate a page title to ``max_len`` chars; keep ``None`` as
    ``None``. Returns ``None`` for empty/whitespace-only titles."""
    if title is None:
        return None
    title = title.strip()
    if not title:
        return None
    if len(title) <= max_len:
        return title
    return title[: max_len - 1] + "…"
