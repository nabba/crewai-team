"""Dashboard URL builders for Signal-message presentation.

Operator approves things from two devices: iPhone (PWA via Tailscale
Funnel HTTPS) and Macbook (Vite dev server on the Tailnet). Signal
messages that ask for approval should surface both URLs so the
operator can tap whichever device is at hand.

Env overrides:
  DASHBOARD_PUBLIC_URL — iPhone-facing base URL (HTTPS, Funnel)
  DASHBOARD_MAC_URL    — Mac-facing base URL   (HTTP, Tailnet:3100)

Defaults derive from the Tailscale Funnel hostname listed in
``app/middleware.py``'s CORS allowlist, so the alert is clickable
out-of-the-box without explicit configuration.

This module is deliberately NOT in config.py (TIER_IMMUTABLE) — these
are notification-presentation details, not safety-critical config.
"""
from __future__ import annotations

import os

# Default hosts. Derived from middleware.py's CORS allowlist so the
# fallback link is the same one the operator already trusts.
DEFAULT_IPHONE_HOST = "https://plgs-macbook-pro---andrus.tail5b289b.ts.net"
DEFAULT_MAC_HOST = "http://plgs-macbook-pro---andrus.tail5b289b.ts.net:3100"


def _join_url(base: str, path: str) -> str:
    base = (base or "").strip().rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}{path}"


def url_iphone(path: str) -> str:
    """Build an iPhone-friendly dashboard URL (HTTPS via Funnel)."""
    base = (os.environ.get("DASHBOARD_PUBLIC_URL") or "").strip()
    if not base:
        base = DEFAULT_IPHONE_HOST
    return _join_url(base, path)


def url_macbook(path: str) -> str:
    """Build a Mac-friendly dashboard URL (Tailnet HTTP on dev port)."""
    base = (os.environ.get("DASHBOARD_MAC_URL") or "").strip()
    if not base:
        base = DEFAULT_MAC_HOST
    return _join_url(base, path)


def signal_links_block(path: str) -> str:
    """Render the two-line iPhone + Mac link block for a Signal message.

    iOS Signal renders each line as a tap target — operator picks
    whichever device is at hand. Phone first because operators are
    far more often holding the phone when a Signal alert lands.
    """
    return (
        f"📱 iPhone: {url_iphone(path)}\n"
        f"💻 Mac:    {url_macbook(path)}"
    )
