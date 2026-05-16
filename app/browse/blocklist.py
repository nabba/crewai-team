"""Per-domain blocklist for browse ingestion.

The blocklist is the privacy contract's last gate before events land
on disk: any URL whose canonical domain matches a pattern here is
dropped at the reader's edge and never stored, summarised, or sent
to an LLM.

Seeded defaults cover the categories the operator flagged: banking,
health portals (Finnish-aware), authentication endpoints. The
operator-editable file at ``workspace/browse/blocklist.txt`` is
loaded additively on top of the seeds.

Match semantics: each pattern is matched against the canonical
lowercase domain. Two shapes are supported:

  * ``example.com``     — exact host match (also matches subdomains
                          of ``example.com``: ``foo.example.com``).
  * ``*.example.com``   — explicit "any subdomain" wildcard. Functionally
                          identical to ``example.com`` but kept distinct
                          for readability.

Lines starting with ``#`` are comments. Blank lines are ignored.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# ── Seeded defaults ──────────────────────────────────────────────────
#
# Keep this list conservative — false positives (a domain in the seed
# blocklist that shouldn't be) cost the operator real signal, while
# false negatives are recoverable by editing the operator-managed
# blocklist.txt. The categories below match the operator's flags:
# banking, health portals, authentication.

_SEEDED_BLOCKLIST: tuple[str, ...] = (
    # Banking & payments (global)
    "paypal.com",
    "stripe.com",
    "revolut.com",
    "wise.com",
    "n26.com",
    # Banking & payments (Nordics — operator is in Helsinki/Tallinn)
    "op.fi",
    "nordea.fi",
    "danskebank.fi",
    "s-pankki.fi",
    "aktia.fi",
    "swedbank.ee",
    "seb.ee",
    "lhv.ee",
    "coop.ee",
    # Health portals (Finland & Estonia)
    "kanta.fi",
    "omaolo.fi",
    "terveysasema.fi",
    "mehilainen.fi",
    "terveystalo.com",
    "pihlajalinna.fi",
    "digilugu.ee",
    # Authentication / SSO redirect endpoints (these leak which site
    # the user was logging into via the path)
    "accounts.google.com",
    "login.microsoftonline.com",
    "appleid.apple.com",
    "auth0.com",
    "okta.com",
    # MyChart-flavoured patient portals (US/global)
    "mychart.com",
)


def _resolve_blocklist_file() -> Path:
    """Operator-editable blocklist file path. Resolved against the
    same base used by :mod:`app.browse.store` so ``BROWSE_BASE_DIR``
    overrides apply uniformly."""
    from app.browse.store import resolve_base
    return resolve_base() / "blocklist.txt"


def _load_operator_lines(path: Path) -> list[str]:
    """Read the operator-managed file. Failure-isolated — a missing
    file returns ``[]`` (the seed-only blocklist still applies)."""
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.debug("browse.blocklist: read failed for %s", path, exc_info=True)
        return []
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(stripped.lower())
    return out


def _compile_patterns(patterns: Iterable[str]) -> set[str]:
    """Normalise each pattern to its bare-domain form.

    ``*.example.com`` and ``example.com`` collapse to the same entry,
    since both mean "match this host or any subdomain of it". The
    matcher uses suffix-with-dot-boundary semantics.
    """
    out: set[str] = set()
    for p in patterns:
        p = p.strip().lower()
        if not p:
            continue
        if p.startswith("*."):
            p = p[2:]
        out.add(p)
    return out


_cached: tuple[Path, set[str]] | None = None


def _patterns() -> set[str]:
    """Memoise the compiled pattern set; invalidate when the operator
    file's mtime changes."""
    global _cached
    path = _resolve_blocklist_file()
    try:
        mtime = path.stat().st_mtime_ns if path.exists() else 0
    except OSError:
        mtime = 0
    cache_key = (path, mtime)
    if _cached is not None and _cached[0] == cache_key:
        return _cached[1]
    compiled = _compile_patterns(
        list(_SEEDED_BLOCKLIST) + _load_operator_lines(path)
    )
    _cached = (cache_key, compiled)
    return compiled


def reset_cache() -> None:
    """Drop the memoised pattern cache. Used by tests after monkeypatching
    the base path."""
    global _cached
    _cached = None


def is_blocked(domain: str) -> bool:
    """``True`` when ``domain`` is blocked by any seeded or operator
    pattern. Empty input → not blocked (the caller's job to drop those).
    """
    if not domain:
        return False
    domain = domain.lower().strip()
    if domain.startswith("www."):
        domain = domain[4:]
    pats = _patterns()
    if domain in pats:
        return True
    # Suffix match with dot boundary so ``example.com`` matches
    # ``foo.example.com`` but NOT ``notexample.com``.
    for p in pats:
        if domain.endswith("." + p):
            return True
    return False


def mute_domain(domain: str) -> bool:
    """Append ``domain`` to the operator file. Idempotent — won't double-
    write if the domain is already present. Returns ``True`` when a new
    line was written."""
    if not domain.strip():
        return False
    domain = domain.lower().strip()
    if domain.startswith("www."):
        domain = domain[4:]
    if is_blocked(domain):
        return False
    path = _resolve_blocklist_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(domain + "\n")
    except OSError:
        logger.warning("browse.blocklist: mute write failed for %s", domain)
        return False
    reset_cache()
    try:
        from app.identity.continuity_ledger import record_event
        record_event(
            kind="browse_ingestion_policy",
            actor="operator",
            summary=f"browse blocklist: muted {domain}",
            detail={"action": "mute_domain", "domain": domain},
        )
    except Exception:
        logger.debug("browse.blocklist: ledger emit failed", exc_info=True)
    return True


def list_operator_entries() -> list[str]:
    """Return the user-managed entries (not including seeded defaults).
    Useful for the React settings panel."""
    return _load_operator_lines(_resolve_blocklist_file())


def list_seed_entries() -> list[str]:
    """Return the seeded default entries. Read-only — operator can't
    delete these from the React UI; they recompile every boot."""
    return list(_SEEDED_BLOCKLIST)
