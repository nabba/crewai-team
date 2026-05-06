"""DossierAdapter — base protocol + registry.

A dossier adapter takes a :class:`CompanyRef` and returns a typed
partial dossier (one or many fields it can fill from its source) plus
optional ref enrichments (e.g. discovering the SEC CIK from a ticker).

Why this differs from ``research_orchestrator.Adapter``
=======================================================
The matrix orchestrator is shaped for "flat string per (subject, field)"
research (e.g. "for these 35 PSPs, find website + sales email").  Every
adapter call resolves a *single* field.

A dossier adapter naturally fills MANY fields per fetch — a single 10-K
download yields revenue + EBITDA + employees + executives + risk
factors + ownership.  Treating those as N independent calls would
multiply cost and miss the intra-document context (e.g. fiscal year
end is needed to interpret the revenue figure).

So the dossier protocol returns a partial result; the collector merges
those into the canonical :class:`CompanyDossier`.

Reuse with the matrix orchestrator
==================================
Adapters that genuinely fill ONE flat string per (company, field) —
e.g. "verify the company's website is X" — should be implemented as
``research_orchestrator`` adapters.  The dossier collector calls the
matrix orchestrator for those fields.  This module is for the richer,
typed, multi-field-per-fetch sources.

Cost / freshness contract
=========================
* ``can_collect()`` MUST be fast (no network).  It tells the collector
  whether the adapter has enough identity to attempt a fetch.
* ``collect()`` is wrapped in a hard per-call timeout by the collector.
* Adapters cache per-process so the same company doesn't re-fetch
  within a single dossier run; the collector clears the cache between
  runs to keep memory bounded.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Protocol, runtime_checkable

from app.dossier.schema import (
    CompanyRef,
    Confidence,
    Source,
)

logger = logging.getLogger(__name__)


@dataclass
class FieldUpdate:
    """One field a dossier adapter is offering to fill.

    Field name is the dossier attribute name (``"revenue_usd"``,
    ``"founded_on"``, etc.); collector validates it against the schema.
    """

    field_name: str
    value: Any
    confidence: Confidence
    as_of: date | None = None
    note: str = ""

    def materialize_source(self, *, adapter: str, url: str = "",
                           document_id: str = "") -> Source:
        """Build the canonical Source for this update.  Adapters call
        this rather than constructing :class:`Source` directly so the
        adapter name is stamped consistently."""
        return Source(
            adapter=adapter,
            url=url,
            document_id=document_id,
            note=self.note,
        )


@dataclass
class DossierAdapterResult:
    """What an adapter returns from one ``collect()`` call.

    ``fields`` is a list (not dict) so adapters can offer multiple
    candidate values for the same field (e.g. revenue from current year
    + revenue from prior year — the collector keeps the most recent).
    The collector picks the highest-priority/highest-confidence value
    per field and records the rest as conflicts.

    ``ref_enrichment`` lets an adapter share newly-discovered
    identifiers with subsequent adapters in the chain — e.g.
    Wikidata returns ``{"ticker": "SPOT", "wikidata_id": "Q689141"}``
    for "Spotify" which then unlocks SEC EDGAR + yfinance.
    """

    adapter_name: str
    base_url: str = ""             # cite-back URL — empty if N/A
    fields: list[FieldUpdate] = field(default_factory=list)
    ref_enrichment: dict[str, str] = field(default_factory=dict)
    error: str = ""                # non-empty when the adapter failed
    note: str = ""

    def is_ok(self) -> bool:
        return not self.error


@runtime_checkable
class DossierAdapter(Protocol):
    """The adapter contract.

    Adapters are stateless, per-process-cacheable, environment-gated
    by a ``guard()`` (mirrors the tool_registry pattern).
    """

    name: str
    priority: int  # higher = more authoritative; used for reconciliation

    def can_collect(self, ref: CompanyRef) -> bool:
        """Return True iff the adapter has enough identity in ``ref`` to
        attempt a fetch.  Must be cheap (no network)."""
        ...

    def collect(self, ref: CompanyRef) -> DossierAdapterResult:
        """Fetch + parse from the adapter's source.

        MUST NOT raise — return ``DossierAdapterResult(error=...)`` on
        failure so the collector can record the failure and continue
        with other adapters.
        """
        ...

    def is_configured(self) -> bool:
        """True iff env vars / API keys / dependencies are in place.
        Adapters that need no config return True unconditionally."""
        ...


# ── Registry ─────────────────────────────────────────────────────────


_ADAPTERS: dict[str, DossierAdapter] = {}


def register_adapter(adapter: DossierAdapter) -> None:
    """Register an adapter under its declared ``name``.

    Idempotent: re-registering the same name replaces in place.
    Mirrors the ``research_orchestrator.register_adapter`` pattern so
    tests can swap adapters with mocks the same way.
    """
    if not adapter.name:
        raise ValueError("DossierAdapter must declare a non-empty name")
    _ADAPTERS[adapter.name] = adapter
    logger.debug("dossier: registered adapter %s (priority=%d)",
                 adapter.name, adapter.priority)


def get_adapter(name: str) -> DossierAdapter | None:
    return _ADAPTERS.get(name)


def all_adapters() -> list[DossierAdapter]:
    """Return all registered adapters, sorted by descending priority.

    Stable secondary sort by name so tests are deterministic.
    """
    return sorted(_ADAPTERS.values(), key=lambda a: (-a.priority, a.name))


def configured_adapters() -> list[DossierAdapter]:
    """Subset of :func:`all_adapters` whose ``is_configured()`` returns True.

    Adapters that need an API key absent from the environment are
    visible in the registry but skipped by the collector.
    """
    return [a for a in all_adapters() if a.is_configured()]


def source_priority_map() -> dict[str, int]:
    """Build the source_priority dict consumed by ``schema.merge_field``.

    Recomputed each call (cheap) so adapters registered after import
    are visible.
    """
    return {a.name: a.priority for a in all_adapters()}


def reset_for_tests() -> None:
    """Clear the registry — used by tests that build minimal adapter sets."""
    _ADAPTERS.clear()


# ── Convenience: HTTP helpers ────────────────────────────────────────
#
# Adapters share a small set of conventions: short timeouts, no infinite
# retries, debug-level logging on failure.  Centralising the HTTP shape
# avoids each adapter inventing its own retry policy.

DEFAULT_TIMEOUT_SECS = 12

_http_session_cache: list[Any] = []  # populated lazily


def http_get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECS,
) -> dict[str, Any] | list[Any] | None:
    """Tiny ``requests``-backed GET-JSON helper.

    Returns the parsed body, ``None`` on any failure.  Never raises —
    adapter contract requires that failures bubble through
    :class:`DossierAdapterResult.error` rather than exceptions.
    """
    try:
        import requests  # lazy — keep startup cost off
    except ImportError:
        logger.debug("dossier: requests not installed; HTTP adapters disabled")
        return None
    try:
        resp = requests.get(url, params=params or {}, headers=headers or {},
                            timeout=timeout)
    except Exception as exc:
        logger.debug("dossier: GET %s failed: %s: %s", url,
                     type(exc).__name__, exc)
        return None
    if resp.status_code != 200:
        logger.debug("dossier: GET %s returned status=%s", url, resp.status_code)
        return None
    try:
        return resp.json()
    except Exception:
        return None


def http_get_text(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECS,
) -> str | None:
    """Same shape as :func:`http_get_json` but returns raw text."""
    try:
        import requests
    except ImportError:
        return None
    try:
        resp = requests.get(url, params=params or {}, headers=headers or {},
                            timeout=timeout)
    except Exception as exc:
        logger.debug("dossier: GET-text %s failed: %s: %s", url,
                     type(exc).__name__, exc)
        return None
    if resp.status_code != 200:
        return None
    return resp.text or None


# ── Per-process cache ────────────────────────────────────────────────
#
# Keys are (adapter_name, identity_key) where identity_key is whatever
# uniquely identifies the company within the adapter's source — e.g.
# ticker for SEC EDGAR, wikidata_id for Wikidata.  Cleared between
# dossier runs.

_CACHE: dict[tuple[str, str], DossierAdapterResult] = {}


def cache_lookup(adapter_name: str, identity_key: str) -> DossierAdapterResult | None:
    return _CACHE.get((adapter_name, identity_key))


def cache_store(
    adapter_name: str, identity_key: str, result: DossierAdapterResult,
) -> None:
    _CACHE[(adapter_name, identity_key)] = result


def clear_cache() -> None:
    _CACHE.clear()
