"""dossier.adapters — registry of typed dossier-collection adapters.

Idiom mirrors :mod:`app.tools.research_adapters`: every submodule
exposes an ``install()`` that registers its adapter under a name in
the registry; this package's ``install_defaults()`` calls them all.

Adapters are always registered (even when their guard ``is_configured``
returns False) so consumers can introspect "the dossier system *knows
about* SEC EDGAR" — the collector just skips unconfigured adapters at
collection time.

Status (Phase 2 — free-tier MVP)
================================

Always-on (no API key)
----------------------
* ``wikidata``       — identity resolution + structured facts
* ``wikipedia``      — narrative description + milestone hints
* ``yfinance_market``— public-company market valuation (yfinance lib)
* ``sec_edgar``      — public US-company audited financials
* ``web_fallback``   — last-resort website + description from web search

Optional (env-keyed)
--------------------
* ``companies_house``— UK statutory registry; requires
                       ``COMPANIES_HOUSE_API_KEY`` (free registration).

Future (Phase 2b — paid)
------------------------
* ``crunchbase``     — funding rounds, investors, HC estimates
* ``similarweb``     — web traffic / engagement
* ``opencorporates`` — global registry coverage
"""
from __future__ import annotations

import logging

from app.dossier.adapters._base import (
    DossierAdapter,
    DossierAdapterResult,
    FieldUpdate,
    all_adapters,
    cache_lookup,
    cache_store,
    clear_cache,
    configured_adapters,
    get_adapter,
    register_adapter,
    reset_for_tests,
    source_priority_map,
)

logger = logging.getLogger(__name__)


__all__ = [
    "DossierAdapter",
    "DossierAdapterResult",
    "FieldUpdate",
    "install_defaults",
    "all_adapters",
    "configured_adapters",
    "get_adapter",
    "register_adapter",
    "reset_for_tests",
    "source_priority_map",
    "cache_lookup",
    "cache_store",
    "clear_cache",
]


_INSTALL_DONE = False


def install_defaults() -> None:
    """Register every shipped adapter with the dossier registry.

    Idempotent: re-calling is a no-op.  Safe to call from
    ``app.main`` startup OR from a test fixture; the registry's
    ``register_adapter`` is itself idempotent (replaces in place).
    """
    global _INSTALL_DONE
    if _INSTALL_DONE:
        return

    # Lazy imports — keep dossier package importable even when an
    # adapter's transitive dependency (yfinance, requests) is missing.
    # Each install() guards itself.
    for module_name in (
        "app.dossier.adapters.wikidata",
        "app.dossier.adapters.wikipedia",
        "app.dossier.adapters.yfinance_market",
        "app.dossier.adapters.sec_edgar",
        "app.dossier.adapters.companies_house",
        "app.dossier.adapters.web_fallback",
    ):
        try:
            mod = __import__(module_name, fromlist=["install"])
            mod.install()
        except Exception as exc:
            logger.debug(
                "dossier.adapters: %s install failed: %s: %s "
                "(adapter remains unavailable this session)",
                module_name, type(exc).__name__, exc,
            )

    _INSTALL_DONE = True
    _log_status_once()


_STATUS_LOGGED = False


def _log_status_once() -> None:
    global _STATUS_LOGGED
    if _STATUS_LOGGED:
        return
    _STATUS_LOGGED = True
    try:
        active = [a.name for a in configured_adapters()]
        all_names = [a.name for a in all_adapters()]
        inactive = [n for n in all_names if n not in active]
        if active:
            logger.info(
                "dossier.adapters: ACTIVE: %s", ", ".join(active),
            )
        if inactive:
            logger.info(
                "dossier.adapters: registered but not configured "
                "(missing API keys / deps): %s", ", ".join(inactive),
            )
    except Exception:
        logger.debug("dossier.adapters: status log failed", exc_info=True)
