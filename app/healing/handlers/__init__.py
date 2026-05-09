"""Operational runbook handlers — registered at import time.

Importing this package wires every shipped handler into the dispatcher
in ``app/healing/runbooks.py``. Registration order matters because the
dispatcher returns the first ``pattern.search(sig)`` match:

  1. Clear the default ``log_only`` catch-all so specific patterns match.
  2. Register specific-hash handlers (A, D, E, C, G).
  3. Register the multi-router (B + H + F) with ``.*`` LAST so
     specific-hash matches always win first.
  4. Re-register a no-op fallback under ``log_only`` so unmatched
     anomalies still get a dispatch event for visibility.

All registrations are best-effort; an import error in one handler must
not prevent the others from loading.

The module is imported eagerly from ``app/healing/__init__.py`` so the
mere act of ``from app.healing.error_diagnosis import diagnose_and_fix``
in ``app/main.py`` triggers our wiring. No code in TIER_IMMUTABLE files
needs to change.
"""
from __future__ import annotations

import logging

from app.healing.runbooks import register_runbook, unregister_runbook

logger = logging.getLogger(__name__)

# Track registration so re-imports (e.g. test runs) don't double-register.
_REGISTERED = False


def _safe_register(module_name: str, register_fn) -> None:
    try:
        register_fn()
        logger.debug("self_heal_handlers: registered %s", module_name)
    except Exception:
        logger.warning(
            "self_heal_handlers: registration failed for %s", module_name,
            exc_info=True,
        )


def _fallback_log_only(anomaly: dict) -> "RunbookResult":  # type: ignore[name-defined]
    """Catch-all logger registered LAST. Mirrors the framework's reference
    handler — anomalies that no specific handler claimed end up here so
    the audit trail still records every dispatch.
    """
    from app.healing.runbooks import RunbookResult
    sig = anomaly.get("pattern_signature") or ""
    sev = anomaly.get("severity", "info")
    logger.info(
        "self_heal_handlers[fallback]: unmatched anomaly sig=%s sev=%s type=%s",
        sig, sev, anomaly.get("anomaly_type"),
    )
    return RunbookResult(
        name="log_only",
        success=True,
        detail=f"fallback log only — sig={sig} sev={sev}",
    )


def install() -> None:
    """Wire all handlers. Idempotent within a single process."""
    global _REGISTERED
    if _REGISTERED:
        return

    # Step 1 — clear the default catch-all so specific hashes can match.
    try:
        unregister_runbook("log_only")
    except Exception:
        logger.debug("self_heal_handlers: unregister log_only failed", exc_info=True)

    # Step 2 — specific-hash handlers. Each ``register()`` is idempotent
    # at the dispatcher level (re-registration just replaces the entry).
    try:
        from app.healing.handlers import db_pool
        _safe_register("db_pool", db_pool.register)
    except Exception:
        logger.warning("self_heal_handlers: db_pool import failed", exc_info=True)

    try:
        from app.healing.handlers import scheduler_overrun
        _safe_register("scheduler_overrun", scheduler_overrun.register)
    except Exception:
        logger.warning("self_heal_handlers: scheduler_overrun import failed", exc_info=True)

    try:
        from app.healing.handlers import schema_drift
        _safe_register("schema_drift", schema_drift.register)
    except Exception:
        logger.warning("self_heal_handlers: schema_drift import failed", exc_info=True)

    try:
        from app.healing.handlers import code_drift
        _safe_register("code_drift", code_drift.register)
    except Exception:
        logger.warning("self_heal_handlers: code_drift import failed", exc_info=True)

    # Step 3 — catch-all multi-router AFTER all specific handlers.
    try:
        from app.healing.handlers import multi_router
        _safe_register("multi_router", multi_router.register)
    except Exception:
        logger.warning("self_heal_handlers: multi_router import failed", exc_info=True)

    # Step 4 — fallback log_only LAST. The dispatcher iterates entries
    # in insertion order, so this only matches anomalies the multi-router
    # already handled… which means it never fires unless the router is
    # disabled in runbook_settings.json. Belt-and-suspenders.
    try:
        register_runbook("log_only", r".*", _fallback_log_only)
    except Exception:
        logger.debug("self_heal_handlers: log_only fallback failed", exc_info=True)

    _REGISTERED = True
    logger.info("self_heal_handlers: registration complete")


# Eager registration on import.
install()
