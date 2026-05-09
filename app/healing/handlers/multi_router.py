"""Single catch-all router for variant-heavy patterns.

Three runbooks (B, H, F) all key on signatures whose normalized text
includes a model name or column name — so each variant has a different
SHA-1 hash and a clean specific-hash registration is impossible. The
runbooks framework routes the FIRST matching entry, so registering
multiple ``.*`` handlers would silently shadow all but the first.

This router is registered LAST with a ``.*`` pattern. Specific-hash
handlers run before it (insertion order); only anomalies they don't
match fall through here. The router then dispatches by sample
substring to the underlying handler functions.

Tradeoff: per-runbook stats (success-rate gate, recurrence) merge
into the single router name in ``runbook_settings.json`` rather
than splitting per sub-runbook. The ``RunbookResult`` returned by
sub-handlers preserves their original name in ``extra`` for
post-hoc inspection via the audit trail.
"""
from __future__ import annotations

import logging
from typing import Any

from app.healing.handlers.model_capability import (
    handle_embed_misroute,
    handle_no_function_calling,
)
from app.healing.handlers.schema_drift import handle_missing_column
from app.healing.runbooks import RunbookResult, register_runbook

logger = logging.getLogger(__name__)

# Insertion-order matters: more specific substrings first so a sample
# matching multiple shapes is routed to the most precise handler.
_SUB_HANDLERS: list[tuple[tuple[str, ...], callable]] = [
    (("does not support function calling",), handle_no_function_calling),
    (("does not support chat",), handle_embed_misroute),
    (("column", "does not exist"), handle_missing_column),
]


def _matches(sample: str, needles: tuple[str, ...]) -> bool:
    sample_l = sample.lower()
    return all(n.lower() in sample_l for n in needles)


def _handle(anomaly: dict[str, Any]) -> RunbookResult:
    sample = (anomaly.get("pattern_sample") or anomaly.get("sample") or "")
    for needles, handler in _SUB_HANDLERS:
        if _matches(sample, needles):
            try:
                result = handler(anomaly)
            except Exception as exc:
                logger.debug(
                    "self_heal_router: sub-handler %s raised: %s",
                    handler.__name__, exc, exc_info=True,
                )
                return RunbookResult(
                    name="self_heal_router",
                    success=False,
                    detail=f"sub-handler {handler.__name__} raised",
                    error=f"{type(exc).__name__}: {exc}",
                )
            # Preserve the sub-handler's identity in extras so the audit
            # trail explains which routes fired.
            extras = dict(result.extra or {})
            extras["routed_to"] = result.name
            return RunbookResult(
                name="self_heal_router",
                success=result.success,
                detail=result.detail,
                duration_ms=result.duration_ms,
                error=result.error,
                extra=extras,
            )

    # No substring matched — anomaly fell through every specific guard.
    return RunbookResult(
        name="self_heal_router",
        success=True,
        detail="no sub-handler matched",
        extra={"unmatched_sample_prefix": sample[:120]},
    )


def register() -> None:
    """Register the catch-all router. MUST be called LAST so specific-hash
    handlers earlier in insertion order get first crack at anomalies.
    """
    register_runbook("self_heal_router", r".*", _handle)
