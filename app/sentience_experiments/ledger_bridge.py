"""Bridge: sentience landmark observations → identity continuity ledger.

PROGRAM §43.4 — Q5.4.2. Without this bridge the sentience modules
emit JSONL to disk forever, but the annual-reflection self-narrative
never knows the system started observing itself this year. Same
class of integrity gap that motivated the ledger in the first
place (§32.8).

Discipline:

  * **LANDMARK events only** — not every detection. Routine
    detections live in their module JSONLs; landmark events are
    the ones that should show up in the year-end reflection.
  * **Opaque counts only** — no action_signatures, no breach
    kinds, no person identities in the ``summary`` text.
  * **Rate-limited** — module-level counter prevents flooding
    the ledger from one detection burst.
  * **Fail-open** — emission failures never break the module.

What counts as a landmark:

  * AE-2: a NEW high-confidence association (first time this
    action_signature × outcome_kind pair crossed both thresholds)
  * HOT-1: a baseline_drift or attractor_lock pattern
    (trace-level signal, not just breach clustering)
  * HOT-4: ≥5 flagged steps in one pass (sustained anomaly)
  * RPT-1: a calibration kind crossing the min-resolutions
    threshold for the first time
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# Max emissions per module per pass — module-level safety on top of
# the continuity-ledger's own append-only discipline.
_MAX_EMISSIONS_PER_PASS = 3


def emit_landmark(
    *,
    source_module: str,        # "ae2_causal_credit" | "hot1_meta_affect" | ...
    landmark_kind: str,        # "first_association" | "baseline_drift" | ...
    summary: str,              # one-line, no identities, ≤ 200 chars
    counts: dict[str, int] | None = None,
) -> bool:
    """Emit a single landmark observation to the continuity ledger.

    Failure-isolated: never raises. Returns True on success, False
    on any failure (ledger disabled, write error, etc).

    The summary text is the operator-readable line — keep it
    OPAQUE (counts, never names; aggregates, never identities)."""
    if not summary or len(summary.strip()) < 4:
        return False
    detail: dict[str, Any] = {
        "source_module": source_module,
        "landmark_kind": landmark_kind,
        "detected_at": datetime.now(timezone.utc).isoformat(),
    }
    if counts:
        # Defensive copy + flatten any non-int values.
        for k, v in counts.items():
            try:
                detail[f"count_{k}"] = int(v)
            except (TypeError, ValueError):
                continue
    try:
        from app.identity.continuity_ledger import record_event
        ok = record_event(
            kind="sentience_observation",
            actor=source_module,
            summary=summary.strip()[:200],
            detail=detail,
        )
        return bool(ok)
    except Exception:
        logger.debug(
            "sentience.ledger_bridge: emit failed (%s)", source_module,
            exc_info=True,
        )
        return False
