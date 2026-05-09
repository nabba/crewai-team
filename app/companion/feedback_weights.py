"""Feedback → workspace-selection weight downweighting (Phase D #4).

Closes a small but real loop the original Wave 2 plan called for:
when the operator gives a 👎 on a Companion idea or related Signal
output for workspace X, the scheduler's next pick should weight
X *down* — not block, just bias away from it for a window.

Design constraints:

  * Must respect ``app/companion/scheduler.py`` as the policy layer.
    We ADD a multiplicative factor on the candidate's CFS weight, not
    replace the underlying grounding-derived weight.
  * Must be bounded — a single 👎 cannot push weight to 0. The
    multiplier floors at ``MIN_MULTIPLIER`` (default 0.4).
  * Must decay — sentiment from a week ago shouldn't depress today's
    selection. Decay halflife is configurable (default 3 days).
  * Must be best-effort — if the storage file is unreadable or the
    feedback router hasn't fired yet, the multiplier defaults to 1.0
    (no change to baseline scheduling).

State at ``workspace/companion/feedback_weights.json``::

    {
      "<workspace_id>": {
        "down_weight_until_ts": <epoch_seconds>,
        "down_count": <int>,
        "last_updated_at": <iso>
      },
      ...
    }

Wiring:

  * ``feedback_router.py`` (Phase B #3) calls ``record_negative(workspace_id)``
    in its companion-sink branch when the polarity is 👎.
  * ``scheduler.collect_candidates`` calls ``current_multiplier(workspace_id)``
    and multiplies the existing affect-weight by the result.

The multiplier formula:

    base = 1.0 - 0.2 × down_count             (each 👎 -0.2)
    decay = 0.5 ** (age_days / halflife_days) (exponential decay)
    multiplier = max(MIN_MULTIPLIER, base + (1 - base) × (1 - decay))

So:
  * 0 thumbs-down       → 1.0
  * 1 thumbs-down today → 0.8
  * 1 thumbs-down 3d ago → 0.9 (half-decayed)
  * 3 thumbs-down today → 0.4 (floored)

Master switch: ``COMPANION_FEEDBACK_WEIGHTS_ENABLED`` (default ON).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_STATE_PATH = Path("/app/workspace/companion/feedback_weights.json")
_RETENTION_DAYS = 30
_HALFLIFE_DAYS = 3.0
_DOWN_PENALTY = 0.2
_MIN_MULTIPLIER = 0.4

_lock = threading.Lock()


def _enabled() -> bool:
    return os.getenv("COMPANION_FEEDBACK_WEIGHTS_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _read_state() -> dict[str, Any]:
    if not _STATE_PATH.exists():
        return {}
    try:
        data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.debug("feedback_weights: state read failed", exc_info=True)
        return {}


def _write_state(data: dict[str, Any]) -> None:
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _STATE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(_STATE_PATH)
    except OSError:
        logger.debug("feedback_weights: state write failed", exc_info=True)


def record_negative(workspace_id: str) -> None:
    """Note a thumbs-down on workspace ``workspace_id``. Best-effort."""
    if not _enabled():
        return
    if not workspace_id:
        return
    now = time.time()
    with _lock:
        data = _read_state()
        entry = data.setdefault(workspace_id, {
            "down_count": 0, "last_updated_at": "", "first_observed_at": now,
        })
        entry["down_count"] = int(entry.get("down_count", 0)) + 1
        entry["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        # Track first observed for decay calc.
        entry.setdefault("first_observed_at", now)
        # Each new 👎 refreshes the "first observed" anchor toward now —
        # but only by half, so a steady stream of 👎 keeps weight pinned
        # while a single old 👎 decays cleanly.
        entry["first_observed_at"] = (
            float(entry["first_observed_at"]) + now
        ) / 2.0
        # Drop entries that have been at multiplier=1.0 for a long time
        # — see _gc inline.
        data = _gc(data, now)
        _write_state(data)


def record_positive(workspace_id: str) -> None:
    """Operator 👍 partially counteracts past downvotes. -1 to count, floored."""
    if not _enabled() or not workspace_id:
        return
    with _lock:
        data = _read_state()
        if workspace_id not in data:
            return
        entry = data[workspace_id]
        entry["down_count"] = max(0, int(entry.get("down_count", 0)) - 1)
        if entry["down_count"] == 0:
            data.pop(workspace_id, None)
        _write_state(data)


def current_multiplier(workspace_id: str) -> float:
    """Return the multiplier in ``[MIN_MULTIPLIER, 1.0]`` for this workspace.

    ``1.0`` when no negative feedback / decayed away. Reads only — never
    raises.
    """
    if not _enabled() or not workspace_id:
        return 1.0
    try:
        data = _read_state()
    except Exception:
        return 1.0
    entry = data.get(workspace_id)
    if not entry:
        return 1.0

    down_count = int(entry.get("down_count", 0) or 0)
    if down_count <= 0:
        return 1.0
    first_observed = float(entry.get("first_observed_at", 0.0) or 0.0)
    age_days = (time.time() - first_observed) / 86400 if first_observed else 0.0

    base = max(0.0, 1.0 - _DOWN_PENALTY * down_count)
    if _HALFLIFE_DAYS <= 0:
        decay = 0.0
    else:
        decay = 0.5 ** (age_days / _HALFLIFE_DAYS)
    # Blend: more decay → multiplier rises back toward 1.0.
    multiplier = base + (1.0 - base) * (1.0 - decay)
    return max(_MIN_MULTIPLIER, min(1.0, multiplier))


def _gc(data: dict[str, Any], now: float) -> dict[str, Any]:
    """Drop entries older than ``_RETENTION_DAYS`` whose multiplier is 1.0."""
    cutoff = now - _RETENTION_DAYS * 86400
    out = {}
    for k, v in data.items():
        first = float((v or {}).get("first_observed_at", now) or now)
        if first < cutoff and int((v or {}).get("down_count", 0) or 0) <= 0:
            continue
        out[k] = v
    return out


def reset() -> None:
    """Operator-callable nuke. For tests + manual recovery."""
    with _lock:
        _write_state({})
