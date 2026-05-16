"""HOT-1 consultation — read prior metacognitive-repair observations
before generating a new one.

PROGRAM §51 — Q16 Theme 8 (sentience/creativity workstream: consume,
don't just observe). The structured_diagnosis pipeline emits a
metacognitive-repair observation on every attempt — but, prior to
this module, nothing READ those observations back. The pipeline
generated a new hypothesis from scratch each time, even if the same
pattern had been attempted (and failed) ten times before.

This module is the missing consumer. It exposes ``consult`` which
returns recent prior-attempt context for a given pattern signature:

  * Total prior attempts (filed + declined)
  * Resolution outcomes (filed → applied, applied → rolled-back, etc.)
  * Confidence trajectory (rising, flat, declining)
  * Repeat-decline rate (how often the LLM has declined this kind)

The structured_diagnosis pipeline uses this context to:

  (a) Skip cheaply when the pattern has chronically failed
      (≥3 prior declines, no successes) — saves LLM spend.
  (b) Prepend a "what's been tried before" hint to the LLM prompt
      so the diagnosis isn't blind to history.

Failure-isolated: any read error returns an empty advisory dict;
the pipeline falls back to the no-context path.

Master switch: ``hot1_consultation_enabled`` (default ON).
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_DEFAULT_WINDOW_DAYS = 60
_OBSERVATION_FILE_ENV = "HOT1_OBSERVATION_LOG"


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_hot1_consultation_enabled
        return get_hot1_consultation_enabled()
    except Exception:
        return os.getenv(
            "HOT1_CONSULTATION_ENABLED", "true",
        ).lower() in ("true", "1", "yes", "on")


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _observation_path() -> Path:
    return Path(
        os.environ.get(_OBSERVATION_FILE_ENV)
        or str(_workspace() / "subia" / "observations" / "metacognitive_repair.jsonl")
    )


def _parse_iso(s: Any) -> Optional[float]:
    if not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _iter_rows(
    *,
    pattern_signature: str,
    cutoff_ts: float,
) -> list[dict[str, Any]]:
    """Walk the observation log, return matching rows (newest last)."""
    p = _observation_path()
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                ts = _parse_iso(row.get("ts"))
                if ts is None or ts < cutoff_ts:
                    continue
                origin = row.get("originating_error") or {}
                if origin.get("pattern_signature") != pattern_signature:
                    continue
                row["_ts"] = ts
                out.append(row)
    except OSError:
        return []
    return out


def consult(
    *,
    pattern_signature: str,
    file_path: str,
    window_days: int = _DEFAULT_WINDOW_DAYS,
    now: Optional[float] = None,
) -> dict[str, Any]:
    """Return prior-attempt context for ``pattern_signature``.

    Returns a dict with keys:
      * ``available`` (bool) — false if the module is disabled OR
        no observations exist.
      * ``window_days`` (int)
      * ``n_attempts`` (int) — total filed + declined
      * ``n_filed`` (int) — proposed CR (not declined)
      * ``n_declined`` (int) — LLM declined
      * ``n_resolved`` (int) — outcome recorded
      * ``n_applied`` (int) — outcome=APPLIED
      * ``n_rolled_back`` (int) — outcome=ROLLED_BACK
      * ``mean_confidence`` (float | None) — across filed attempts
      * ``last_attempt_at`` (str | None)
      * ``recommendation`` (str) — one of "skip" / "proceed_with_caveat"
        / "proceed_normally"
      * ``hint_for_prompt`` (str | None) — short text to splice into
        the LLM prompt when ``recommendation != "skip"``.
    """
    out: dict[str, Any] = {
        "available": False,
        "window_days": int(window_days),
        "n_attempts": 0,
        "n_filed": 0,
        "n_declined": 0,
        "n_resolved": 0,
        "n_applied": 0,
        "n_rolled_back": 0,
        "mean_confidence": None,
        "last_attempt_at": None,
        "recommendation": "proceed_normally",
        "hint_for_prompt": None,
    }
    if not _enabled():
        return out
    cur = float(now) if now is not None else time.time()
    cutoff = cur - window_days * 86400
    rows = _iter_rows(pattern_signature=pattern_signature, cutoff_ts=cutoff)
    if not rows:
        return out
    out["available"] = True
    out["n_attempts"] = len(rows)
    confidences: list[float] = []
    for row in rows:
        hot = row.get("higher_order_thought") or {}
        if hot.get("declined"):
            out["n_declined"] += 1
        else:
            out["n_filed"] += 1
            conf = hot.get("self_assessed_confidence")
            if isinstance(conf, (int, float)):
                confidences.append(float(conf))
        outcome = (row.get("outcome") or "").lower() if isinstance(row.get("outcome"), str) else ""
        if outcome:
            out["n_resolved"] += 1
        if outcome == "applied":
            out["n_applied"] += 1
        elif outcome == "rolled_back":
            out["n_rolled_back"] += 1
    if confidences:
        out["mean_confidence"] = round(sum(confidences) / len(confidences), 3)
    last_ts = max(r["_ts"] for r in rows)
    out["last_attempt_at"] = datetime.fromtimestamp(
        last_ts, tz=timezone.utc,
    ).isoformat()

    # Recommendation heuristic.
    if out["n_declined"] >= 3 and out["n_applied"] == 0:
        out["recommendation"] = "skip"
        out["hint_for_prompt"] = (
            f"This pattern has been declined {out['n_declined']} times "
            f"in the last {window_days} days with no successful "
            f"applies. The fix likely needs operator-led work."
        )
    elif out["n_rolled_back"] >= 1 and out["n_applied"] == 0:
        out["recommendation"] = "proceed_with_caveat"
        out["hint_for_prompt"] = (
            f"Caveat: a prior fix for this pattern was applied and "
            f"then rolled back. The first hypothesis was wrong — "
            f"think hard about the SECOND-order cause before "
            f"proposing the same shape of fix."
        )
    elif out["n_applied"] >= 1:
        # We've succeeded before — reduce caution.
        out["recommendation"] = "proceed_normally"
        out["hint_for_prompt"] = (
            f"Note: similar fixes for this pattern have been applied "
            f"successfully {out['n_applied']} times before. If this "
            f"instance looks similar, follow the same shape."
        )
    elif out["n_attempts"] >= 2:
        out["recommendation"] = "proceed_with_caveat"
        out["hint_for_prompt"] = (
            f"Caveat: {out['n_attempts']} prior attempts for this "
            f"pattern; none have a recorded outcome yet. Be explicit "
            f"about what would distinguish this hypothesis from the "
            f"prior ones."
        )
    return out
