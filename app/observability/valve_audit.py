"""
valve_audit.py — Shared "reducing-valve audit" rejection logger.

Hypothesis (Huxley framing): the system has many filters (relevance,
salience, refusal detection, quality gates, novelty, surfacing) and at
least one is probably too narrow in a way no individual filter knows.
This module is the measurement-only instrumentation: each filter calls
`log_rejection(...)` on its rejection path; a separate daily replay job
(see `valve_audit_replay.py`) re-evaluates a sample and surfaces filters
that drop too aggressively.

Constraints:
    - Measurement only. NEVER changes filter behaviour.
    - Out of scope: safety_guardian, eval_sandbox, sanitiser Tier-1
      hard-rejects, governance gates. Those filters are intentionally
      narrow and are NOT instrumented.
    - Filters in TIER_IMMUTABLE files (vetting.py) must be instrumented
      at the CALLER, never inside the immutable file itself.

Output:
    /app/workspace/logs/valve_audit.jsonl
    Append-only, one rejection per line. Daily rotation handled by the
    replay job's housekeeping (or by ops infra; this module just appends).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy path resolution so tests can monkeypatch app.paths before first call.
_LOG_LOCK = threading.Lock()

# Cap raw input_text per row to keep storage bounded.
_INPUT_TEXT_CAP_BYTES = 4096

# Master kill-switch — env var lets ops disable instrumentation without
# shipping a code change. Defaults to ON so the audit gathers data the
# moment it's deployed.
_AUDIT_ENABLED_ENV = "VALVE_AUDIT_ENABLED"


# ── Public API ───────────────────────────────────────────────────────────────

def log_rejection(
    *,
    filter_id: str,
    callsite: str,
    input_text: str | None = None,
    reason: str = "",
    score: float | None = None,
    threshold: float | None = None,
    model_used: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append one rejection event to the audit log.

    Fire-and-forget — never raises. Called from filter rejection paths.

    Args:
        filter_id: Stable id (e.g. "F1", "F4", "F8") matching the
            inventory in the audit plan. Operators look up the predicate
            by filter_id, so this MUST be stable across deploys.
        callsite: Format "file.py:line" — the line that produced the
            rejection. Only used for human auditing.
        input_text: The text/material that was rejected. Truncated to
            _INPUT_TEXT_CAP_BYTES; full text replays are out of scope.
            Hashed (sha1[:16]) so duplicate rejections can be deduped
            cheaply at replay time.
        reason: Categorical reason string ("confidence_below_threshold",
            "below_novelty", etc). Same-meaning rejections from the same
            filter MUST use the same reason — replay aggregates by it.
        score: The score the predicate computed (if any).
        threshold: The threshold the score failed against (if any).
        model_used: Identifier of the model whose output was filtered
            (if applicable — None for purely heuristic filters).
        extra: Filter-specific extra fields; merged into the JSONL row
            under "extra". Keep payloads small.
    """
    if not _enabled():
        return
    try:
        path = _log_path()
        text = (input_text or "")
        if len(text) > _INPUT_TEXT_CAP_BYTES:
            text = text[:_INPUT_TEXT_CAP_BYTES]
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "filter_id": filter_id,
            "callsite": callsite,
            "input_text": text,
            "input_hash": hashlib.sha1(
                (input_text or "").encode("utf-8", errors="replace")
            ).hexdigest()[:16],
            "reason": reason,
            "score": score,
            "threshold": threshold,
            "model_used": model_used,
            "extra": extra or {},
        }
        line = json.dumps(row, default=str)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        # Audit MUST NOT raise into the filter's hot path. Worst case
        # we lose a row.
        logger.debug("valve_audit: log_rejection failed", exc_info=True)


def log_path() -> Path:
    """Public accessor for the current log path (for tests / replay)."""
    return _log_path()


def is_enabled() -> bool:
    return _enabled()


# ── Internals ────────────────────────────────────────────────────────────────

def _enabled() -> bool:
    val = os.environ.get(_AUDIT_ENABLED_ENV, "1").strip().lower()
    return val not in {"0", "false", "no", "off"}


def _log_path() -> Path:
    """Resolve the log path lazily so tests can monkeypatch app.paths."""
    try:
        from app.paths import WORKSPACE_ROOT
        return WORKSPACE_ROOT / "logs" / "valve_audit.jsonl"
    except Exception:
        # Fallback for environments where app.paths can't initialise.
        return Path("/tmp/valve_audit.jsonl")
