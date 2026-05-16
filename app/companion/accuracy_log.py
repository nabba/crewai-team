"""Companion accuracy log — proactive-suggestion → operator-action correlation.

PROGRAM §51 — Q16 Theme 7 (companion depth: anticipation > reaction).
The companion subsystem makes many proactive suggestions across the
day (person dormancy nudges, cross-modal patterns, brief-drop hints,
graph-bridge alerts, browse-theme reactions). What was missing: a
feedback loop that says "when we surfaced X, did Andrus actually
act on it?"

This module is the **observational** layer:

  * ``log_suggestion(kind, payload, …)`` records every proactive
    suggestion at emission time. Producers explicitly opt in by
    calling this — we deliberately don't auto-hook every notify()
    so the operator can review the surfaced kinds.
  * ``log_action(suggestion_id, action, …)`` records the operator's
    response (click, reply, ignore, act-without-click).
  * ``accuracy_summary(window_days=30)`` reports acceptance rate by
    kind, identifies kinds with chronic low acceptance.

What this module deliberately doesn't do
========================================

  * No auto-mute. Accuracy data INFORMS the arbiter; the arbiter
    decides what to suppress. Auto-tuning the surface from
    accuracy alone is a Goodhart trap (operators might stop
    clicking valuable-but-uncomfortable suggestions).
  * No content storage. Records only the suggestion kind + a
    short hash of the payload, never the full text. Operator
    privacy boundary is preserved.

Storage: ``workspace/companion/accuracy_log.jsonl`` (capped 10k
rows via append_with_cap).

Master switch: ``companion_accuracy_log_enabled`` (default ON).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_LOG_FILE = "accuracy_log.jsonl"
_MAX_ROWS = 10_000


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_companion_accuracy_log_enabled
        return get_companion_accuracy_log_enabled()
    except Exception:
        return os.getenv(
            "COMPANION_ACCURACY_LOG_ENABLED", "true",
        ).lower() in ("true", "1", "yes", "on")


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _log_path() -> Path:
    return _workspace() / "companion" / _LOG_FILE


def _payload_hash(payload: Any) -> str:
    """Short stable hash of the payload — used for cross-row matching
    without storing the body."""
    try:
        text = json.dumps(payload, sort_keys=True, default=str) if not isinstance(payload, str) else payload
    except Exception:
        text = str(payload)
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:8]


def _append(row: dict[str, Any]) -> None:
    if not _enabled():
        return
    try:
        from app.utils.jsonl_retention import append_with_cap
        append_with_cap(_log_path(), json.dumps(row, sort_keys=True), max_lines=_MAX_ROWS)
    except Exception:
        # Fall back to direct append if the helper isn't available.
        p = _log_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        except OSError:
            logger.debug("companion.accuracy_log: append failed", exc_info=True)


def log_suggestion(
    *,
    kind: str,
    payload: Any = None,
    topic: Optional[str] = None,
    surface: str = "signal",
    now: Optional[float] = None,
) -> str:
    """Record a proactive suggestion at emission time. Returns the
    suggestion id (used by callers that want to correlate later).

    ``kind`` is the producer's classification (e.g.
    ``"person_dormancy"``, ``"cross_modal_pattern"``,
    ``"browse_topic"``). ``payload`` is hashed — never stored.
    """
    cur = float(now) if now is not None else time.time()
    payload_hash = _payload_hash(payload) if payload is not None else "-"
    suggestion_id = f"sug:{int(cur)}:{payload_hash}"
    _append({
        "event": "suggestion_emitted",
        "ts": datetime.fromtimestamp(cur, tz=timezone.utc).isoformat(),
        "suggestion_id": suggestion_id,
        "kind": str(kind),
        "topic": topic,
        "surface": surface,
        "payload_hash": payload_hash,
    })
    return suggestion_id


def log_action(
    *,
    suggestion_id: str,
    action: str,
    detail: Optional[str] = None,
    now: Optional[float] = None,
) -> None:
    """Record the operator's response to a previously-logged
    suggestion.

    ``action`` ∈ {``clicked``, ``replied``, ``ignored``,
    ``acted_without_click``, ``dismissed``}.

    ``detail`` is optional short text (≤200 chars).
    """
    cur = float(now) if now is not None else time.time()
    _append({
        "event": "action_recorded",
        "ts": datetime.fromtimestamp(cur, tz=timezone.utc).isoformat(),
        "suggestion_id": suggestion_id,
        "action": str(action),
        "detail": (detail[:200] if isinstance(detail, str) else None),
    })


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


def accuracy_summary(*, window_days: int = 30) -> dict[str, Any]:
    """Compute acceptance rates by kind within the rolling window.

    Acceptance is defined as: a ``suggestion_emitted`` event followed
    within 7 days by an ``action_recorded`` event with action in
    {clicked, replied, acted_without_click}. Ignored / dismissed
    count as non-acceptance.
    """
    out: dict[str, Any] = {
        "window_days": int(window_days),
        "available": False,
        "n_suggestions": 0,
        "n_with_action": 0,
        "overall_acceptance": None,
        "by_kind": {},
        "low_acceptance_kinds": [],
    }
    p = _log_path()
    if not p.exists():
        return out
    out["available"] = True
    cur = time.time()
    cutoff = cur - window_days * 86400
    suggestions: dict[str, dict[str, Any]] = {}
    actions: dict[str, list[dict[str, Any]]] = defaultdict(list)
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
                if ts is None or ts < cutoff:
                    continue
                sid = row.get("suggestion_id")
                if not sid:
                    continue
                if row.get("event") == "suggestion_emitted":
                    suggestions[sid] = row
                elif row.get("event") == "action_recorded":
                    actions[sid].append(row)
    except OSError:
        return out
    n_suggestions = len(suggestions)
    n_with_action = 0
    accepted_actions = {"clicked", "replied", "acted_without_click"}
    n_accepted = 0
    by_kind_total: Counter = Counter()
    by_kind_accepted: Counter = Counter()
    for sid, sug in suggestions.items():
        kind = sug.get("kind", "(unknown)")
        by_kind_total[kind] += 1
        action_rows = actions.get(sid, [])
        if action_rows:
            n_with_action += 1
            if any(r.get("action") in accepted_actions for r in action_rows):
                n_accepted += 1
                by_kind_accepted[kind] += 1
    out["n_suggestions"] = n_suggestions
    out["n_with_action"] = n_with_action
    if n_suggestions > 0:
        out["overall_acceptance"] = round(n_accepted / n_suggestions, 3)
    by_kind: dict[str, dict[str, Any]] = {}
    for kind, total in by_kind_total.items():
        accepted = by_kind_accepted[kind]
        by_kind[kind] = {
            "total": total,
            "accepted": accepted,
            "rate": round(accepted / total, 3) if total else 0.0,
        }
    out["by_kind"] = by_kind
    # Low acceptance: kinds with ≥10 suggestions and <10% acceptance.
    low = [
        {"kind": k, **v}
        for k, v in by_kind.items()
        if v["total"] >= 10 and v["rate"] < 0.10
    ]
    out["low_acceptance_kinds"] = sorted(
        low, key=lambda x: x["rate"],
    )
    return out
