"""Topic-level feedback weights (Phase G #2, 2026-05-09).

Phase D #4 added workspace-level downweighting (👎 reduces
``app/companion/scheduler.py`` selection probability for a workspace).
But it left a gap: a 👎 on "I dislike forest carbon" reduces the
*workspace* weight, not the *topic* weight in interest_model. The
operator wanted "👎 on a topic downweights that topic's selection
in the next idle cycle."

This module fills that gap. Parallel structure to ``feedback_weights``:

  * ``record_negative(topic_text, comment)`` — extract candidate topics
    from the comment text + look them up against the live
    interest_profile; downweight each match.
  * ``record_positive(topic_text)`` — partially counteract.
  * ``current_multiplier(topic)`` — read multiplier ∈ [0.4, 1.0] for
    use by interest_model when applying scores.

Storage: ``workspace/companion/topic_weights.json``::

    {"<lowered topic>": {"down_count": <int>,
                          "first_observed_at": <epoch>,
                          "last_updated_at": <iso>}}

Decay halflife: 7 days (longer than workspace weights — topic taste
moves slower than workspace cadence).

Master switch: ``COMPANION_TOPIC_WEIGHTS_ENABLED`` (default ON).

Wiring:
  * ``feedback_router._dispatch`` calls ``record_negative_from_comment``
    when a 👎 reaction includes ``original_response`` text.
  * ``interest_model._score_terms`` multiplies each term's score by
    ``current_multiplier(term)`` before accumulating.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


_STATE_PATH = Path("/app/workspace/companion/topic_weights.json")
_HALFLIFE_DAYS = 7.0
_DOWN_PENALTY = 0.2
_MIN_MULTIPLIER = 0.4
_RETENTION_DAYS = 60

_lock = threading.Lock()


def _enabled() -> bool:
    return os.getenv("COMPANION_TOPIC_WEIGHTS_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _read_state() -> dict[str, Any]:
    if not _STATE_PATH.exists():
        return {}
    try:
        data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.debug("topic_weights: state read failed", exc_info=True)
        return {}


def _write_state(data: dict[str, Any]) -> None:
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _STATE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(_STATE_PATH)
    except OSError:
        logger.debug("topic_weights: state write failed", exc_info=True)


def _normalize(topic: str) -> str:
    return (topic or "").strip().lower()


# ── Topic-extraction from comments ───────────────────────────────────────


def _candidate_topics_in_comment(comment: str) -> list[str]:
    """Find topics from the live interest_profile that appear in
    ``comment``. Returns the matched topic names (in profile order).

    Why through the profile rather than free-form NLP: we only want
    to downweight topics the system ALREADY tracks. A 👎 mentioning
    "estonia winter" only matters if "estonia winter" is a profile
    topic — otherwise the downweight has nothing to apply to.
    """
    if not comment:
        return []
    try:
        from app.companion.interest_model import current_profile
    except Exception:
        return []
    try:
        profile = current_profile()
    except Exception:
        return []
    cl = comment.lower()
    out: list[str] = []
    for t in profile.get("topics") or []:
        if not isinstance(t, dict):
            continue
        name = (t.get("name") or "").strip().lower()
        if name and name in cl:
            out.append(name)
    return out


# ── Public API ───────────────────────────────────────────────────────────


def record_negative(topic: str) -> None:
    """Note a thumbs-down on ``topic``. Best-effort. ``topic`` is
    case-insensitive; extra whitespace stripped."""
    if not _enabled():
        return
    norm = _normalize(topic)
    if not norm:
        return
    now = time.time()
    with _lock:
        data = _read_state()
        entry = data.setdefault(norm, {
            "down_count": 0, "first_observed_at": now,
            "last_updated_at": "",
        })
        entry["down_count"] = int(entry.get("down_count", 0)) + 1
        entry["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        # Average the anchor toward now (matches feedback_weights
        # behaviour — repeated 👎s pull the decay anchor forward
        # without resetting it entirely).
        entry["first_observed_at"] = (
            float(entry.get("first_observed_at", now)) + now
        ) / 2.0
        data = _gc(data, now)
        _write_state(data)


def record_positive(topic: str) -> None:
    """Operator 👍 partially counteracts past downvotes."""
    if not _enabled():
        return
    norm = _normalize(topic)
    if not norm:
        return
    with _lock:
        data = _read_state()
        if norm not in data:
            return
        entry = data[norm]
        entry["down_count"] = max(0, int(entry.get("down_count", 0)) - 1)
        if entry["down_count"] == 0:
            data.pop(norm, None)
        _write_state(data)


def record_negative_from_comment(comment: str) -> list[str]:
    """Extract topic mentions from a 👎 comment + record each.

    Returns the list of topics downweighted (for audit).
    """
    matches = _candidate_topics_in_comment(comment)
    for t in matches:
        record_negative(t)
    return matches


def current_multiplier(topic: str) -> float:
    """Multiplier ∈ [MIN_MULTIPLIER, 1.0] for ``topic``. ``1.0`` when
    no negative feedback / decayed away."""
    if not _enabled():
        return 1.0
    norm = _normalize(topic)
    if not norm:
        return 1.0
    try:
        data = _read_state()
    except Exception:
        return 1.0
    entry = data.get(norm)
    if not entry:
        return 1.0
    down_count = int(entry.get("down_count", 0) or 0)
    if down_count <= 0:
        return 1.0
    first = float(entry.get("first_observed_at", 0.0) or 0.0)
    if first == 0.0:
        return 1.0
    age_days = max(0.0, (time.time() - first) / 86400)
    base = max(0.0, 1.0 - _DOWN_PENALTY * down_count)
    decay = 0.5 ** (age_days / _HALFLIFE_DAYS) if _HALFLIFE_DAYS > 0 else 0.0
    multiplier = base + (1.0 - base) * (1.0 - decay)
    return max(_MIN_MULTIPLIER, min(1.0, multiplier))


def _gc(data: dict[str, Any], now: float) -> dict[str, Any]:
    cutoff = now - _RETENTION_DAYS * 86400
    out = {}
    for k, v in data.items():
        first = float((v or {}).get("first_observed_at", now) or now)
        if first < cutoff and int((v or {}).get("down_count", 0) or 0) <= 0:
            continue
        out[k] = v
    return out


def reset() -> None:
    """Operator-callable nuke. Tests + manual recovery."""
    with _lock:
        _write_state({})
