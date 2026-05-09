"""Notify-metadata sidechannel — correlate Signal reactions back to source.

Phase B #3 (2026-05-09). When ``app.notify.api.notify(...)`` sends a
completion ping, callers can attach metadata identifying the source
(skill id, meta-agent recipe id, task id, idea id). We record the
``(send_ts, metadata)`` pair in this sidechannel.

Later, when a Signal reaction arrives on that message, the
``feedback_router`` walks ``feedback.events`` rows joined against this
sidechannel by timestamp — giving us the source identifier we need to
update skill counters, recipe ledger outcomes, and the companion event
log.

Why not extend ``feedback_pipeline``? It's TIER_IMMUTABLE (security
boundary). Sidechannel is a plain JSONL the mutable consumer can read
without touching the IMMUTABLE writer.

File format::

    workspace/companion/notify_meta.jsonl
    {"ts": 1715200000123, "metadata": {"skill_id": "...", ...}}
    {"ts": 1715200005456, "metadata": {"recipe_id": "...", ...}}

Retention: 14 days. Reactions arriving later than that are silently
ignored — the long tail of reactions on month-old messages isn't
useful for the learning loop anyway.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_META_PATH = Path("/app/workspace/companion/notify_meta.jsonl")
_RETENTION_DAYS = 14
_MAX_LOOKUP_WINDOW_S = 5  # ts must match within ±5 s — Signal-cli timestamps are ms-precise

_lock = threading.Lock()


def record(send_ts: int, metadata: dict[str, Any]) -> None:
    """Append a ``(send_ts, metadata)`` row. Best-effort; never raises.

    ``send_ts`` is the Signal message timestamp returned by signal-cli
    (milliseconds since epoch). ``metadata`` is opaque to this module —
    typical keys: ``skill_id``, ``recipe_id``, ``task_id``, ``idea_id``,
    ``workspace_id``.
    """
    if not metadata or not send_ts:
        return
    entry = {"ts": int(send_ts), "metadata": metadata}
    try:
        with _lock:
            _META_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _META_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, sort_keys=True))
                f.write("\n")
    except Exception:
        logger.debug("notify_meta: record failed", exc_info=True)


def lookup(target_ts: int, window_s: int = _MAX_LOOKUP_WINDOW_S) -> dict | None:
    """Return the metadata for the row whose ts matches ``target_ts``
    within ``window_s``. None if no match.

    Signal-cli timestamps are usually exact (ms-precise) but a small
    window absorbs clock skew between the bot's wall-clock and the
    forwarder's reported send timestamp.
    """
    if not _META_PATH.exists():
        return None
    target_ms = int(target_ts)
    window_ms = int(window_s * 1000)
    try:
        with _META_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                ts = int(row.get("ts", 0))
                if abs(ts - target_ms) <= window_ms:
                    return row.get("metadata") or None
    except Exception:
        logger.debug("notify_meta: lookup failed", exc_info=True)
    return None


def prune(retention_days: int = _RETENTION_DAYS) -> int:
    """Rewrite the file dropping rows older than ``retention_days``.
    Returns the count dropped. Best-effort.
    """
    if not _META_PATH.exists():
        return 0
    cutoff_ms = int((time.time() - retention_days * 86400) * 1000)
    kept: list[str] = []
    dropped = 0
    try:
        with _META_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                    ts = int(row.get("ts", 0))
                except Exception:
                    # Garbage line — drop it.
                    dropped += 1
                    continue
                if ts >= cutoff_ms:
                    kept.append(stripped)
                else:
                    dropped += 1
        with _lock:
            tmp = _META_PATH.with_suffix(".jsonl.tmp")
            tmp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
            tmp.replace(_META_PATH)
    except Exception:
        logger.debug("notify_meta: prune failed", exc_info=True)
    return dropped
