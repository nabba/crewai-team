"""governance_signal_bridge.py — Signal timestamp ↔ governance request map.

Bridges the existing TIER_IMMUTABLE `control_plane.governance` approval
queue to Signal 👍/👎 reactions. The governance table itself has no
``signal_ts`` column (and is TIER_IMMUTABLE, so adding one requires the
Tier-3 amendment protocol). This module keeps a small JSON sidecar map
so the reaction handler in ``main.py`` can resolve a reacted-to message
timestamp back to its governance request UUID.

Why a JSON sidecar instead of a Postgres column:
 - Postgres column would touch ``governance.py`` (TIER_IMMUTABLE)
 - Migrations under ``migrations/*.sql`` are not auto-applied at boot
   (``startup_migrations.apply_all`` only handles pgvector HNSW indexes)
 - The map is purely a notification-routing aid; loss of the file just
   means the operator falls back to the text command or React dashboard
 - Entries expire fast — governance requests default to 24h TTL

Used by:
 - ``app.auto_deployer.schedule_deploy`` — calls ``register`` after the
   approval-needed message is sent (via ``send_message_blocking`` so
   the Signal timestamp is captured).
 - ``app.main`` reaction handler — calls ``find_request_id`` to resolve
   the reaction target.
 - ``app.agents.commander.commands`` text-command path — calls
   ``find_pending_by_id_prefix`` for ``approve <hex>``.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.paths import WORKSPACE_ROOT
from app.safe_io import safe_write_json

logger = logging.getLogger(__name__)

# Entries older than this are purged on every access. 25h gives a small
# margin over the governance default 24h TTL so we don't drop entries
# that are still actionable.
_MAX_AGE_SECONDS = 25 * 3600

_LOCK = threading.Lock()


def _bridge_path() -> Path:
    return WORKSPACE_ROOT / "governance_signal_bridge.json"


def _load() -> dict:
    p = _bridge_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text() or "{}")
    except Exception:
        logger.debug("governance_signal_bridge: load failed; starting fresh", exc_info=True)
        return {}


def _save(data: dict) -> None:
    safe_write_json(_bridge_path(), data)


def _purge_expired(data: dict) -> dict:
    """Drop entries older than _MAX_AGE_SECONDS. Returns new dict."""
    now = datetime.now(timezone.utc).timestamp()
    kept = {}
    for ts_str, entry in data.items():
        try:
            created = float(entry.get("created_at_epoch") or 0)
            if (now - created) <= _MAX_AGE_SECONDS:
                kept[ts_str] = entry
        except Exception:
            continue
    return kept


def register(signal_ts: int, request_id: str) -> None:
    """Record a (signal_ts → governance_request_id) mapping.

    Called from auto_deployer right after the approval-needed Signal
    message is sent. Fire-and-forget — any failure is logged and
    swallowed so the deploy path stays alive.
    """
    if not signal_ts or not request_id:
        return
    try:
        with _LOCK:
            data = _purge_expired(_load())
            data[str(int(signal_ts))] = {
                "request_id": str(request_id),
                "created_at_epoch": datetime.now(timezone.utc).timestamp(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            _save(data)
    except Exception:
        logger.debug("governance_signal_bridge.register failed", exc_info=True)


def find_request_id(signal_ts: int) -> str | None:
    """Return the governance request_id for a Signal timestamp, or None.

    Called from the reaction handler in main.py. None means the
    reaction wasn't on a tracked governance message — caller falls
    through to other reaction handlers / feedback pipeline.

    Also persists the post-purge state if expired entries were dropped,
    so the file doesn't grow unbounded under read-heavy workloads.
    """
    if not signal_ts:
        return None
    try:
        with _LOCK:
            raw = _load()
            kept = _purge_expired(raw)
            if len(kept) != len(raw):
                _save(kept)
            entry = kept.get(str(int(signal_ts)))
            if entry:
                return str(entry.get("request_id") or "") or None
    except Exception:
        logger.debug("governance_signal_bridge.find_request_id failed", exc_info=True)
    return None


def unregister(request_id: str) -> None:
    """Drop any entries pointing at this request_id (post-resolution cleanup)."""
    if not request_id:
        return
    try:
        with _LOCK:
            data = _load()
            kept = {
                ts: entry for ts, entry in data.items()
                if str(entry.get("request_id") or "") != str(request_id)
            }
            if len(kept) != len(data):
                _save(kept)
    except Exception:
        logger.debug("governance_signal_bridge.unregister failed", exc_info=True)


def find_pending_by_id_prefix(id_prefix: str) -> dict | None:
    """Find a single pending governance request whose UUID starts with id_prefix.

    Used by the ``approve <hex>`` text-command fallback in
    ``agents/commander/commands.py``. Returns the request row dict
    (id, request_type, title, detail_json, status, created_at) when
    exactly one pending request matches the prefix; None on no match
    or ambiguous prefix.
    """
    if not id_prefix:
        return None
    try:
        from app.control_plane.governance import get_governance
        pending = get_governance().get_pending() or []
    except Exception:
        logger.debug("governance_signal_bridge: get_pending failed", exc_info=True)
        return None
    matches = [r for r in pending if str(r.get("id", "")).startswith(id_prefix)]
    if len(matches) == 1:
        return matches[0]
    return None
