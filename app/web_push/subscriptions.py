"""
web_push.subscriptions — JSON-backed device registry.

One JSON file per deployment, lock-protected for concurrent writes.
Schema:

    {
      "<endpoint URL>": {
        "endpoint": "...",
        "keys": {"p256dh": "...", "auth": "..."},
        "user_agent": "Mozilla/5.0 ...",
        "added_at": "2026-05-08T20:00:00Z"
      },
      ...
    }

Endpoint URL is the unique key — the same browser/profile always returns the
same endpoint, so re-subscribing simply overwrites instead of duplicating.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any

from app.paths import WORKSPACE_ROOT

logger = logging.getLogger(__name__)

_STORE_PATH = WORKSPACE_ROOT / "web_push_subscriptions.json"
_lock = threading.Lock()


def _load() -> dict[str, dict[str, Any]]:
    if not _STORE_PATH.exists():
        return {}
    try:
        data = json.loads(_STORE_PATH.read_text())
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.warning(f"web_push: failed to load store: {exc}")
    return {}


def _save(state: dict) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STORE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(_STORE_PATH)


def add_subscription(payload: dict[str, Any]) -> bool:
    """Persist a subscription. Returns True if added or refreshed.

    Required keys in ``payload``: endpoint, keys.p256dh, keys.auth.
    """
    endpoint = (payload or {}).get("endpoint", "")
    keys = (payload or {}).get("keys") or {}
    p256dh, auth = keys.get("p256dh"), keys.get("auth")
    if not (endpoint and p256dh and auth):
        return False

    record = {
        "endpoint": endpoint,
        "keys": {"p256dh": p256dh, "auth": auth},
        "user_agent": (payload or {}).get("user_agent") or (payload or {}).get("userAgent") or "",
        "added_at": datetime.now(timezone.utc).isoformat(),
    }

    with _lock:
        state = _load()
        state[endpoint] = record
        _save(state)
    logger.info(f"web_push: subscription registered ({len(state)} total)")
    return True


def remove_subscription(endpoint: str) -> bool:
    """Remove a subscription by endpoint. Returns True if removed."""
    if not endpoint:
        return False
    with _lock:
        state = _load()
        if endpoint not in state:
            return False
        del state[endpoint]
        _save(state)
    logger.info(f"web_push: subscription removed ({len(state)} remaining)")
    return True


def list_subscriptions() -> list[dict[str, Any]]:
    """Return all stored subscriptions (sorted by added_at)."""
    with _lock:
        state = _load()
    return sorted(state.values(), key=lambda r: r.get("added_at", ""), reverse=True)


def prune_subscription(endpoint: str) -> None:
    """Internal: remove a subscription that returned 410 Gone during a send."""
    remove_subscription(endpoint)
