"""Per-listener heartbeat helper for the firebase poller threads.

Wave 0/1 closure (#A3, 2026-05-09). The original
``listener_heartbeat`` monitor watched workspace-wide activity mtime
as a coarse proxy. That misses the case where one specific listener
silently dies while other activity continues — e.g., the mode_listener
hangs but conversations.db keeps getting touched by inbound messages.

This helper provides ``touch(name)`` so each poller can record a
per-listener heartbeat at every successful loop iteration. The
``listener_heartbeat`` monitor walks the directory and alerts on
any listener whose individual mtime is stale — even when other
listeners are healthy.

The heartbeat dir is ``workspace/heartbeats/`` and lives outside the
healing/ subsystem state to avoid namespace conflicts. Names follow
the threading thread name convention (``firebase-mode-poll``,
``firebase-kb-poll``, etc.) for traceability with
``threading.enumerate()``.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


_HEARTBEAT_DIR = Path("/app/workspace/heartbeats")

# Known listeners — used by the monitor to know what to expect. Names
# must match the threading.Thread `name=` argument so `threading.enumerate()`
# can be cross-referenced.
KNOWN_LISTENERS: tuple[str, ...] = (
    "firebase-mode-poll",
    "firebase-kb-poll",
    "firebase-phil-poll",
    "firebase-fiction-poll",
    "firebase-episteme-poll",
    "firebase-experiential-poll",
    "firebase-aesthetics-poll",
    "firebase-tensions-poll",
    "firebase-chat-poll",
)


def _heartbeat_path(name: str) -> Path:
    """Return the heartbeat file path for ``name``. Caller validates name."""
    # Replace path separators just in case — the names should already
    # be filesystem-safe, but defense in depth.
    safe = name.replace("/", "_").replace("\\", "_")
    return _HEARTBEAT_DIR / f"{safe}.heartbeat"


def touch(name: str) -> None:
    """Update mtime on the per-listener heartbeat file. Best-effort.

    Safe to call inside a poll loop — never raises, never blocks on
    anything beyond a stat + utime call.
    """
    try:
        _HEARTBEAT_DIR.mkdir(parents=True, exist_ok=True)
        path = _heartbeat_path(name)
        # touch(exist_ok=True) bumps mtime even if the file exists.
        path.touch(exist_ok=True)
    except Exception:
        logger.debug(
            "listener_heartbeats: touch failed for %s", name, exc_info=True,
        )


def list_heartbeats() -> list[dict]:
    """Return ``[{name, mtime, age_s}]`` for every heartbeat file.

    Used by the monitor. Returns empty list if the dir doesn't exist.
    """
    import time as _time
    if not _HEARTBEAT_DIR.exists():
        return []
    out = []
    now = _time.time()
    try:
        for f in _HEARTBEAT_DIR.glob("*.heartbeat"):
            try:
                stat = f.stat()
            except OSError:
                continue
            name = f.stem
            out.append({
                "name": name,
                "mtime": stat.st_mtime,
                "age_s": now - stat.st_mtime,
            })
    except OSError:
        return []
    return out
