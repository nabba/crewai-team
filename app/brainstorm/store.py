"""Session storage — JSON files under ``workspace/brainstorm/``.

Layout::

    workspace/brainstorm/
      sessions/<session_id>.json    # full session state
      active/<safe_sender>.txt      # pointer: current active session_id

Atomic writes via temp-file + rename, mirroring ``app.companion.state``.
The base directory honours ``BRAINSTORM_DIR`` env var so tests can isolate.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from threading import Lock
from typing import Iterable

from app.brainstorm.session import BrainstormSession

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path("workspace/brainstorm")
_LOCK = Lock()


def _base_dir() -> Path:
    return Path(os.environ.get("BRAINSTORM_DIR", str(_DEFAULT_DIR)))


def _sessions_dir() -> Path:
    return _base_dir() / "sessions"


def _active_dir() -> Path:
    return _base_dir() / "active"


def _safe_sender(sender: str) -> str:
    """Sanitize a sender ID for use as a filename."""
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in sender)
    return cleaned or "anon"


def _session_path(session_id: str) -> Path:
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_") or "default"
    return _sessions_dir() / f"{safe}.json"


def _active_path(sender: str) -> Path:
    return _active_dir() / f"{_safe_sender(sender)}.txt"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            delete=False,
            prefix=".tmp.",
            suffix=path.suffix or ".tmp",
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        os.replace(tmp_path, path)


# ── Session CRUD ──────────────────────────────────────────────────────────


def save(session: BrainstormSession) -> None:
    """Persist a session to disk (atomic)."""
    payload = json.dumps(session.to_dict(), indent=2, sort_keys=True)
    _atomic_write(_session_path(session.session_id), payload)


def load(session_id: str) -> BrainstormSession | None:
    """Load a session by id; return None if missing or unreadable."""
    p = _session_path(session_id)
    if not p.exists():
        return None
    try:
        return BrainstormSession.from_dict(json.loads(p.read_text()))
    except Exception as exc:
        logger.warning("brainstorm.store: load failed for %s: %s", session_id, exc)
        return None


def delete(session_id: str) -> bool:
    """Remove a session file and clear any active pointer that references it."""
    p = _session_path(session_id)
    if not p.exists():
        return False
    try:
        p.unlink()
    except OSError:
        return False
    # Sweep active pointers that referenced this session.
    adir = _active_dir()
    if adir.exists():
        for ap in adir.glob("*.txt"):
            try:
                if ap.read_text().strip() == session_id:
                    ap.unlink()
            except OSError:
                pass
    return True


def list_sessions(sender: str | None = None) -> list[BrainstormSession]:
    """Return all sessions, newest first. Optionally filtered by sender."""
    sdir = _sessions_dir()
    if not sdir.exists():
        return []
    out: list[BrainstormSession] = []
    for p in sdir.glob("*.json"):
        try:
            data = json.loads(p.read_text())
            sess = BrainstormSession.from_dict(data)
        except Exception:
            continue
        if sender is not None and sess.sender != sender:
            continue
        out.append(sess)
    out.sort(key=lambda s: s.updated_at, reverse=True)
    return out


# ── Active-session pointer ────────────────────────────────────────────────


def set_active(sender: str, session_id: str) -> None:
    """Mark ``session_id`` as the active session for ``sender``."""
    _atomic_write(_active_path(sender), session_id.strip())


def get_active(sender: str) -> BrainstormSession | None:
    """Return the active session for ``sender``, or None."""
    p = _active_path(sender)
    if not p.exists():
        return None
    try:
        sid = p.read_text().strip()
    except OSError:
        return None
    if not sid:
        return None
    sess = load(sid)
    if sess is None or sess.status not in ("active", "paused"):
        return None
    return sess


def clear_active(sender: str) -> None:
    """Drop the active-session pointer for ``sender`` (idempotent)."""
    p = _active_path(sender)
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass


def iter_paused(sender: str) -> Iterable[BrainstormSession]:
    """Yield paused sessions belonging to ``sender``, newest first."""
    for s in list_sessions(sender=sender):
        if s.status == "paused":
            yield s
