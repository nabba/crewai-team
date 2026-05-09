"""Lock-file housekeeper — periodic cleanup of orphaned ``.lock`` files.

The codebase uses ``fcntl.flock`` for cross-process coordination
(``app/workspace_versioning.py``, ``app/tools/wiki_tools.py``,
``app/memory/wiki_index_reconciler.py``, ...). The kernel auto-releases
the *lock* when the holding process dies — but the *lock file* stays
on disk. Over years of operation these accumulate, both as disk
clutter and as a slow signal that "many short-lived processes are
churning here."

This monitor walks known lock-file directories every 6 h and deletes
files that:

  1. Are older than ``_MIN_AGE_S`` (default 1 h) — defends against
     racing with a fresh acquirer.
  2. Are NOT currently fcntl-held — proven by attempting
     ``fcntl.LOCK_EX | fcntl.LOCK_NB`` and seeing it succeed. If
     someone holds the lock the call raises ``BlockingIOError`` and
     we skip the file.

Both conditions are required: the AND defends against PID-reuse
hazards (a fresh process landing on a recycled PID would otherwise
look "alive"), and the fcntl probe is the actual source of truth on
whether the lock is held.

A pile-up alert fires when more than ``_PILE_UP_THRESHOLD`` lock
files exist across the watched dirs — that's a strong signal of a
leak (a code path forgetting to release).
"""
from __future__ import annotations

import errno
import fcntl
import logging
import os
import time
from pathlib import Path
from typing import Iterable

from app.life_companion._common import (
    audit_event,
    background_enabled,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)

_STATE_FILE = "lock_housekeeper.json"

# Directories where lock files legitimately live. Add to this list as
# new lock-using subsystems ship. Order doesn't matter — we walk all.
_WATCHED_DIRS: tuple[Path, ...] = (
    Path("/app/workspace/locks"),
    Path("/app/workspace/dreams"),
    Path("/app/workspace"),  # for .workspace.lock from workspace_versioning
)

# Minimum age before we'll consider a lock file orphaned. Defends
# against deleting a file someone is in the middle of acquiring.
_MIN_AGE_S = 60 * 60  # 1 hour

# Alert threshold — pile-up suggests a code path leaking locks.
_PILE_UP_THRESHOLD = 50

# Cooldown between pile-up alerts so we don't spam during a sustained
# leak — operator gets one Signal per day.
_ALERT_COOLDOWN_S = 24 * 3600


def _candidate_files() -> Iterable[Path]:
    """Yield ``.lock`` files in the watched directories. Non-recursive
    by default — every dir on this list is shallow.
    """
    for d in _WATCHED_DIRS:
        if not d.exists() or not d.is_dir():
            continue
        try:
            for p in d.iterdir():
                if not p.is_file():
                    continue
                if not (p.name.endswith(".lock") or p.name.startswith(".")
                        and p.name.endswith(".lock")):
                    continue
                yield p
        except Exception:
            logger.debug("lock_housekeeper: cannot list %s", d, exc_info=True)


def _is_uncontested(path: Path) -> bool:
    """Probe whether the lock is currently held. Returns True iff we
    can acquire and release it without anyone blocking.

    Important: closing the FD (via ``with`` context) automatically
    releases the lock — there's no window where we leave a dangling
    lock for another process to inherit.
    """
    try:
        with path.open("rb+") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError) as exc:
                # EWOULDBLOCK / EAGAIN = held by someone alive.
                if getattr(exc, "errno", None) in (errno.EWOULDBLOCK, errno.EAGAIN):
                    return False
                # Other errors → fail-safe: don't delete.
                logger.debug(
                    "lock_housekeeper: flock probe error %s on %s",
                    exc, path,
                )
                return False
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            return True
    except FileNotFoundError:
        return False  # already gone — pretend it was contested
    except OSError as exc:
        # Permission denied / EISDIR / etc. — fail-safe: don't delete.
        logger.debug(
            "lock_housekeeper: open failed for %s: %s", path, exc,
        )
        return False


def run() -> None:
    """One pass — cadence is enforced by the daemon driver, NOT by us.

    This monitor is registered with cadence ``6 h`` so we don't need
    an internal time-since-last-run guard.
    """
    if not background_enabled():
        return

    candidates = list(_candidate_files())
    if not candidates:
        return

    now = time.time()
    deleted_paths: list[str] = []
    skipped_held: int = 0
    skipped_too_young: int = 0

    for path in candidates:
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue
        age = now - mtime
        if age < _MIN_AGE_S:
            skipped_too_young += 1
            continue
        if not _is_uncontested(path):
            skipped_held += 1
            continue
        try:
            path.unlink()
            deleted_paths.append(str(path))
        except FileNotFoundError:
            pass
        except OSError:
            logger.debug(
                "lock_housekeeper: unlink failed for %s", path, exc_info=True,
            )

    audit_event(
        "lock_housekeeper_pass",
        candidates=len(candidates),
        deleted=len(deleted_paths),
        skipped_held=skipped_held,
        skipped_too_young=skipped_too_young,
    )
    if deleted_paths:
        logger.info(
            "lock_housekeeper: removed %d orphan lock files",
            len(deleted_paths),
        )

    # ── Pile-up alert ──────────────────────────────────────────────
    if len(candidates) >= _PILE_UP_THRESHOLD:
        state = read_state_json(_STATE_FILE, {"last_alert_at": 0.0})
        if (now - float(state.get("last_alert_at", 0))) >= _ALERT_COOLDOWN_S:
            state["last_alert_at"] = now
            state["last_pile_up_size"] = len(candidates)
            write_state_json(_STATE_FILE, state)
            send_signal_alert(
                f"🔒 Self-heal: {len(candidates)} lock files piled up across "
                f"watched directories — strong signal of a code path leaking "
                f"fcntl locks.\n\n"
                f"  • deleted this pass: {len(deleted_paths)}\n"
                f"  • currently held: {skipped_held}\n"
                f"  • too young to clean: {skipped_too_young}\n\n"
                f"Watched dirs: "
                + ", ".join(f"`{d}`" for d in _WATCHED_DIRS)
                + "\n\n"
                f"Investigate which subsystem is acquiring without "
                f"releasing.",
                tag="lock_housekeeper",
            )
