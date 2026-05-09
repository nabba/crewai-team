"""
safe_io.py — Atomic file I/O utilities.

Provides crash-safe write operations using tempfile + os.replace.
All functions create parent directories automatically.

Use instead of bare Path.write_text() or open(..., 'w') for any
file that must survive process crashes without corruption.

## Disk-quota guard (Wave 1, 2026-05-09)

Every write goes through ``_check_free_space()``. When free space on
the target volume drops below ``DISK_FREE_THRESHOLD_MB`` (default
200 MB), the write is refused with ``DiskQuotaError`` rather than
risking a half-written file on a near-full disk.

Behaviour is conservative: the check fails OPEN if ``shutil.disk_usage``
itself errors out — we don't want a buggy guard to halt the system.
Set ``DISK_FREE_THRESHOLD_MB=0`` to disable the check entirely.

Every refusal is best-effort audited as
``actor='safe_io', action='disk_quota_block'`` so operators can see
on the dashboard when writes started getting blocked.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Disk-quota guard ──────────────────────────────────────────────────────


class DiskQuotaError(OSError):
    """Raised when a write is refused because free disk space is below
    ``DISK_FREE_THRESHOLD_MB``. Subclass of OSError so existing
    error-handling code that catches OSError still routes correctly.
    """


def _free_threshold_mb() -> int:
    """Read the threshold env var with a safe default. ``0`` disables."""
    raw = os.getenv("DISK_FREE_THRESHOLD_MB", "200").strip()
    try:
        v = int(raw)
    except ValueError:
        v = 200
    return max(0, v)


def _check_free_space(path: Path) -> None:
    """Refuse writes when free space < threshold. Fail-open on errors."""
    threshold = _free_threshold_mb()
    if threshold == 0:  # explicitly disabled
        return
    # Walk up to the first existing ancestor — the path itself may not
    # exist yet (we're about to create it). ``shutil.disk_usage`` needs
    # an existing directory.
    probe = path if path.exists() else path.parent
    while probe and not probe.exists():
        probe = probe.parent
    if probe is None or not probe.exists():
        return  # fail-open — can't probe an invalid path
    try:
        usage = shutil.disk_usage(str(probe))
    except Exception:
        # shutil bug / unsupported FS / permission denied → fail open.
        # The write will likely fail naturally if there's a real
        # problem; we don't want our defensive guard to halt the system.
        logger.debug("safe_io: disk_usage probe failed for %s", probe, exc_info=True)
        return
    free_mb = usage.free / (1024 * 1024)
    if free_mb < threshold:
        # Best-effort audit. The audit subsystem must NEVER itself
        # call back into safe_io (would recurse) — wrap in try/except
        # so a misconfigured audit doesn't break the guard.
        try:
            from app.control_plane.audit import get_audit
            get_audit().log(
                actor="safe_io",
                action="disk_quota_block",
                detail={
                    "path": str(path),
                    "free_mb": round(free_mb, 2),
                    "threshold_mb": threshold,
                },
            )
        except Exception:
            logger.debug("safe_io: audit log failed", exc_info=True)
        raise DiskQuotaError(
            f"refusing write to {path}: {free_mb:.0f} MB free "
            f"< {threshold} MB threshold (DISK_FREE_THRESHOLD_MB)"
        )


# ── Public API ────────────────────────────────────────────────────────────


def safe_write(path: Path | str, data: str | bytes) -> None:
    """Atomic write using tempfile + os.replace.

    Creates parent directories. Data is fully written to a temp file
    before atomically replacing the target — no partial writes.

    Refuses writes when free disk space < ``DISK_FREE_THRESHOLD_MB``
    (raises ``DiskQuotaError``). Set the env var to ``0`` to disable.
    """
    path = Path(path)
    _check_free_space(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = data.encode("utf-8") if isinstance(data, str) else data
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        os.write(fd, encoded)
        os.close(fd)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass
        raise


def safe_write_json(path: Path | str, obj: Any, indent: int = 2) -> None:
    """Atomic JSON write. Serializes with default=str for datetime support."""
    safe_write(path, json.dumps(obj, indent=indent, default=str))


def safe_append(path: Path | str, line: str) -> None:
    """Append a line + fsync for crash safety on JSONL/log files.

    Not atomic (appends can interleave under concurrent access), but
    fsync ensures the line reaches disk before returning.

    Refuses appends when free disk space < ``DISK_FREE_THRESHOLD_MB``.
    """
    path = Path(path)
    _check_free_space(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")
        f.flush()
        os.fsync(f.fileno())
