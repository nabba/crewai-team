"""
safe_io.py — Atomic file I/O utilities.

Provides crash-safe write operations using tempfile + os.replace.
All functions create parent directories automatically.

Use instead of bare Path.write_text() or open(..., 'w') for any
file that must survive process crashes without corruption.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def safe_write(path: Path | str, data: str | bytes) -> None:
    """Atomic write using tempfile + os.replace.

    Creates parent directories. Data is fully written to a temp file
    before atomically replacing the target — no partial writes.
    """
    path = Path(path)
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
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")
        f.flush()
        os.fsync(f.fileno())
