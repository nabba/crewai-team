"""lock_metrics — passive contention timing for shared workspace files.

PROGRAM §49 — Q14.6 (year-2+ resilience §10.6). Closes the gap that
:mod:`app.healing.monitors.lock_housekeeper` doesn't cover:
``lock_housekeeper`` sweeps STALE locks (dead processes that crashed
holding a lock); it doesn't see LIVE contention (two healthy
processes racing on the same JSONL file).

Design constraint (operator decision): **zero behavior change** —
this module never blocks, never adds new locks, never alters the
write path. It just observes elapsed-time per write and records
outliers. The hypothesis is that long write times correlate with
underlying OS-level queueing from concurrent writers; whether
that's true is exactly what the operator wants to find out.

API:

  * ``record_write_timing(resource, elapsed_ms)`` — append a row to
    the contention JSONL if ``elapsed_ms >= _RECORD_THRESHOLD_MS``.
    Capped at ``_MAX_LINES`` via the existing ``append_with_cap``.
  * ``time_write(resource)`` — context manager. Times the inner
    block, calls ``record_write_timing`` on exit. Used by
    :mod:`app.safe_io` to wrap the central primitives.

The monitor :mod:`app.healing.monitors.lock_contention` reads the
JSONL weekly, computes p99 latency per resource, alerts when p99
exceeds threshold.

Failure-isolated. Any failure inside ``record_write_timing`` is
swallowed silently — instrumentation must never break the production
write path.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


# Slow-write threshold — only record writes that took longer than
# this. Below 50ms is normal kernel-level I/O; above suggests
# queueing.
_RECORD_THRESHOLD_MS = 50

# Bound the contention JSONL so it doesn't grow unbounded over years.
_MAX_LINES = 10_000

_LOG_FILE_NAME = "lock_waits.jsonl"


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _log_path() -> Path:
    return _workspace() / "healing" / _LOG_FILE_NAME


def _resource_label(path: str) -> str:
    """Normalise a full filesystem path to a stable resource key.

    e.g. ``/app/workspace/companion/interest_profile.json`` →
    ``companion/interest_profile.json``. Lets the monitor aggregate
    p99 per-resource regardless of host workspace location."""
    p = Path(path).resolve()
    workspace = _workspace().resolve()
    try:
        rel = p.relative_to(workspace)
        return str(rel)
    except ValueError:
        # Path not under workspace — return a generic bucket.
        return f"other/{p.name}"


def record_write_timing(resource: str, elapsed_ms: float) -> None:
    """Append a row to the contention JSONL if elapsed exceeds the
    threshold. Failure-isolated.

    ``resource`` may be a full path; it's normalised to the workspace-
    relative form so the monitor's per-resource aggregation is stable
    across deploy environments."""
    if elapsed_ms < _RECORD_THRESHOLD_MS:
        return
    try:
        from app.utils.jsonl_retention import append_with_cap
        line = json.dumps({
            "ts": time.time(),
            "resource": _resource_label(resource),
            "elapsed_ms": round(elapsed_ms, 2),
        }, sort_keys=True)
        append_with_cap(_log_path(), line, _MAX_LINES)
    except Exception:
        # Never propagate — instrumentation must not break callers.
        logger.debug(
            "lock_metrics: record_write_timing failed", exc_info=True,
        )


@contextmanager
def time_write(resource: str) -> Iterator[None]:
    """Context manager: time the block, record on exit if slow.

    Use::

        with time_write(str(path)):
            # the actual write operation
            ...
    """
    start = time.monotonic()
    try:
        yield
    finally:
        try:
            elapsed_ms = (time.monotonic() - start) * 1000
            record_write_timing(resource, elapsed_ms)
        except Exception:
            pass  # never propagate
