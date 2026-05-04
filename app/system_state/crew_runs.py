"""Lightweight in-memory ring buffer of recent crew outcomes.

Used by the system_state service so a "did this crew just succeed?"
question can be answered without going to the journal or grepping
logs. Capped to keep memory bounded; older entries fall off.

Phase 5.1: tap-in is done from ``app/crews/base_crew.py`` at crew
completion. We stay non-fatal everywhere — recording a crew run
must NEVER raise into the request path (the actual crew result is
the user-facing artifact; tracking it is observational).

Why in-memory instead of a DB
-----------------------------
The buffer answers "very recent" questions only — minutes to hours.
The journal already persists every crew run with full context;
this is a fast lookup layer in front of it. If the gateway restarts,
the buffer empties and we fall back to "no recent runs visible" —
which the routing logic correctly interprets as "try the crew."
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

# Per-crew bounded deque — 50 most-recent runs is plenty for the
# "did it just succeed?" question. Anything older than ~1 hour is
# also irrelevant to routing decisions.
_BUFFER_PER_CREW = 50

_LOCK = threading.Lock()
_BUFFER: dict[str, deque["CrewRunRecord"]] = {}


@dataclass(frozen=True)
class CrewRunRecord:
    crew: str
    ok: bool
    ts: str  # ISO-8601 UTC
    error: str | None = None
    duration_s: float | None = None
    task_id: str | None = None

    def to_dict(self) -> dict:
        d = {
            "crew": self.crew,
            "ok": self.ok,
            "ts": self.ts,
        }
        if self.error is not None:
            d["error"] = self.error[:300]  # truncate stack traces
        if self.duration_s is not None:
            d["duration_s"] = round(self.duration_s, 1)
        if self.task_id is not None:
            d["task_id"] = self.task_id
        return d


def record_crew_run(
    crew: str,
    *,
    ok: bool,
    error: str | None = None,
    duration_s: float | None = None,
    task_id: str | None = None,
) -> None:
    """Append one outcome to the per-crew buffer. Non-fatal.

    Called from ``base_crew.run_single_agent_crew`` (and equivalents)
    at task completion. If anything goes wrong here, we silently
    drop — observational telemetry must never break the request path.
    """
    if not crew:
        return
    try:
        record = CrewRunRecord(
            crew=crew,
            ok=ok,
            ts=datetime.now(timezone.utc).isoformat(),
            error=error,
            duration_s=duration_s,
            task_id=task_id,
        )
        with _LOCK:
            buf = _BUFFER.get(crew)
            if buf is None:
                buf = deque(maxlen=_BUFFER_PER_CREW)
                _BUFFER[crew] = buf
            buf.append(record)
    except Exception:
        return  # explicit silent drop


def recent_runs(
    crew: str | None = None,
    *,
    limit: int = 10,
) -> dict[str, list[dict]]:
    """Read recent runs.

    Args:
        crew: If set, return only that crew's runs. Else all crews.
        limit: Max entries per crew (newest first).

    Returns:
        ``{crew_name: [{ts, ok, error?, duration_s?}, ...]}``
        Newest first within each crew. Empty dict when no runs
        recorded since gateway boot.
    """
    out: dict[str, list[dict]] = {}
    with _LOCK:
        crews = [crew] if crew else list(_BUFFER.keys())
        for c in crews:
            buf = _BUFFER.get(c)
            if buf is None:
                continue
            # newest first; limit
            entries = list(buf)
            entries.reverse()
            out[c] = [r.to_dict() for r in entries[:limit]]
    return out


def reset_for_tests() -> None:
    """Clear the buffer. Tests use this to start fresh."""
    with _LOCK:
        _BUFFER.clear()


def stats() -> dict[str, int]:
    """Buffer size per crew — for diagnostics."""
    with _LOCK:
        return {c: len(b) for c, b in _BUFFER.items()}
