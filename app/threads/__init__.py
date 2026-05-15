"""Long-horizon thread primitive — cross-day, cross-crew question persistence.

Where :mod:`app.control_plane.crew_tasks` tracks one crew run from
start to completion, a :class:`Thread` tracks a *line of inquiry* that
spans many crew runs across many days. Use cases:

  - "I am working on Q for 3 days; here are the sub-questions and
    what's resolved + what's blocked."
  - The recovery loop, when stuck, can read open threads to find
    relevant prior context (``thread.notes``, ``thread.blockers``).
  - The companion can surface stale threads (no progress in N days)
    as a nudge.

State machine::

    OPEN ──→ IN_PROGRESS ──┬──→ RESOLVED
        │            │      │
        │            ├──→ BLOCKED ──→ IN_PROGRESS
        │            │
        ↓            ↓
     ABANDONED   ABANDONED

Storage (per-record JSON + lazy in-memory index, mirrors the
change_requests + architecture_requests pattern):

    workspace/threads/<id>.json       — full Thread serialised
    workspace/threads/.lock            — process-level lock (future)

This is the *primitive*. REST API + React surface + recovery-loop
integration land in subsequent commits — they all consume the
public surface here.
"""

from app.threads.lifecycle import (
    abandon_thread,
    add_blocker,
    add_subquestion,
    add_unblock_hint,
    clear_blockers,
    clear_unblock_hints,
    create_thread,
    link_crew_task,
    link_inquiry,
    mark_blocked,
    mark_in_progress,
    record_note,
    resolve_subquestion,
    resolve_thread,
)
from app.threads.models import (
    InvalidThreadTransition,
    SubQuestion,
    Thread,
    ThreadStatus,
)
from app.threads.store import get, list_all, list_open, reset_for_tests

__all__ = [
    "InvalidThreadTransition",
    "SubQuestion",
    "Thread",
    "ThreadStatus",
    "abandon_thread",
    "add_blocker",
    "add_subquestion",
    "add_unblock_hint",
    "clear_blockers",
    "clear_unblock_hints",
    "create_thread",
    "get",
    "link_crew_task",
    "link_inquiry",
    "list_all",
    "list_open",
    "mark_blocked",
    "mark_in_progress",
    "record_note",
    "reset_for_tests",
    "resolve_subquestion",
    "resolve_thread",
]
