"""Append-only event log per workspace.

Records state transitions and feedback events for Companion ideas. Each
event is one JSON line at ``workspace/companion/events/<workspace_id>.jsonl``.

Event sourcing: the ``ideas.jsonl`` file holds the immutable creation
record (state at creation time); this event log holds everything that
happens to an idea afterwards. Current state is derived by reading both
and folding events forward — see ``app.companion.idea_store.current_state``.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

_EVENTS_DIR = Path(os.environ.get(
    "COMPANION_EVENTS_DIR", "workspace/companion/events"))
_LOCK = Lock()


class EventType(str, Enum):
    SURFACED = "surfaced"
    FEEDBACK = "feedback"
    ARCHIVED = "archived"
    APPROVED = "approved"
    DOCUMENTED = "documented"  # Phase 8 — idea promoted to a document
    WIKI_REGISTERED = "wiki_registered"  # Phase 9 — also in the workspace wiki
    # Phase 11 — grand-task synthesis (12 h cadence)
    GRAND_TASK_PROPOSED = "grand_task_proposed"
    GRAND_TASK_ACCEPTED = "grand_task_accepted"
    GRAND_TASK_REJECTED = "grand_task_rejected"
    # Phase 13 — cross-workspace transfer
    CROSS_WORKSPACE_INBOX = "cross_workspace_inbox"
    CROSS_WORKSPACE_ACCEPTED = "cross_workspace_accepted"
    CROSS_WORKSPACE_DISMISSED = "cross_workspace_dismissed"
    # Phase 11.5 — cold-start seed bootstrap from CP mission + tickets
    SEED_DERIVED = "seed_derived"


@dataclass
class Event:
    event_id: str = field(
        default_factory=lambda: f"ev_{uuid.uuid4().hex[:12]}")
    workspace_id: str = ""
    idea_id: str = ""
    type: EventType = EventType.FEEDBACK
    ts: float = field(default_factory=time.time)
    payload: dict = field(default_factory=dict)


def append(event: Event) -> str:
    """Append the event to the workspace's log. Returns the event_id."""
    p = _path_for(event.workspace_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(event), default=_json_default, sort_keys=True)
    with _LOCK:
        with open(p, "a") as f:
            f.write(line + "\n")
    return event.event_id


def read_all(workspace_id: str) -> list[Event]:
    """Read every event for a workspace, oldest first."""
    p = _path_for(workspace_id)
    if not p.exists():
        return []
    out: list[Event] = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            kwargs = {k: raw[k] for k in Event.__dataclass_fields__
                      if k in raw}
            if "type" in kwargs:
                kwargs["type"] = EventType(kwargs["type"])
            out.append(Event(**kwargs))
        except Exception:
            continue
    return out


def read_for_idea(workspace_id: str, idea_id: str) -> list[Event]:
    """All events for one idea, oldest first."""
    return [e for e in read_all(workspace_id) if e.idea_id == idea_id]


def iter_all_workspaces(
    *, type_filter: EventType | None = None,
    since_ts: float | None = None,
) -> list[Event]:
    """Walk every workspace's event log; yield events newest-first.

    Phase F #1 (2026-05-09): consumers like ``interest_model`` and
    ``lessons_learned`` need cross-workspace feedback signals; the
    earlier code looked at a single ``events.jsonl`` that doesn't
    exist (events are sharded per workspace under
    ``events/<ws_id>.jsonl``). This helper centralises the walk so
    the wrong-path bug can't recur.

    Args:
        type_filter: When set, drop events whose ``type`` doesn't match.
        since_ts:    When set, drop events older than this epoch second.
    """
    if not _EVENTS_DIR.exists():
        return []
    out: list[Event] = []
    for p in _EVENTS_DIR.glob("*.jsonl"):
        try:
            text = p.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                kwargs = {k: raw[k] for k in Event.__dataclass_fields__
                          if k in raw}
                if "type" in kwargs:
                    kwargs["type"] = EventType(kwargs["type"])
                ev = Event(**kwargs)
            except Exception:
                continue
            if type_filter is not None and ev.type != type_filter:
                continue
            if since_ts is not None and ev.ts < since_ts:
                continue
            out.append(ev)
    out.sort(key=lambda e: e.ts, reverse=True)
    return out


def _path_for(workspace_id: str) -> Path:
    safe = "".join(c for c in workspace_id if c.isalnum() or c in "-_") \
        or "default"
    return _EVENTS_DIR / f"{safe}.jsonl"


def _json_default(o):
    if isinstance(o, EventType):
        return o.value
    raise TypeError(f"Not JSON-serialisable: {type(o)}")
