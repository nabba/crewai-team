"""Thumbs / comment recording — high-level API over the event log.

Phase 4 ships record + read. Phase 5 adds ``summary()`` for Reflexion-style
prompt seeding (negative thumbs become "avoid these directions" hints in
the next cycle's prompt).
"""

from __future__ import annotations

import logging
from enum import Enum

from app.companion import events as _events

logger = logging.getLogger(__name__)


class Polarity(str, Enum):
    UP = "up"
    DOWN = "down"


class Source(str, Enum):
    SIGNAL = "signal"
    REACT = "react"


def record(idea_id: str, workspace_id: str, *,
           polarity: Polarity, comment: str = "",
           source: Source = Source.REACT) -> str:
    """Append a FEEDBACK event. Returns the event_id."""
    ev = _events.Event(
        workspace_id=workspace_id,
        idea_id=idea_id,
        type=_events.EventType.FEEDBACK,
        payload={
            "polarity": polarity.value if isinstance(polarity, Polarity)
            else str(polarity),
            "comment": (comment or "")[:2000],
            "source": source.value if isinstance(source, Source)
            else str(source),
        },
    )
    return _events.append(ev)


def for_idea(workspace_id: str, idea_id: str) -> list[_events.Event]:
    """All FEEDBACK events for one idea, oldest first."""
    return [e for e in _events.read_for_idea(workspace_id, idea_id)
            if e.type == _events.EventType.FEEDBACK]


def summary(workspace_id: str) -> dict:
    """Aggregate counts across all feedback for a workspace.

    Output keys:
      up, down, with_comment              — int counts
      recent_negative_idea_ids            — last 5 thumbs-down idea_ids
      recent_positive_idea_ids            — last 5 thumbs-up idea_ids

    Both recent_* lists are sorted newest-first. Phase 5's Reflexion module
    consumes these to seed the next cycle's prompt with past lessons.
    """
    up = down = with_comment = 0
    neg: list[tuple[float, str]] = []
    pos: list[tuple[float, str]] = []
    for ev in _events.read_all(workspace_id):
        if ev.type != _events.EventType.FEEDBACK:
            continue
        pol = (ev.payload or {}).get("polarity")
        if pol == Polarity.UP.value:
            up += 1
            pos.append((ev.ts, ev.idea_id))
        elif pol == Polarity.DOWN.value:
            down += 1
            neg.append((ev.ts, ev.idea_id))
        if (ev.payload or {}).get("comment"):
            with_comment += 1
    neg.sort(reverse=True)
    pos.sort(reverse=True)
    return {
        "up": up,
        "down": down,
        "with_comment": with_comment,
        "recent_negative_idea_ids": [iid for _, iid in neg[:5]],
        "recent_positive_idea_ids": [iid for _, iid in pos[:5]],
    }
