"""Append-only ledger of identity-shaping events (§2.8).

Closes the gap from the original analysis: the Tier-3 amendment
protocol gates *intentional* edits, but it doesn't surface
*aggregate* drift across many small approved amendments. The
narrative-self FIFO holds 5 identity claims; at year 2 the older
you is gone. This ledger is the multi-year record.

Seven event kinds, all append-only::

    tier3_amendment        Tier-3 IMMUTABLE file edit landed
    governance_ratchet     SAFETY/QUALITY threshold raised or relaxed
    soul_edit              constitution.md or souls/* edited
    integrity_regen        SubIA integrity manifest regenerated
    scorecard_change       Butlin/RSM/SK indicator status changed
    self_quarantine_change file added/removed from quarantine list
    substrate_migration    embedding-model migration cutover applied
                           (PROGRAM §40 Item 12) — rewrites the meaning
                           of every embedding the system holds

Storage: ``workspace/identity/continuity_ledger.jsonl``. One
``IdentityEvent`` per line. Append-only — never delete, never
overwrite. The append API is robust against concurrent writers
(O_APPEND + line-at-a-time JSON) at the cost of duplicate-detection
being the consumer's job.

Read API:

  ``list_events(*, since_iso=None, kinds=None)``  — chronological
  ``summarise_drift(window_days=365)`` — aggregate counts per kind
                                          for the last N days

Master switch ``IDENTITY_LEDGER_ENABLED`` (default ``true``). When
disabled, ``record_event`` silently no-ops; readers return [].
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_DEFAULT_PATH = Path("/app/workspace/identity/continuity_ledger.jsonl")
_path_override: Path | None = None


IDENTITY_EVENT_KINDS: frozenset[str] = frozenset({
    "tier3_amendment",
    "governance_ratchet",
    "soul_edit",
    "integrity_regen",
    "scorecard_change",
    "self_quarantine_change",
    "substrate_migration",  # PROGRAM §40 Item 12 — Q3.1
    "person_correlation_policy",  # PROGRAM §42 — Q4.2 (Q4.2.2#1)
    "sentience_observation",  # PROGRAM §43 — Q5.4.2
    "resilience_drill",  # PROGRAM §44 — Q6.1
})


def _enabled() -> bool:
    return os.getenv("IDENTITY_LEDGER_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _resolve_path() -> Path:
    return _path_override if _path_override else _DEFAULT_PATH


@dataclass(frozen=True)
class IdentityEvent:
    """One identity-shaping event."""

    ts: str
    kind: str
    actor: str
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "kind": self.kind,
            "actor": self.actor,
            "summary": self.summary,
            "detail": dict(self.detail),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IdentityEvent":
        return cls(
            ts=data["ts"],
            kind=data["kind"],
            actor=data.get("actor", ""),
            summary=data.get("summary", ""),
            detail=dict(data.get("detail", {})),
        )


def record_event(
    *,
    kind: str,
    actor: str,
    summary: str,
    detail: dict[str, Any] | None = None,
    path: Path | str | None = None,
    now: datetime | None = None,
) -> bool:
    """Append an identity event to the ledger. Returns True on success.

    Failure-isolated — append errors are logged at debug and the call
    returns False. The recorder must NEVER block whatever subsystem
    is informing it (the consciousness boundary is observational).
    """
    if not _enabled():
        return False
    if kind not in IDENTITY_EVENT_KINDS:
        logger.debug("identity_ledger: unknown kind %r — skipping", kind)
        return False
    if not summary.strip():
        return False

    event = IdentityEvent(
        ts=(now or datetime.now(timezone.utc)).isoformat(),
        kind=kind,
        actor=(actor or "unknown").strip(),
        summary=summary.strip(),
        detail=dict(detail or {}),
    )
    target = Path(path) if path else _resolve_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        return True
    except OSError as exc:
        logger.debug("identity_ledger: append failed: %s", exc)
        return False


def list_events(
    *,
    since_iso: str | None = None,
    kinds: set[str] | None = None,
    path: Path | str | None = None,
) -> list[IdentityEvent]:
    """Read every event, optionally filtered. Chronological order."""
    target = Path(path) if path else _resolve_path()
    if not target.exists():
        return []
    out: list[IdentityEvent] = []
    try:
        with open(target, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    event = IdentityEvent.from_dict(raw)
                except (KeyError, TypeError):
                    continue
                if kinds and event.kind not in kinds:
                    continue
                if since_iso and event.ts < since_iso:
                    continue
                out.append(event)
    except OSError:
        return []
    out.sort(key=lambda e: e.ts)
    return out


@dataclass(frozen=True)
class DriftSummary:
    """Aggregate counts over the lookback window."""

    window_days: int
    n_events: int
    by_kind: dict[str, int]
    by_actor: dict[str, int]
    first_seen: str | None
    last_seen: str | None


def summarise_drift(
    window_days: int = 365,
    *,
    path: Path | str | None = None,
    now: datetime | None = None,
) -> DriftSummary:
    """Read all events in the last ``window_days``, aggregate by kind +
    actor. Useful as the input to the annual reflection essay."""
    cur = now or datetime.now(timezone.utc)
    cutoff = (cur - timedelta(days=window_days)).isoformat()
    events = list_events(since_iso=cutoff, path=path)

    by_kind = Counter(e.kind for e in events)
    by_actor = Counter(e.actor for e in events)
    first = events[0].ts if events else None
    last = events[-1].ts if events else None
    return DriftSummary(
        window_days=window_days,
        n_events=len(events),
        by_kind=dict(by_kind),
        by_actor=dict(by_actor),
        first_seen=first,
        last_seen=last,
    )


def _reset_for_tests(path: Path | None = None) -> None:
    global _path_override
    _path_override = path
