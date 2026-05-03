"""Tool quarantine list — runtime filter for known-bad tools.

Mirrors the pattern in ``workspace/skills/_quarantine/`` but uses a
single JSON file instead of a directory move, because tools live in
code (not filesystem skills) and can't simply be relocated.

Schema (``workspace/tool_registry/quarantine.json``)::

    {
      "quarantined": [
        {"name": "broken_tool", "reason": "...", "since": "2026-..."},
        ...
      ]
    }

When a tool is quarantined:
  * ``ToolRegistry.is_quarantined(name)`` returns True.
  * ``search_tools(...)`` filters it out entirely (not just down-ranks).
  * The ``/api/cp/tools/{name}`` endpoint shows the quarantine reason.

Adding to / removing from the quarantine list is a deliberate human
act: edit the JSON file, gateway picks it up on the next read. There
is no programmatic API for agents to quarantine each other's tools —
quarantine is operator-only, same governance as ``app/souls/``.

Why JSON, not Postgres
----------------------
The list is small (typically 0–5 entries), edited by hand, version-
controlled in git, and needs to survive without a database. Postgres
copy is via the snapshot table for read-only visibility, but the
write path is the JSON file.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


_QUARANTINE_PATH = Path("/app/workspace/tool_registry/quarantine.json")


@dataclass(frozen=True)
class QuarantineEntry:
    name: str
    reason: str
    since: str  # ISO date

    def to_dict(self) -> dict:
        return {"name": self.name, "reason": self.reason, "since": self.since}


class _QuarantineCache:
    """Re-reads the JSON file on each call — small file, lookup is rare,
    no need for fancy invalidation. Operator edits are picked up on the
    next call."""

    _lock = threading.Lock()

    @classmethod
    def load(cls) -> dict[str, QuarantineEntry]:
        with cls._lock:
            if not _QUARANTINE_PATH.exists():
                return {}
            try:
                data = json.loads(_QUARANTINE_PATH.read_text())
            except Exception as exc:
                logger.warning(
                    "tool_registry quarantine: malformed JSON at %s: %s — "
                    "treating as empty (operator should fix the file)",
                    _QUARANTINE_PATH, exc,
                )
                return {}
            entries: dict[str, QuarantineEntry] = {}
            for row in data.get("quarantined", []):
                try:
                    entries[row["name"]] = QuarantineEntry(
                        name=row["name"],
                        reason=row.get("reason", "(no reason given)"),
                        since=row.get("since", "unknown"),
                    )
                except KeyError as exc:
                    logger.warning(
                        "tool_registry quarantine: row missing field %s: %s",
                        exc, row,
                    )
            return entries


def is_quarantined(name: str) -> bool:
    """True iff ``name`` appears in the quarantine list."""
    return name in _QuarantineCache.load()


def quarantine_entry(name: str) -> QuarantineEntry | None:
    """Return the entry (with reason + since) or None."""
    return _QuarantineCache.load().get(name)


def all_quarantined() -> list[QuarantineEntry]:
    """List all quarantined tool entries."""
    return list(_QuarantineCache.load().values())


def quarantined_names() -> set[str]:
    """Set of names — used as a fast membership test in discovery."""
    return set(_QuarantineCache.load().keys())
