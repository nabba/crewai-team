"""
journal.py — Unified chronological activity journal.

Single timeline of all system events: task completions, failures,
evolution outcomes, self-reflections, configuration changes, etc.

Replaces scattered journals (error_journal.json, audit_journal.json)
with a unified, structured, queryable activity log.

Storage: workspace/self_awareness_data/journal/JOURNAL.jsonl

IMMUTABLE — infrastructure-level module.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

JOURNAL_DIR = Path("/app/workspace/self_awareness_data/journal")
JOURNAL_FILE = JOURNAL_DIR / "JOURNAL.jsonl"
MAX_ENTRIES = 1000  # Keep last N entries on disk


class JournalEntryType(str, Enum):
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    EVOLUTION_RESULT = "evolution_result"
    SELF_REFLECTION = "self_reflection"
    CONFIGURATION_CHANGE = "configuration_change"
    ERROR = "error"
    OBSERVATION = "observation"
    DECISION = "decision"
    LEARNING = "learning"
    DEPLOYMENT = "deployment"


@dataclass
class JournalEntry:
    """A single event in the system's activity journal."""
    entry_type: JournalEntryType = JournalEntryType.OBSERVATION
    summary: str = ""
    agents_involved: list[str] = field(default_factory=list)
    duration_seconds: float = None
    outcome: str = ""  # success | failure | partial
    details: dict = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entry_type"] = self.entry_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "JournalEntry":
        d["entry_type"] = JournalEntryType(d.get("entry_type", "observation"))
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


class Journal:
    """Unified chronological activity journal."""

    def __init__(self):
        JOURNAL_DIR.mkdir(parents=True, exist_ok=True)

    def write(self, entry: JournalEntry) -> None:
        """Append an entry to the journal."""
        try:
            with open(JOURNAL_FILE, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception:
            logger.debug("journal: write failed", exc_info=True)

    def read_recent(self, n: int = 20, entry_type: str = "") -> list[JournalEntry]:
        """Read the N most recent entries, optionally filtered by type."""
        if not JOURNAL_FILE.exists():
            return []

        entries = []
        try:
            for line in JOURNAL_FILE.read_text().splitlines():
                if line.strip():
                    try:
                        d = json.loads(line)
                        if entry_type and d.get("entry_type") != entry_type:
                            continue
                        entries.append(JournalEntry.from_dict(d))
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass

        return entries[-n:]

    def search(self, query: str, n: int = 10) -> list[JournalEntry]:
        """Simple text search across journal entries."""
        query_lower = query.lower()
        matches = []

        if not JOURNAL_FILE.exists():
            return []

        for line in JOURNAL_FILE.read_text().splitlines():
            if query_lower in line.lower():
                try:
                    matches.append(JournalEntry.from_dict(json.loads(line)))
                except Exception:
                    pass

        return matches[-n:]

    def count(self) -> dict:
        """Count entries by type."""
        counts = {}
        if not JOURNAL_FILE.exists():
            return counts

        for line in JOURNAL_FILE.read_text().splitlines():
            try:
                d = json.loads(line)
                et = d.get("entry_type", "unknown")
                counts[et] = counts.get(et, 0) + 1
            except Exception:
                pass

        return counts

    def trim(self, keep_latest: int = MAX_ENTRIES) -> int:
        """Trim old entries to keep journal manageable."""
        if not JOURNAL_FILE.exists():
            return 0

        lines = JOURNAL_FILE.read_text().splitlines()
        if len(lines) <= keep_latest:
            return 0

        trimmed = len(lines) - keep_latest
        from app.safe_io import safe_write
        safe_write(JOURNAL_FILE, "\n".join(lines[-keep_latest:]) + "\n")
        return trimmed

    def format_recent(self, n: int = 10) -> str:
        """Human-readable recent journal entries."""
        entries = self.read_recent(n)
        if not entries:
            return "Journal: no entries yet"

        lines = [f"📓 Activity Journal (last {len(entries)} entries)", ""]
        for e in entries:
            icon = {
                "startup": "🚀", "shutdown": "🔴",
                "task_completed": "✅", "task_failed": "❌",
                "evolution_result": "🧬", "self_reflection": "🪞",
                "error": "⚠️", "observation": "👁️",
                "decision": "🎯", "learning": "📚",
                "deployment": "🚢",
            }.get(e.entry_type.value, "📝")

            duration = f" ({e.duration_seconds:.1f}s)" if e.duration_seconds else ""
            agents = f" [{', '.join(e.agents_involved)}]" if e.agents_involved else ""
            lines.append(f"  {icon} {e.timestamp[:19]} {e.entry_type.value}{agents}{duration}")
            lines.append(f"     {e.summary[:120]}")

        return "\n".join(lines)


# ── Module-level singleton ───────────────────────────────────────────────────

_journal: Journal | None = None


def get_journal() -> Journal:
    global _journal
    if _journal is None:
        _journal = Journal()
    return _journal
