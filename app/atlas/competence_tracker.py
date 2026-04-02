"""
competence_tracker.py — Tracks what the system knows and doesn't know.

Maintains a real-time competence map across domains:
  - APIs: which APIs we have tested clients for
  - Concepts: which technical concepts we understand
  - Patterns: which reusable code patterns we've verified
  - Tools: which external tools/services we can use

Each entry has a confidence score that decays over time and improves
with successful usage. The tracker enables:
  1. Gap detection: "I don't know how to use the Airtable API"
  2. Learning prioritization: "GraphQL subscriptions is my weakest area"
  3. Task routing: "This task needs Stripe — do I know it?"

IMMUTABLE — infrastructure-level module.
"""

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

COMPETENCE_DIR = Path("/app/workspace/atlas/competence")

# Confidence decay: 1% per day for unused knowledge
DECAY_RATE_PER_DAY = 0.01
CONFIDENCE_FLOOR = 0.20

# Depth levels
DEPTH_NONE = "none"
DEPTH_SHALLOW = "shallow"     # heard of it, untested
DEPTH_INTERMEDIATE = "intermediate"  # some usage, partially tested
DEPTH_EXPERT = "expert"       # extensively used and tested


@dataclass
class CompetenceEntry:
    """A single competence domain entry."""
    domain: str = ""            # "apis", "concepts", "patterns", "tools"
    name: str = ""              # "Stripe API", "OAuth2 flows", "retry_with_backoff"
    confidence: float = 0.0     # 0.0 - 1.0
    depth: str = DEPTH_NONE     # none | shallow | intermediate | expert
    last_verified: str = ""     # ISO timestamp
    last_used: str = ""
    usage_count: int = 0
    usage_success_count: int = 0
    source: str = ""            # how we learned this
    skill_id: str = ""          # link to skill library entry (if any)
    notes: str = ""

    def effective_confidence(self) -> float:
        """Compute current confidence with time decay."""
        base = self.confidence
        if self.last_verified:
            try:
                verified = datetime.fromisoformat(self.last_verified)
                days = (datetime.now(timezone.utc) - verified).days
                decay = days * DECAY_RATE_PER_DAY
                base = max(CONFIDENCE_FLOOR, base - decay)
            except Exception:
                pass
        return base

    def effective_depth(self) -> str:
        """Compute depth from confidence."""
        conf = self.effective_confidence()
        if conf >= 0.80:
            return DEPTH_EXPERT
        elif conf >= 0.50:
            return DEPTH_INTERMEDIATE
        elif conf > 0.0:
            return DEPTH_SHALLOW
        return DEPTH_NONE

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CompetenceEntry":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


class CompetenceTracker:
    """Tracks system competence across all knowledge domains."""

    def __init__(self, competence_dir: Path = COMPETENCE_DIR):
        self._dir = competence_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, CompetenceEntry] = {}  # key: f"{domain}:{name}"
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        """Load competence map from disk."""
        path = self._dir / "competence_map.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for entry_data in data:
                    entry = CompetenceEntry.from_dict(entry_data)
                    key = f"{entry.domain}:{entry.name}".lower()
                    self._entries[key] = entry
            except Exception:
                logger.warning("competence_tracker: failed to load", exc_info=True)

    def _save(self) -> None:
        """Persist competence map to disk."""
        path = self._dir / "competence_map.json"
        data = [e.to_dict() for e in self._entries.values()]
        path.write_text(json.dumps(data, indent=2))

    def register(
        self,
        domain: str,
        name: str,
        confidence: float,
        source: str = "",
        skill_id: str = "",
        notes: str = "",
    ) -> CompetenceEntry:
        """Register or update a competence entry."""
        key = f"{domain}:{name}".lower()
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            existing = self._entries.get(key)
            if existing:
                # Update — keep the higher confidence
                existing.confidence = max(existing.confidence, confidence)
                existing.last_verified = now
                if source:
                    existing.source = source
                if skill_id:
                    existing.skill_id = skill_id
                if notes:
                    existing.notes = notes
                existing.depth = existing.effective_depth()
                self._save()
                return existing
            else:
                entry = CompetenceEntry(
                    domain=domain,
                    name=name,
                    confidence=confidence,
                    depth=DEPTH_SHALLOW if confidence < 0.5 else DEPTH_INTERMEDIATE,
                    last_verified=now,
                    source=source,
                    skill_id=skill_id,
                    notes=notes,
                )
                self._entries[key] = entry
                self._save()
                return entry

    def record_usage(self, domain: str, name: str, success: bool) -> None:
        """Record that we used a competence (success or failure)."""
        key = f"{domain}:{name}".lower()
        with self._lock:
            entry = self._entries.get(key)
            if not entry:
                return
            entry.usage_count += 1
            if success:
                entry.usage_success_count += 1
                entry.confidence = min(1.0, entry.confidence + 0.02)
            else:
                entry.confidence = max(CONFIDENCE_FLOOR, entry.confidence - 0.05)
            entry.last_used = datetime.now(timezone.utc).isoformat()
            entry.depth = entry.effective_depth()
            self._save()

    def check_competence(self, domain: str, name: str) -> Optional[CompetenceEntry]:
        """Check if we have competence in a specific area."""
        key = f"{domain}:{name}".lower()
        entry = self._entries.get(key)
        if entry:
            entry.confidence = entry.effective_confidence()
            entry.depth = entry.effective_depth()
        return entry

    def check_task_readiness(self, requirements: list[dict]) -> dict:
        """Check readiness for a task given its requirements.

        Args:
            requirements: list of {domain, name} dicts

        Returns:
            {
                ready: bool,
                known: [{name, confidence, depth}],
                unknown: [{name, domain}],
                stale: [{name, confidence, last_verified}],
                estimated_learning_time_minutes: int,
            }
        """
        known = []
        unknown = []
        stale = []

        for req in requirements:
            domain = req.get("domain", "")
            name = req.get("name", "")
            entry = self.check_competence(domain, name)

            if entry is None or entry.effective_confidence() == 0:
                unknown.append({"name": name, "domain": domain})
            elif entry.effective_confidence() < 0.5:
                stale.append({
                    "name": name,
                    "confidence": entry.effective_confidence(),
                    "last_verified": entry.last_verified,
                })
            else:
                known.append({
                    "name": name,
                    "confidence": entry.effective_confidence(),
                    "depth": entry.effective_depth(),
                })

        # Estimate learning time (rough heuristic)
        learning_time = len(unknown) * 15 + len(stale) * 5  # minutes

        return {
            "ready": len(unknown) == 0 and len(stale) == 0,
            "known": known,
            "unknown": unknown,
            "stale": stale,
            "estimated_learning_time_minutes": learning_time,
        }

    def get_gaps(self, min_confidence: float = 0.5) -> list[CompetenceEntry]:
        """Find areas where our competence is below threshold."""
        gaps = []
        for entry in self._entries.values():
            if entry.effective_confidence() < min_confidence:
                gaps.append(entry)
        gaps.sort(key=lambda e: e.effective_confidence())
        return gaps

    def get_strengths(self, min_confidence: float = 0.8) -> list[CompetenceEntry]:
        """Find our strongest competence areas."""
        strengths = []
        for entry in self._entries.values():
            if entry.effective_confidence() >= min_confidence:
                strengths.append(entry)
        strengths.sort(key=lambda e: e.effective_confidence(), reverse=True)
        return strengths

    def sync_from_skill_library(self) -> int:
        """Update competence map from skill library state."""
        try:
            from app.atlas.skill_library import get_library
            library = get_library()
            updated = 0

            for manifest in library.search():
                domain = "apis" if manifest.category == "apis" else "patterns"
                self.register(
                    domain=domain,
                    name=manifest.name,
                    confidence=manifest.effective_confidence(),
                    source=f"skill_library:{manifest.source_type}",
                    skill_id=manifest.skill_id,
                )
                updated += 1

            return updated
        except Exception:
            return 0

    def format_competence_map(self) -> str:
        """Generate human-readable competence map."""
        if not self._entries:
            return "Competence map: empty"

        lines = ["🧠 Competence Map", ""]

        by_domain: dict[str, list] = {}
        for entry in self._entries.values():
            d = entry.domain or "other"
            if d not in by_domain:
                by_domain[d] = []
            by_domain[d].append(entry)

        for domain, entries in sorted(by_domain.items()):
            lines.append(f"  {domain.upper()} ({len(entries)} entries):")
            for e in sorted(entries, key=lambda x: x.effective_confidence(), reverse=True):
                conf = e.effective_confidence()
                status = "✅" if conf >= 0.8 else "⚠️" if conf >= 0.5 else "❌"
                depth = e.effective_depth()
                lines.append(f"    {status} {e.name:<30} conf={conf:.2f}  depth={depth}")
            lines.append("")

        return "\n".join(lines)


# ── Module-level singleton ───────────────────────────────────────────────────

_tracker: CompetenceTracker | None = None


def get_tracker() -> CompetenceTracker:
    """Get or create the singleton competence tracker."""
    global _tracker
    if _tracker is None:
        _tracker = CompetenceTracker()
    return _tracker
