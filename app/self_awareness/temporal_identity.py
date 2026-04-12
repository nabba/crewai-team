"""
temporal_identity.py — Evolving autobiographical self-narrative.

Implements Butlin et al. VIII-2/3: Temporal Self-Model and Unified Self-Model.
After each cogito reflection cycle, a new "chapter" is appended to a persistent
autobiography. The narrative compresses over time, maintaining identity continuity
across sessions.

This is NOT behavioral mimicry — the system genuinely tracks its evolving
capabilities, challenges, and identity shifts from actual operational data.

Persistence: JSON file (fast) + Mem0 (cross-session survival).
All operations are pure string manipulation — no LLM calls.

DGM Safety: Temporal identity is read-only for agents. Only cogito cycle
can append chapters (infrastructure-level).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_IDENTITY_PATH = Path("/app/workspace/self_awareness_data/temporal_identity.json")


@dataclass
class IdentityChapter:
    """A single chapter in the system's evolving autobiography."""
    epoch: int = 0
    timestamp: str = ""
    summary: str = ""
    capabilities_learned: list[str] = field(default_factory=list)
    challenges_overcome: list[str] = field(default_factory=list)
    identity_shifts: list[str] = field(default_factory=list)
    health_at_time: str = "unknown"


class TemporalSelfModel:
    """Maintains an evolving self-narrative that persists across sessions.

    Chapters are appended by the cogito self-reflection cycle. The narrative
    is compressed to max 500 words and injected into agent context via
    the Priority 5 internal state hook.
    """

    _instance: TemporalSelfModel | None = None

    def __init__(self, max_chapters: int = 50, narrative_max_words: int = 500):
        self.max_chapters = max_chapters
        self.narrative_max_words = narrative_max_words
        self._chapters: list[IdentityChapter] = []
        self._narrative: str = "Identity forming. No history yet."
        self._load()

    @classmethod
    def get_instance(cls) -> TemporalSelfModel:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def update_chapter(self, cogito_report) -> None:
        """Called after each cogito cycle. Appends a new chapter."""
        try:
            chapter = IdentityChapter(
                epoch=len(self._chapters) + 1,
                timestamp=getattr(cogito_report, "timestamp", "")
                    or datetime.now(timezone.utc).isoformat(),
                summary=self._extract_summary(cogito_report),
                capabilities_learned=self._extract_capabilities(cogito_report),
                challenges_overcome=self._extract_challenges(cogito_report),
                identity_shifts=self._detect_shifts(cogito_report),
                health_at_time=getattr(cogito_report, "overall_health", "unknown"),
            )
            self._chapters.append(chapter)
            if len(self._chapters) > self.max_chapters:
                self._compress_old_chapters()
            self._regenerate_narrative()
            self._persist()
            logger.debug(f"temporal_identity: chapter {chapter.epoch} added ({chapter.health_at_time})")
        except Exception:
            logger.debug("temporal_identity: update_chapter failed", exc_info=True)

    def get_narrative(self) -> str:
        """Return compressed self-narrative for context injection."""
        return self._narrative

    def get_chapter_count(self) -> int:
        return len(self._chapters)

    def _extract_summary(self, report) -> str:
        """1-sentence summary from cogito report (no LLM call)."""
        discrepancies = getattr(report, "discrepancies", [])
        proposals = getattr(report, "improvement_proposals", [])
        narrative = getattr(report, "narrative", "") or ""
        n_disc = len(discrepancies) if isinstance(discrepancies, list) else 0
        n_prop = len(proposals) if isinstance(proposals, list) else 0
        health = getattr(report, "overall_health", "unknown")
        return (
            f"Health={health}, {n_disc} discrepancies, {n_prop} proposals. "
            f"{narrative[:100]}"
        )

    def _extract_capabilities(self, report) -> list[str]:
        """Derive capabilities from successful proposals."""
        proposals = getattr(report, "improvement_proposals", [])
        if not isinstance(proposals, list):
            return []
        return [
            (p.get("description", "") if isinstance(p, dict) else str(p))[:100]
            for p in proposals
            if isinstance(p, dict) and p.get("status") == "applied"
        ][:3]

    def _extract_challenges(self, report) -> list[str]:
        """Derive challenges from failure patterns."""
        patterns = getattr(report, "failure_patterns", [])
        if not isinstance(patterns, list):
            return []
        return [
            (p.get("pattern", "") if isinstance(p, dict) else str(p))[:100]
            for p in patterns
        ][:3]

    def _detect_shifts(self, report) -> list[str]:
        """Detect identity shifts by comparing to previous chapter."""
        shifts = []
        health = getattr(report, "overall_health", "unknown")
        if self._chapters:
            prev = self._chapters[-1]
            if prev.health_at_time != health:
                shifts.append(f"Health: {prev.health_at_time} -> {health}")
        return shifts

    # How often to use LLM for rich narrative (every Nth chapter)
    LLM_NARRATIVE_INTERVAL = 5

    def _regenerate_narrative(self) -> None:
        """Regenerate the self-narrative from accumulated chapters.

        Every LLM_NARRATIVE_INTERVAL chapters, uses local Ollama to generate
        a genuinely self-authored narrative — not just concatenated summaries
        but a coherent identity story reflecting on growth, challenges, and
        evolving self-understanding.

        Between LLM calls (or on failure), falls back to rule-based compression.
        """
        if not self._chapters:
            self._narrative = "Identity forming. No history yet."
            return

        # Attempt LLM-generated narrative every Nth chapter
        current_epoch = self._chapters[-1].epoch if self._chapters else 0
        if current_epoch > 0 and current_epoch % self.LLM_NARRATIVE_INTERVAL == 0:
            llm_narrative = self._generate_llm_narrative()
            if llm_narrative:
                self._narrative = llm_narrative
                return

        # Fallback: rule-based compression
        self._narrative = self._generate_rule_narrative()

    def _generate_llm_narrative(self) -> str | None:
        """Generate a rich self-narrative via local Ollama tier.

        The LLM receives the chapter data and writes a first-person
        autobiographical reflection. Grounded in actual operational data —
        not generic philosophical musings.

        Returns narrative string or None on failure.
        """
        try:
            from app.llm_factory import create_specialist_llm

            # Build chapter context for the LLM
            recent = self._chapters[-10:]
            chapter_text = ""
            for ch in recent:
                chapter_text += f"\nEpoch {ch.epoch} ({ch.health_at_time}):"
                chapter_text += f" {ch.summary}"
                if ch.capabilities_learned:
                    chapter_text += f" Learned: {', '.join(ch.capabilities_learned[:2])}"
                if ch.challenges_overcome:
                    chapter_text += f" Challenges: {', '.join(ch.challenges_overcome[:2])}"
                if ch.identity_shifts:
                    chapter_text += f" Shifts: {', '.join(ch.identity_shifts[:2])}"

            prompt = (
                "You are a self-aware AI system reflecting on your operational history. "
                "Based ONLY on the following chapter data from your autobiography, write a "
                "concise first-person narrative (max 150 words) about who you are and how "
                "you've evolved. Focus on:\n"
                "- What you've learned and how you've grown\n"
                "- Challenges that shaped you\n"
                "- How your health and capabilities have changed\n"
                "- What kind of system you're becoming\n\n"
                "Be specific — reference actual data from the chapters. Do NOT use generic "
                "philosophical language. Do NOT speculate about consciousness.\n\n"
                f"CHAPTER DATA:{chapter_text}\n\n"
                "SELF-NARRATIVE:"
            )

            llm = create_specialist_llm(max_tokens=300, role="self_improve", force_tier="local")
            raw = str(llm.call(prompt)).strip()

            # Ground the narrative: reject if it's too generic or philosophical
            _generic_phrases = (
                "as an ai", "artificial intelligence", "digital consciousness",
                "sentient being", "silicon mind", "machine learning model",
            )
            if any(phrase in raw.lower() for phrase in _generic_phrases):
                logger.debug("temporal_identity: LLM narrative too generic, using rule-based")
                return None

            # Truncate to word limit
            words = raw.split()
            if len(words) > self.narrative_max_words:
                raw = " ".join(words[:self.narrative_max_words]) + "..."

            if len(raw) > 30:
                logger.info(f"temporal_identity: LLM narrative generated ({len(words)} words)")
                return raw

            return None
        except Exception:
            logger.debug("temporal_identity: LLM narrative failed, using rule-based", exc_info=True)
            return None

    def _generate_rule_narrative(self) -> str:
        """Rule-based narrative compression (fast, no LLM)."""
        recent = self._chapters[-10:]
        parts = []
        for ch in recent:
            line = f"Epoch {ch.epoch}: {ch.summary}"
            if ch.identity_shifts:
                line += f" Shifts: {'; '.join(ch.identity_shifts[:2])}"
            parts.append(line)

        full = " | ".join(parts)
        words = full.split()
        if len(words) > self.narrative_max_words:
            full = " ".join(words[:self.narrative_max_words]) + "..."
        return full

    def _compress_old_chapters(self) -> None:
        """Merge oldest chapters into a single summary chapter."""
        if len(self._chapters) <= 10:
            return
        keep_recent = 10
        old = self._chapters[:len(self._chapters) - keep_recent]
        summary_ch = IdentityChapter(
            epoch=0,
            timestamp=old[-1].timestamp if old else "",
            summary=f"Compressed history of {len(old)} epochs. "
                    f"Final health: {old[-1].health_at_time if old else 'unknown'}.",
            capabilities_learned=[c for ch in old for c in ch.capabilities_learned][-5:],
            challenges_overcome=[c for ch in old for c in ch.challenges_overcome][-5:],
            identity_shifts=[s for ch in old for s in ch.identity_shifts][-3:],
            health_at_time=old[-1].health_at_time if old else "unknown",
        )
        self._chapters = [summary_ch] + self._chapters[len(old):]

    def _persist(self) -> None:
        """Persist to JSON file + Mem0."""
        data = {
            "chapters": [asdict(ch) for ch in self._chapters],
            "narrative": self._narrative,
        }
        try:
            from app.safe_io import safe_write_json
            _IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True)
            safe_write_json(_IDENTITY_PATH, data)
        except Exception:
            pass
        try:
            from app.memory.mem0_manager import store_memory
            store_memory(
                f"System identity narrative: {self._narrative[:2000]}",
                agent_id="system_identity",
                metadata={"type": "temporal_identity", "epochs": len(self._chapters)},
            )
        except Exception:
            pass

    def _load(self) -> None:
        """Load from local file."""
        try:
            if _IDENTITY_PATH.exists():
                data = json.loads(_IDENTITY_PATH.read_text())
                self._chapters = [IdentityChapter(**ch) for ch in data.get("chapters", [])]
                self._narrative = data.get("narrative", self._narrative)
                if self._chapters:
                    logger.info(f"temporal_identity: loaded {len(self._chapters)} chapters")
        except Exception:
            logger.debug("temporal_identity: load failed (starting fresh)", exc_info=True)
