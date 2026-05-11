"""Cross-modal pattern detector — convergence signals across the
operator's inputs.

PROGRAM §41 (2026-05-11) — Q4 Item 15.

`interest_model` already aggregates topics from 5 sources into a
single score with recency-weighted decay. That answers "what topics
is Andrus interested in?" — an aggregate question.

This module answers a different question: **"which topics are
crossing modalities at unusual rates?"** — a convergence question.
A topic that appears in calendar AND email AND tickets in the same
3-week window is a stronger signal than a topic with a high aggregate
score that only ever appears in one source.

Sources consumed (read-only):

  * ``convs``     — conversations.db (via interest_model.current_profile)
  * ``emails``    — email subjects (via interest_model.current_profile)
  * ``events``    — calendar event titles (via interest_model.current_profile)
  * ``feedback``  — companion FEEDBACK events (via interest_model.current_profile)
  * ``affect``    — affect-episode topics (via interest_model.current_profile)
  * ``tickets``   — control_plane.tickets titles (Q4 addition)

Pattern strength formula:

    modality_factor   = min(1.0, modality_count / 4.0)
        # 1.0 at 4+ modalities, 0.75 at 3, 0.5 at 2, 0.25 at 1
    volume_factor     = clip01(log10(total + 1) / log10(20))
        # 0 at total=0, 1.0 at total=19
    strength          = modality_factor * volume_factor

Threshold for an emitted pattern (configurable):
    modalities ≥ 3 AND total ≥ 8 AND strength ≥ 0.7

Output:
  * Persistent: ``workspace/companion/cross_modal_patterns.jsonl``
    (append-only with archive rotation via existing helper)
  * Surfaced: daily-briefing "💡 Proactive insights" section

Cross-link to Q4#16: when a pattern hits a topic that matches an
open tension's question, ``tensions.boost_freshness_for_topic`` is
called so the tension's last_touched_at bumps.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────


_WINDOW_DAYS = 21
_MIN_MODALITIES = 3
_MIN_TOTAL_OCCURRENCES = 8
_MIN_STRENGTH = 0.7

_PATTERNS_FILE = Path("/app/workspace/companion/cross_modal_patterns.jsonl")
_PATTERNS_MAX_LINES = 5_000   # archive-rotated; preserves history


def _default_patterns_file() -> Path:
    """Lazy-resolve so a non-default WORKSPACE_ROOT is honored."""
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "companion" / "cross_modal_patterns.jsonl"
    except Exception:
        return _PATTERNS_FILE


# ── Data model ───────────────────────────────────────────────────────────


@dataclass
class Pattern:
    """One cross-modal convergence detected."""
    topic: str
    modalities: list[str]                 # which sources contained the topic
    occurrences_per_modality: dict[str, int]
    occurrences_total: int
    window_days: int
    strength: float                        # 0..1, see formula above
    detected_at: str                       # ISO-8601
    first_seen_age_days: float | None      # from interest_model.last_seen_age_days
    triggered_tension_boost: int           # how many tensions had freshness boosted

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Strength computation ─────────────────────────────────────────────────


def _strength(modality_count: int, total: int) -> float:
    if modality_count <= 0 or total <= 0:
        return 0.0
    modality_factor = min(1.0, modality_count / 4.0)
    # log10(total + 1) / log10(20) → 0 at total=0, 1 at total≈19
    volume_factor = math.log10(total + 1) / math.log10(20)
    volume_factor = max(0.0, min(1.0, volume_factor))
    return round(modality_factor * volume_factor, 4)


# ── Ticket subject source (Q4 addition) ──────────────────────────────────


def _ticket_titles_recent(window_days: int) -> list[tuple[str, str]]:
    """Return (title, ticket_id) for tickets updated in the window.
    Read-only; failure-isolated."""
    try:
        from app.control_plane.db import execute
    except Exception:
        return []
    try:
        rows = execute(
            """SELECT id::text AS id, title
                 FROM control_plane.tickets
                WHERE updated_at >= NOW() - (INTERVAL '1 day' * %s)
                  AND title IS NOT NULL
             ORDER BY updated_at DESC LIMIT 500""",
            (int(window_days),), fetch=True,
        ) or []
    except Exception:
        logger.debug("cross_modal_patterns: ticket query failed", exc_info=True)
        return []
    return [
        (str(r.get("title") or ""), str(r.get("id") or ""))
        for r in rows if r.get("title")
    ]


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")


def _ticket_modality_counts(topics: list[dict], window_days: int) -> dict[str, int]:
    """For each topic name, count ticket titles whose tokens include
    the topic. Bigrams in the topic name are matched as substrings."""
    if not topics:
        return {}
    titles = _ticket_titles_recent(window_days)
    if not titles:
        return {}
    counts: dict[str, int] = {}
    topic_names = [t["name"] for t in topics if isinstance(t.get("name"), str)]
    # Pre-lowercase topic names for substring matching.
    needles = [(n, n.lower()) for n in topic_names if n]
    for title, _tid in titles:
        title_lower = (title or "").lower()
        if not title_lower:
            continue
        for orig, lo in needles:
            if lo in title_lower:
                counts[orig] = counts.get(orig, 0) + 1
    return counts


# ── Detector ─────────────────────────────────────────────────────────────


def detect_patterns(
    window_days: int = _WINDOW_DAYS,
    min_modalities: int = _MIN_MODALITIES,
    min_total: int = _MIN_TOTAL_OCCURRENCES,
    min_strength: float = _MIN_STRENGTH,
) -> list[Pattern]:
    """Read the current interest profile, add ticket-modality counts,
    emit Pattern records for topics crossing the thresholds.

    Read-only on interest_model state; failure-isolated.
    """
    try:
        from app.companion.interest_model import current_profile
        profile = current_profile() or {}
    except Exception:
        logger.debug("cross_modal_patterns: interest_model unavailable", exc_info=True)
        return []
    topics = profile.get("topics") or []
    if not topics:
        return []

    # Cross-modal ticket source — Q4 addition layered onto interest_model's
    # 5 existing sources.
    ticket_counts = _ticket_modality_counts(topics, window_days)

    out: list[Pattern] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for t in topics:
        if not isinstance(t, dict):
            continue
        name = str(t.get("name") or "").strip()
        if not name:
            continue
        sources = dict(t.get("sources") or {})
        # Layer the new ticket modality on top of existing.
        if name in ticket_counts and ticket_counts[name] > 0:
            sources["tickets"] = ticket_counts[name]
        modality_count = sum(1 for v in sources.values() if v > 0)
        total = sum(sources.values())
        strength = _strength(modality_count, total)
        if (
            modality_count < min_modalities
            or total < min_total
            or strength < min_strength
        ):
            continue
        try:
            last_age = float(t.get("last_seen_age_days") or 0.0)
        except (TypeError, ValueError):
            last_age = None
        # Cross-link to Q4#16: bump freshness of any open tension whose
        # question references this topic.
        boosted = _boost_matching_tensions(name)
        out.append(Pattern(
            topic=name,
            modalities=[m for m, v in sources.items() if v > 0],
            occurrences_per_modality={m: int(v) for m, v in sources.items() if v > 0},
            occurrences_total=int(total),
            window_days=window_days,
            strength=strength,
            detected_at=now_iso,
            first_seen_age_days=last_age,
            triggered_tension_boost=boosted,
        ))
    # Sort newest-first (by strength desc as tiebreaker).
    out.sort(key=lambda p: (p.detected_at, p.strength), reverse=True)
    return out


def _boost_matching_tensions(topic: str) -> int:
    """Best-effort cross-link to Q4#16. Never raises."""
    try:
        from app.companion.tensions import boost_freshness_for_topic
        return boost_freshness_for_topic(topic)
    except Exception:
        logger.debug(
            "cross_modal_patterns: tension boost failed", exc_info=True,
        )
        return 0


# ── Persistence ──────────────────────────────────────────────────────────


def _persist_patterns(patterns: list[Pattern]) -> int:
    """Append each pattern to the JSONL with archive rotation. Returns
    count persisted."""
    if not patterns:
        return 0
    path = _default_patterns_file()
    try:
        from app.utils.jsonl_retention import append_with_archive_rotate
    except Exception:
        return 0
    persisted = 0
    for p in patterns:
        try:
            append_with_archive_rotate(
                path,
                json.dumps(p.to_dict(), sort_keys=True),
                max_lines=_PATTERNS_MAX_LINES,
            )
            persisted += 1
        except Exception:
            logger.debug(
                "cross_modal_patterns: persist failed for %s", p.topic,
                exc_info=True,
            )
    return persisted


def list_recent_patterns(
    n: int = 20, min_strength: float = _MIN_STRENGTH,
) -> list[dict[str, Any]]:
    """Read recent persisted patterns from disk. Newest-first."""
    path = _default_patterns_file()
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if float(row.get("strength") or 0.0) < min_strength:
                    continue
                rows.append(row)
    except OSError:
        logger.debug("cross_modal_patterns: read failed", exc_info=True)
        return []
    # Reverse so newest-first.
    rows.sort(key=lambda r: r.get("detected_at", ""), reverse=True)
    return rows[:n]


# ── Idle-job entry ───────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    """One detection pass. Cadence-guarded by the caller
    (companion.loop / healing monitors). Read-only on inputs; persists
    new patterns + boosts matching tensions."""
    try:
        patterns = detect_patterns()
    except Exception:
        logger.debug("cross_modal_patterns: detect raised", exc_info=True)
        return {"ok": False, "patterns_detected": 0}
    persisted = _persist_patterns(patterns)
    summary = {
        "ok": True,
        "patterns_detected": len(patterns),
        "persisted": persisted,
        "tension_boosts": sum(p.triggered_tension_boost for p in patterns),
    }
    if patterns:
        # Optional GW publish so SubIA sees convergence events.
        try:
            from app.workspace_publish import publish_to_workspace
            top = patterns[0]
            publish_to_workspace(
                source="cross_modal_patterns",
                content=(
                    f"Cross-modal convergence: {top.topic!r} appeared in "
                    f"{len(top.modalities)} modalities × {top.occurrences_total} "
                    f"hits over {top.window_days}d. Strength {top.strength:.2f}."
                ),
                salience=min(0.7, 0.4 + 0.1 * len(patterns)),
                signal_type="trend_reversal",  # convergence shifts a baseline
            )
        except Exception:
            logger.debug(
                "cross_modal_patterns: GW publish failed", exc_info=True,
            )
    return summary
