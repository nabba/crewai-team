"""Person suggestions — Level 3 of person-correlation.

PROGRAM §42 (2026-05-11) — Q4.2 Level 3.

PRESCRIPTIVE feature: the system emits operator-facing suggestions
about people based on presence patterns. Two categories:

  * ``dormancy_nudge``       — high-presence person, no appearance 60d+
  * ``responsiveness_nudge`` — person mentioned in N unanswered threads

Each category is independently opt-in. All suggestions phrased as
QUESTIONS, never directives. Rate limit shares the L4.4 cap
(≤3 per briefing total).

Master switches:
  * ``person_correlation_enabled`` (L1)
  * ``person_suggestions_enabled`` (L3 master)
  * ``person_suggestions_dormancy_enabled`` (category opt-in)
  * ``person_suggestions_responsiveness_enabled`` (category opt-in)

Per-person mute-suggestions independent of L1 mute: a person can be
present in L1 surfaces but excluded from suggestions. The
``person_mute_suggestions.json`` file holds the suggestion-mute list.

Emitted log at ``workspace/companion/person_suggestions_emitted.jsonl``
for operator transparency.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_PER_BRIEFING_CAP = 3
_DORMANCY_THRESHOLD_DAYS = 60
_HIGH_PRESENCE_MODALITY_THRESHOLD = 3
# Q4.2.1#6 — re-emission cooldown: don't re-fire the same (category,
# person_id) within this window. Keeps morning briefings from broadcasting
# the same dormancy nudge daily until the operator acts.
_REEMIT_COOLDOWN_HOURS = 24


def _default_emitted_log() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "companion" / "person_suggestions_emitted.jsonl"
    except Exception:
        return Path("/app/workspace/companion/person_suggestions_emitted.jsonl")


def _default_sug_mutes() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "companion" / "person_mute_suggestions.json"
    except Exception:
        return Path("/app/workspace/companion/person_mute_suggestions.json")


# ── Master switches ──────────────────────────────────────────────────────


def _enabled() -> bool:
    try:
        from app.runtime_settings import (
            get_person_correlation_enabled,
            get_person_suggestions_enabled,
        )
        return get_person_correlation_enabled() and get_person_suggestions_enabled()
    except Exception:
        return False


def _dormancy_enabled() -> bool:
    if not _enabled():
        return False
    try:
        from app.runtime_settings import get_person_suggestions_dormancy_enabled
        return get_person_suggestions_dormancy_enabled()
    except Exception:
        return False


def _responsiveness_enabled() -> bool:
    if not _enabled():
        return False
    try:
        from app.runtime_settings import get_person_suggestions_responsiveness_enabled
        return get_person_suggestions_responsiveness_enabled()
    except Exception:
        return False


# ── Per-person suggestion mutes ─────────────────────────────────────────


def _load_sug_mutes() -> set[str]:
    p = _default_sug_mutes()
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return set(str(x).lower() for x in (data.get("muted") or []))
    except (OSError, json.JSONDecodeError):
        return set()


def mute_suggestions_for(person_id: str) -> bool:
    pid = (person_id or "").strip().lower()
    if not pid:
        return False
    p = _default_sug_mutes()
    p.parent.mkdir(parents=True, exist_ok=True)
    muted = _load_sug_mutes()
    if pid in muted:
        return False
    muted.add(pid)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"muted": sorted(muted)}, indent=2), encoding="utf-8")
    tmp.replace(p)
    return True


def unmute_suggestions_for(person_id: str) -> bool:
    pid = (person_id or "").strip().lower()
    p = _default_sug_mutes()
    muted = _load_sug_mutes()
    if pid not in muted:
        return False
    muted.discard(pid)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"muted": sorted(muted)}, indent=2), encoding="utf-8")
    tmp.replace(p)
    return True


# ── Suggestion data shape ────────────────────────────────────────────────


@dataclass
class PersonSuggestion:
    category: str          # "dormancy_nudge" / "responsiveness_nudge"
    person_id: str
    display_name: str      # "" if unknown
    text: str              # operator-facing question — never directive
    detected_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Generators ───────────────────────────────────────────────────────────


def _generate_dormancy_nudges(profile_people: list[dict]) -> list[PersonSuggestion]:
    """Find high-modality-count people who haven't appeared in 60d+."""
    if not _dormancy_enabled():
        return []
    sug_mutes = _load_sug_mutes()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=_DORMANCY_THRESHOLD_DAYS)
    out: list[PersonSuggestion] = []
    for p in profile_people:
        pid = (p.get("person_id") or "").lower()
        if not pid or pid in sug_mutes:
            continue
        if int(p.get("modality_count") or 0) < _HIGH_PRESENCE_MODALITY_THRESHOLD:
            continue
        last_seen_str = p.get("last_seen") or ""
        try:
            last_seen = datetime.fromisoformat(last_seen_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        if last_seen > cutoff:
            continue
        days = int((now - last_seen).total_seconds() / 86400.0)
        display = (p.get("display_names") or [""])[0] or pid
        out.append(PersonSuggestion(
            category="dormancy_nudge",
            person_id=pid,
            display_name=display,
            text=f"{display} hasn't appeared in {days}d "
                 f"(previously crossed {p.get('modality_count')} modalities). "
                 f"Reach out?",
            detected_at=now.isoformat(),
        ))
    return out


def _generate_responsiveness_nudges(profile_people: list[dict]) -> list[PersonSuggestion]:
    """Stub: flag people mentioned in recent unreplied emails.

    Real implementation requires gmail thread-state introspection;
    for the first ship we use a conservative heuristic — anyone whose
    cooccurring_topics include certain "action" keywords AND who has
    very recent occurrences spike. This is intentionally narrow."""
    if not _responsiveness_enabled():
        return []
    # Honest stance: a proper responsiveness signal needs gmail
    # thread-state which we don't have wired. Returning empty rather
    # than guessing. Future ship can wire thread-state.
    return []


# ── Public entry ─────────────────────────────────────────────────────────


def generate_suggestions() -> list[dict[str, Any]]:
    """Return ≤3 suggestions (combined across categories). No-op when
    master switches off. Records emitted suggestions in the log."""
    if not _enabled():
        return []
    try:
        from app.companion.person_model import current_profile
        prof = current_profile() or {}
    except Exception:
        return []
    people = prof.get("people") or []
    if not people:
        return []

    # Q4.2.2#2 — Affect/welfare gate: a dormancy nudge during a critical
    # affect window is exactly the wrong pressure. The arbiter already
    # gates `notify()` consumers; the briefing-direct path here did not.
    # Suppress emission entirely under welfare breach.
    try:
        from app.notify.arbiter import welfare_breaching
        if welfare_breaching():
            logger.debug("person_suggestions: suppressed under welfare breach")
            return []
    except Exception:
        # Failure-isolated — if affect probing fails, fall through to
        # normal generation rather than suppressing silently.
        logger.debug("person_suggestions: welfare probe failed", exc_info=True)

    suggestions: list[PersonSuggestion] = []
    suggestions.extend(_generate_dormancy_nudges(people))
    suggestions.extend(_generate_responsiveness_nudges(people))

    # Q4.2 — graph-driven suggestions when L4.4 enabled. These compete
    # for the same per-briefing cap so the system can't drown the
    # operator in graph-derived nudges even at high graph activity.
    try:
        from app.companion.graph_features.graph_suggestions import (
            generate_graph_suggestions,
        )
        for gs in generate_graph_suggestions(people):
            suggestions.append(gs)
    except Exception:
        logger.debug("person_suggestions: graph suggestions skipped", exc_info=True)

    # Q4.2.1#1 — Apply mute-suggestions filter to the FULL merged list
    # so L4.4 bridge/weak-tie nudges respect /person mute-suggestions
    # the same way L3 dormancy nudges do. Without this filter an
    # operator who muted Maria's L3 nudge still got L4.4 nudges about
    # her, breaking the advertised feature.
    sug_mutes = _load_sug_mutes()
    suggestions = [s for s in suggestions if s.person_id not in sug_mutes]

    # Q4.2.1#6 — Re-emission cooldown: skip (category, person_id) we've
    # already emitted within the last 24h so daily briefings don't
    # broadcast the same nudge every morning.
    recent_keys = _recent_emission_keys(_REEMIT_COOLDOWN_HOURS)

    # Rate limit + dedupe by person_id (one suggestion per person max
    # to avoid double-pinging the same person across categories).
    seen_pids: set[str] = set()
    capped: list[dict[str, Any]] = []
    for s in suggestions:
        if s.person_id in seen_pids:
            continue
        if (s.category, s.person_id) in recent_keys:
            continue
        capped.append(s.to_dict())
        seen_pids.add(s.person_id)
        if len(capped) >= _PER_BRIEFING_CAP:
            break

    # Log emitted suggestions for operator review.
    if capped:
        _log_emitted(capped)

    return capped


def _log_emitted(suggestions: list[dict[str, Any]]) -> None:
    path = _default_emitted_log()
    try:
        from app.utils.jsonl_retention import append_with_cap
        for s in suggestions:
            append_with_cap(
                path, json.dumps(s, sort_keys=True), max_lines=2000,
            )
    except Exception:
        logger.debug("person_suggestions: log append failed", exc_info=True)


def recent_emitted(limit: int = 50) -> list[dict[str, Any]]:
    """Read recent emitted suggestions for the operator review surface.

    Q4.2.1#2 — gate on master switch. When the operator has disabled L3
    suggestions, the historical log shouldn't leak via this endpoint.
    The audit trail remains on disk for review if they re-enable.
    """
    if not _enabled():
        return []
    path = _default_emitted_log()
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
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    rows.sort(key=lambda r: r.get("detected_at", ""), reverse=True)
    return rows[:limit]


def _recent_emission_keys(window_hours: int) -> set[tuple[str, str]]:
    """Q4.2.1#6 — return (category, person_id) tuples emitted within the
    last ``window_hours``. Used by ``generate_suggestions`` to dedupe
    against prior briefings."""
    path = _default_emitted_log()
    if not path.exists():
        return set()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    keys: set[tuple[str, str]] = set()
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
                ts_str = row.get("detected_at") or ""
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if ts < cutoff:
                    continue
                cat = row.get("category") or ""
                pid = (row.get("person_id") or "").lower()
                if cat and pid:
                    keys.add((cat, pid))
    except OSError:
        return set()
    return keys
