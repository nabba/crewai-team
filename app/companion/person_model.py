"""Person presence model — Level 1 of the person-correlation stack.

PROGRAM §42 (2026-05-11) — Q4.2 Level 1.

Aggregates per-person modality counts across the operator's input
channels: gmail senders, calendar attendees, conversation_store
participants. **Skips ticket assignees** (those are agent roles,
not humans).

Identity model: canonical email address as the person_id. Aliasing
(maria@old vs maria@new) is deferred — operator handles via
``/person mute`` or ``/person forget``.

Master switch: ``person_correlation_enabled`` (default OFF). With
switch OFF, ``compile_profile`` no-ops and the idle job never runs.

Storage:
  * ``workspace/companion/person_profile.json``  — current snapshot
  * ``workspace/companion/person_history.jsonl`` — append-only log
                                                    of sightings
  * ``workspace/companion/person_mutes.json``    — operator mute list

Decay: per-person counts decay if untouched for ``decay_months``
(default 12, configurable). The DECAY here is profile-level —
inactive people fade out of surfaces. Distinct from tensions which
preserve forever; people deserve more forgetting.

Goodhart guards:
  * No body parsing (only sender email + display name extraction)
  * No ranking surface — counts only, sorted by recency not score
  * Per-person mute respected at the read path (muted people are
    excluded from ALL surfaces including counts)
  * Per-person forget deletes the entry entirely
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────


_DEFAULT_DECAY_MONTHS = 12
_RUN_CADENCE_S = 12 * 3600     # 12h — matches interest_model

_lock = threading.RLock()  # reentrant: forget() holds it and calls unmute()


def _default_profile_path() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "companion" / "person_profile.json"
    except Exception:
        return Path("/app/workspace/companion/person_profile.json")


def _default_history_path() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "companion" / "person_history.jsonl"
    except Exception:
        return Path("/app/workspace/companion/person_history.jsonl")


def _default_mutes_path() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "companion" / "person_mutes.json"
    except Exception:
        return Path("/app/workspace/companion/person_mutes.json")


def _enabled() -> bool:
    """Master switch. Default OFF."""
    try:
        from app.runtime_settings import get_person_correlation_enabled
        return get_person_correlation_enabled()
    except Exception:
        return False


# ── Data model ───────────────────────────────────────────────────────────


@dataclass
class PersonProfile:
    """Per-person presence record. Counts only — no body content."""
    person_id: str                          # canonical email
    display_names: list[str] = field(default_factory=list)
    first_seen: str = ""                    # ISO-8601 UTC
    last_seen: str = ""                     # bumps on every new sighting
    occurrences_per_modality: dict[str, int] = field(default_factory=dict)
    # Topic co-occurrence: when this person was sighted, what topics
    # appeared alongside? Populated when L1 is enabled — feeds the
    # person×topic surface.
    cooccurring_topics: dict[str, int] = field(default_factory=dict)

    def total_occurrences(self) -> int:
        return sum(self.occurrences_per_modality.values())

    def modality_count(self) -> int:
        return sum(1 for v in self.occurrences_per_modality.values() if v > 0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PersonProfile":
        return cls(
            person_id=str(d.get("person_id") or ""),
            display_names=list(d.get("display_names") or []),
            first_seen=str(d.get("first_seen") or ""),
            last_seen=str(d.get("last_seen") or ""),
            occurrences_per_modality=dict(d.get("occurrences_per_modality") or {}),
            cooccurring_topics=dict(d.get("cooccurring_topics") or {}),
        )


# ── Mutes ────────────────────────────────────────────────────────────────


def _load_mutes(path: Path | None = None) -> set[str]:
    p = path or _default_mutes_path()
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return set(str(x).lower() for x in (data.get("muted") or []))
    except (OSError, json.JSONDecodeError):
        return set()


def _save_mutes(muted: set[str], path: Path | None = None) -> None:
    p = path or _default_mutes_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(
        {"muted": sorted(muted)}, indent=2,
    ), encoding="utf-8")
    tmp.replace(p)


def mute(person_id: str, path: Path | None = None) -> bool:
    """Add ``person_id`` to the mute list. Muted people are excluded
    from ALL surfaces. Returns True if newly muted, False if already."""
    pid = (person_id or "").strip().lower()
    if not pid:
        return False
    with _lock:
        muted = _load_mutes(path)
        if pid in muted:
            return False
        muted.add(pid)
        _save_mutes(muted, path)
    return True


def unmute(person_id: str, path: Path | None = None) -> bool:
    """Remove from the mute list. Returns True if was muted."""
    pid = (person_id or "").strip().lower()
    with _lock:
        muted = _load_mutes(path)
        if pid not in muted:
            return False
        muted.discard(pid)
        _save_mutes(muted, path)
    return True


def is_muted(person_id: str, path: Path | None = None) -> bool:
    return (person_id or "").strip().lower() in _load_mutes(path)


# ── Profile storage ──────────────────────────────────────────────────────


def _load_profile(path: Path | None = None) -> dict[str, PersonProfile]:
    p = path or _default_profile_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, PersonProfile] = {}
    for pid, row in (data.get("people") or {}).items():
        if isinstance(row, dict):
            out[pid] = PersonProfile.from_dict(row)
    return out


def _save_profile(profile: dict[str, PersonProfile], path: Path | None = None) -> None:
    p = path or _default_profile_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "people": {pid: pp.to_dict() for pid, pp in profile.items()},
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def forget(person_id: str, path: Path | None = None) -> bool:
    """Delete a single person's profile entry. Mute is independent —
    forgetting unmutes too (they're gone from tracking entirely)."""
    pid = (person_id or "").strip().lower()
    if not pid:
        return False
    with _lock:
        profile = _load_profile(path)
        if pid not in profile:
            unmute(pid)
            return False
        del profile[pid]
        _save_profile(profile, path)
        # Also remove from mute list since they're no longer tracked.
        unmute(pid)
    return True


def forget_all(path: Path | None = None) -> int:
    """Delete the entire person profile + mute list. Returns the
    count of people forgotten."""
    with _lock:
        profile = _load_profile(path)
        n = len(profile)
        _save_profile({}, path)
        _save_mutes(set())
    # Q4.2.2#1 — major policy reversal → continuity ledger. Opaque
    # count only, never person_ids.
    if n > 0:
        try:
            from app.identity.continuity_ledger import record_event
            record_event(
                kind="person_correlation_policy",
                actor="operator",
                summary=f"forget_all — {n} people removed from profile",
                detail={"level": "L1", "action": "forget_all", "count": n},
            )
        except Exception:
            logger.debug("forget_all ledger emit failed", exc_info=True)
    return n


# ── Source collectors ────────────────────────────────────────────────────


_EMAIL_RE = re.compile(r"[\w\.\-\+]+@[\w\.\-]+\.\w+")


def _normalize_email(raw: str) -> str:
    """Lowercase canonical form. Strip display-name parts."""
    if not raw:
        return ""
    m = _EMAIL_RE.search(raw)
    if not m:
        return raw.strip().lower()
    return m.group(0).strip().lower()


def _extract_display_name(raw: str) -> str:
    """If raw is 'Maria Smith <maria@example.com>', return 'Maria Smith'.
    If just an email, return ''."""
    if not raw or "<" not in raw:
        return ""
    name = raw.split("<", 1)[0].strip().strip('"').strip("'")
    return name if name else ""


def _gather_email_senders(lookback_days: int) -> list[tuple[str, str, float]]:
    """Yield (email, display_name, age_days). Soft fail."""
    try:
        from app.tools.gmail_tools import _list_recent
    except Exception:
        return []
    try:
        msgs = _list_recent(limit=100) or []
    except Exception:
        return []
    out: list[tuple[str, str, float]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        raw = str(m.get("from") or "")
        email = _normalize_email(raw)
        if not email:
            continue
        out.append((email, _extract_display_name(raw), 0.5))
    return out


def _gather_calendar_attendees(lookback_days: int) -> list[tuple[str, str, float]]:
    """Yield (email, display_name, age_days). Soft fail."""
    try:
        from app.tools.gcal_tools import _list_events
    except Exception:
        return []
    now = datetime.now(timezone.utc)
    time_min = (now - timedelta(days=lookback_days)).isoformat().replace("+00:00", "Z")
    time_max = (now + timedelta(days=14)).isoformat().replace("+00:00", "Z")
    try:
        events = _list_events(time_min=time_min, time_max=time_max, max_results=50)
    except Exception:
        return []
    out: list[tuple[str, str, float]] = []
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        attendees = ev.get("attendees") or []
        for a in attendees:
            if isinstance(a, dict):
                email = _normalize_email(str(a.get("email") or ""))
                display = str(a.get("displayName") or "")
            else:
                email = _normalize_email(str(a))
                display = ""
            if not email:
                continue
            out.append((email, display, 1.0))
    return out


def _gather_conversation_participants(lookback_days: int) -> list[tuple[str, str, float]]:
    """Yield (sender_id, '', age_days) for non-andrus participants in
    conversation_store. Sender IDs may be Signal phones or Discord
    user IDs; we treat them as person_ids regardless of email shape.

    Q4.2.1#5 — time-bound to ``lookback_days`` rather than scanning all
    history. Mirrors how Q4.1's tension_detector windows the same
    table; on multi-year DBs the unbounded scan was unnecessarily
    expensive and resurfaced people the operator may have forgotten."""
    try:
        from app import conversation_store
        conn = conversation_store._get_conn()
    except Exception:
        return []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    try:
        rows = conn.execute(
            """SELECT DISTINCT sender_id FROM messages
                WHERE role = 'user' AND ts >= ?""",
            (cutoff,),
        ).fetchall()
    except Exception:
        # Schema may not have `ts` on older DBs — fall back to
        # unbounded so we never crash the idle job.
        try:
            rows = conn.execute(
                """SELECT DISTINCT sender_id FROM messages
                    WHERE role = 'user'"""
            ).fetchall()
        except Exception:
            return []
    out: list[tuple[str, str, float]] = []
    for (sender_id,) in rows:
        if not sender_id:
            continue
        # Treat any sender_id as a person_id; the canonical-email
        # normalization is a no-op for non-email IDs.
        out.append((str(sender_id).lower(), "", 1.0))
    return out


# ── Compile + decay ──────────────────────────────────────────────────────


def compile_profile(lookback_days: int = 30) -> dict[str, Any]:
    """Aggregate sightings + persist. Returns summary dict.
    No-op when master switch is OFF."""
    if not _enabled():
        return {"ok": False, "skipped": True, "reason": "person_correlation_enabled=False"}

    muted = _load_mutes()
    profile = _load_profile()
    history_path = _default_history_path()

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    new_sightings = 0

    def _record(email: str, display: str, modality: str) -> None:
        nonlocal new_sightings
        if not email or email in muted:
            return
        pp = profile.setdefault(email, PersonProfile(
            person_id=email, first_seen=now_iso,
        ))
        pp.last_seen = now_iso
        if display and display not in pp.display_names:
            pp.display_names.append(display[:60])
            # Cap display_names list at 5
            if len(pp.display_names) > 5:
                pp.display_names = pp.display_names[-5:]
        pp.occurrences_per_modality[modality] = pp.occurrences_per_modality.get(modality, 0) + 1
        new_sightings += 1

    for email, display, _age in _gather_email_senders(lookback_days):
        _record(email, display, "emails")
    for email, display, _age in _gather_calendar_attendees(lookback_days):
        _record(email, display, "calendar")
    for email, display, _age in _gather_conversation_participants(lookback_days):
        _record(email, display, "convs")

    # Topic co-occurrence: read interest_profile and bump cooccurring_topics
    # for everyone sighted in this pass. Light heuristic — assume every
    # person seen in the lookback window co-occurs with the topics seen
    # in the same window.
    try:
        from app.companion.interest_model import current_profile as _ip
        ip = _ip() or {}
        recent_topics = [
            t.get("name", "") for t in (ip.get("topics") or [])
            if isinstance(t, dict)
        ][:10]
        for pp in profile.values():
            if pp.last_seen != now_iso:
                continue
            for topic in recent_topics:
                if not topic:
                    continue
                pp.cooccurring_topics[topic] = pp.cooccurring_topics.get(topic, 0) + 1
    except Exception:
        logger.debug("person_model: topic co-occurrence skipped", exc_info=True)

    # Q4.2.2#4 — when a person re-appears in this pass, boost any open
    # tension that mentions them. This composes with the cross-modal
    # detector's convergence-boost: direct sighting is a weaker but more
    # frequent signal than convergence. Best-effort, failure-isolated.
    person_tension_boosts = 0
    try:
        from app.companion.tensions import boost_freshness_for_person
        for pp in profile.values():
            if pp.last_seen != now_iso:
                continue
            try:
                person_tension_boosts += boost_freshness_for_person(
                    pp.person_id, list(pp.display_names),
                )
            except Exception:
                continue
    except Exception:
        logger.debug("person_model: tension person-boost skipped", exc_info=True)

    # Apply decay: drop people not seen in decay_months
    try:
        from app.runtime_settings import get_person_correlation_decay_months
        decay_months = get_person_correlation_decay_months()
    except Exception:
        decay_months = _DEFAULT_DECAY_MONTHS
    decay_cutoff = (now - timedelta(days=int(decay_months) * 30)).isoformat()
    decayed = 0
    for pid in list(profile.keys()):
        if profile[pid].last_seen < decay_cutoff:
            del profile[pid]
            decayed += 1

    _save_profile(profile)

    # History line
    try:
        from app.utils.jsonl_retention import append_with_cap
        append_with_cap(
            history_path,
            json.dumps({
                "ts": now_iso,
                "new_sightings": new_sightings,
                "active_people": len(profile),
                "decayed": decayed,
            }, sort_keys=True),
            max_lines=5000,
        )
    except Exception:
        logger.debug("person_model: history append failed", exc_info=True)

    # Q4.2.2#5 — GW publish OPAQUE COUNTS only (no person_ids, no
    # names). Lets SubIA observe the "operator's input universe is
    # broadening/narrowing" signal without seeing identities. Skipped
    # silently on failure.
    if new_sightings > 0 or decayed > 0:
        try:
            from app.workspace_publish import publish_to_workspace
            publish_to_workspace(
                source="person_correlation",
                content=(
                    f"{new_sightings} new person-sightings across "
                    f"{len(profile)} active; {decayed} decayed."
                ),
                salience=0.3,  # routine — never urgent
                signal_type="background",
            )
        except Exception:
            logger.debug("person_model: GW publish failed", exc_info=True)

    return {
        "ok": True,
        "new_sightings": new_sightings,
        "active_people": len(profile),
        "decayed": decayed,
        "muted_count": len(muted),
        "tension_boosts": person_tension_boosts,
    }


def current_profile() -> dict[str, Any]:
    """Read the persisted profile. Returns dict suitable for the
    REST surface. Filters muted people out — never returns muted."""
    if not _enabled():
        return {"people": [], "muted": [], "enabled": False}
    muted = _load_mutes()
    profile = _load_profile()
    people = []
    for pid, pp in profile.items():
        if pid in muted:
            continue
        people.append({
            **pp.to_dict(),
            "total_occurrences": pp.total_occurrences(),
            "modality_count": pp.modality_count(),
        })
    # Sort by last_seen (newest first) — explicitly NOT by total
    # occurrences. Ranking surface would be a Goodhart gateway.
    people.sort(key=lambda p: p.get("last_seen", ""), reverse=True)
    return {
        "people": people,
        "muted": sorted(muted),
        "enabled": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Idle-job entry ───────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    """One pass — cadence-guarded internally. Master-switch-gated."""
    if not _enabled():
        return {"ok": True, "skipped": True}
    try:
        from app.healing.handlers._common import read_state_json, write_state_json
    except Exception:
        return {"ok": False, "reason": "state helpers unavailable"}
    state = read_state_json("person_model.json", {"last_run_at": 0.0})
    now = time.time()
    if now - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return {"ok": True, "skipped": True, "reason": "cadence"}
    state["last_run_at"] = now
    try:
        summary = compile_profile()
    except Exception:
        logger.exception("person_model: compile_profile raised")
        summary = {"ok": False}
    state["last_summary"] = summary
    write_state_json("person_model.json", state)
    return summary
