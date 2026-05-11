"""Companion-level tensions store — open questions Andrus left with
the system, tracked on his behalf.

PROGRAM §41 (2026-05-11) — Q4 Item 16.

A **tension** is an open question or unresolved decision the operator
has surfaced (explicitly or implicitly) that the companion tracks
across time, accumulates relevant material against, and re-surfaces
when conditions warrant. Distinct from SubIA wonder (which is
internal contemplation gated against task completion) — tensions are
deliberately user-facing.

Lifecycle:

    OPEN (fresh)
      │
      ▼
    OPEN (decayed)        ← freshness drops below 0.5 after ~30d
      │
      ▼
    DORMANT               ← 90d untouched; no longer in active list
      │
      ▼
    RESOLVED              ← operator says "I figured this out" OR
                            cross-modal patterns answer it

Storage: ``workspace/companion/tensions/<id>.json`` — one file per
tension. Operator can grep + hand-edit; the read API tolerates
unknown fields for forward-compat.

The store is intentionally SMALL: max 30 OPEN at once. Over-detection
makes the surface useless; freshness decay + the cap force the system
to be selective about what it picks up.
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────


STATUS_OPEN = "OPEN"
STATUS_DORMANT = "DORMANT"
STATUS_RESOLVED = "RESOLVED"

_VALID_STATUSES = {STATUS_OPEN, STATUS_DORMANT, STATUS_RESOLVED}

_FRESHNESS_HALFLIFE_DAYS = 30.0
_DORMANT_AGE_DAYS = 90.0
_MAX_OPEN = 30

_lock = threading.Lock()


def _default_tensions_dir() -> Path:
    """Lazy-resolve so local dev with custom WORKSPACE_ROOT works."""
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "companion" / "tensions"
    except Exception:
        return Path("/app/workspace/companion/tensions")


# ── Data model ────────────────────────────────────────────────────────────


@dataclass
class TensionSource:
    """One piece of material accumulated against a tension."""
    kind: str               # "conversation" / "email" / "ticket" / "pattern" / "manual"
    ts: str                 # ISO-8601 UTC
    snippet: str = ""       # ≤200 chars, operator-readable
    ref: str | None = None  # opaque ref (ticket id, email id, etc.)


@dataclass
class Tension:
    """An open question/decision tracked on the operator's behalf."""
    id: str
    question: str                                 # human-readable, ≤300 chars
    created_at: str                               # ISO-8601 UTC
    last_touched_at: str                          # ISO-8601 UTC — bumps on every update
    status: str = STATUS_OPEN
    sources: list[TensionSource] = field(default_factory=list)
    workspace_id: str | None = None
    resolution: str | None = None                 # set when RESOLVED
    resolved_at: str | None = None
    detection_source: str = "manual"              # how it entered the store

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Ensure sources serialize as plain dicts (asdict already does this)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Tension":
        srcs = [
            TensionSource(**s) for s in d.get("sources", [])
            if isinstance(s, dict)
        ]
        return cls(
            id=str(d["id"]),
            question=str(d.get("question") or ""),
            created_at=str(d.get("created_at") or ""),
            last_touched_at=str(d.get("last_touched_at") or d.get("created_at") or ""),
            status=str(d.get("status") or STATUS_OPEN),
            sources=srcs,
            workspace_id=d.get("workspace_id"),
            resolution=d.get("resolution"),
            resolved_at=d.get("resolved_at"),
            detection_source=str(d.get("detection_source") or "manual"),
        )

    def freshness(self, now: datetime | None = None) -> float:
        """Exponential decay on time-since-touch. 1.0 fresh, 0.0 stale.
        ``last_touched_at`` is bumped on every update so accumulating
        material keeps a tension fresh."""
        if self.status != STATUS_OPEN:
            return 0.0
        try:
            touched = datetime.fromisoformat(
                (self.last_touched_at or self.created_at).replace("Z", "+00:00")
            )
        except ValueError:
            return 0.0
        cur = now or datetime.now(timezone.utc)
        age_days = max(0.0, (cur - touched).total_seconds() / 86400.0)
        return math.exp(-math.log(2) * age_days / _FRESHNESS_HALFLIFE_DAYS)


# ── Storage ──────────────────────────────────────────────────────────────


def _tension_path(tid: str, base: Path | None = None) -> Path:
    base = base or _default_tensions_dir()
    # Defensive: prevent path traversal via id.
    safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", tid)[:64]
    return base / f"{safe_id}.json"


def _save_tension(t: Tension, base: Path | None = None) -> None:
    path = _tension_path(t.id, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(t.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def _load_tension(tid: str, base: Path | None = None) -> Tension | None:
    path = _tension_path(tid, base)
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return Tension.from_dict(d)
    except (OSError, json.JSONDecodeError):
        logger.debug("tensions: load failed for %s", tid, exc_info=True)
        return None


def _iter_tension_files(base: Path | None = None) -> Iterable[Path]:
    base = base or _default_tensions_dir()
    if not base.exists():
        return []
    return [p for p in base.iterdir() if p.is_file() and p.suffix == ".json"]


# ── Public API ───────────────────────────────────────────────────────────


def create_tension(
    *,
    question: str,
    sources: list[TensionSource] | None = None,
    workspace_id: str | None = None,
    detection_source: str = "manual",
    base: Path | None = None,
) -> Tension | None:
    """Create a new OPEN tension. Returns the Tension on success, None
    if the OPEN cap is full OR ``question`` is too short/empty.

    The cap is the discipline that keeps the surface usable: detection
    is allowed to be eager because the cap forces selection."""
    q = (question or "").strip()
    if len(q) < 8 or len(q) > 300:
        logger.debug(
            "tensions: rejected question (len=%d outside [8, 300])", len(q),
        )
        return None
    with _lock:
        open_count = sum(
            1 for t in list_tensions(status=STATUS_OPEN, base=base)
            if t.status == STATUS_OPEN
        )
        if open_count >= _MAX_OPEN:
            logger.info(
                "tensions: OPEN cap %d reached; new tension rejected: %r",
                _MAX_OPEN, q[:60],
            )
            return None
        now = datetime.now(timezone.utc).isoformat()
        t = Tension(
            id=uuid.uuid4().hex[:12],
            question=q,
            created_at=now,
            last_touched_at=now,
            status=STATUS_OPEN,
            sources=list(sources or []),
            workspace_id=workspace_id,
            detection_source=detection_source,
        )
        _save_tension(t, base)
    logger.info("tensions: created %s (%r)", t.id, q[:60])
    return t


def update_tension(
    tid: str,
    *,
    add_sources: list[TensionSource] | None = None,
    status: str | None = None,
    resolution: str | None = None,
    bump_touched: bool = True,
    base: Path | None = None,
) -> Tension | None:
    """Mutate a tension. ``add_sources`` accumulates material;
    ``status`` transitions; ``resolution`` is recorded when going to
    RESOLVED. ``bump_touched=True`` (default) keeps OPEN tensions
    fresh; set False for housekeeping-only updates (e.g. decay sweep)."""
    with _lock:
        t = _load_tension(tid, base)
        if t is None:
            return None
        if add_sources:
            for s in add_sources:
                t.sources.append(s)
        if status is not None:
            if status not in _VALID_STATUSES:
                raise ValueError(f"invalid status {status!r}")
            t.status = status
            if status == STATUS_RESOLVED:
                t.resolved_at = datetime.now(timezone.utc).isoformat()
                if resolution:
                    t.resolution = resolution[:500]
        if bump_touched:
            t.last_touched_at = datetime.now(timezone.utc).isoformat()
        _save_tension(t, base)
    return t


def list_tensions(
    *,
    status: str | None = None,
    min_freshness: float = 0.0,
    base: Path | None = None,
) -> list[Tension]:
    """Read all tensions, optionally filtered.

    Default behavior (no args): every tension on disk.
    ``status="OPEN"``: only open. ``min_freshness=0.5``: only those
    whose freshness ≥ threshold (decayed-stale OPENs filter out).
    """
    out: list[Tension] = []
    now = datetime.now(timezone.utc)
    for path in _iter_tension_files(base):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            t = Tension.from_dict(d)
        except (OSError, json.JSONDecodeError):
            continue
        if status is not None and t.status != status:
            continue
        if min_freshness > 0 and t.freshness(now=now) < min_freshness:
            continue
        out.append(t)
    out.sort(key=lambda t: t.last_touched_at, reverse=True)
    return out


def resolve_tension(
    tid: str, resolution: str, base: Path | None = None,
) -> Tension | None:
    """Mark a tension RESOLVED with operator-provided resolution text."""
    return update_tension(
        tid, status=STATUS_RESOLVED, resolution=resolution,
        bump_touched=True, base=base,
    )


def boost_freshness_for_topic(
    topic: str, base: Path | None = None,
) -> int:
    """When a cross-modal pattern hits a topic, bump
    ``last_touched_at`` on any OPEN tension whose question references
    that topic. Returns count of tensions boosted.

    This is the Q4#15 ↔ Q4#16 cross-link: cross-modal patterns answer
    or accumulate evidence against tracked tensions."""
    if not topic or len(topic) < 3:
        return 0
    topic_lower = topic.lower()
    boosted = 0
    for t in list_tensions(status=STATUS_OPEN, base=base):
        if topic_lower in t.question.lower():
            update_tension(
                t.id,
                add_sources=[TensionSource(
                    kind="pattern",
                    ts=datetime.now(timezone.utc).isoformat(),
                    snippet=f"Cross-modal pattern hit for topic {topic!r}",
                )],
                bump_touched=True, base=base,
            )
            boosted += 1
    return boosted


def boost_freshness_for_person(
    person_id: str,
    display_names: list[str] | None = None,
    base: Path | None = None,
) -> int:
    """Q4.2.2#4 — symmetric to ``boost_freshness_for_topic`` for the
    person-correlation cross-link (PROGRAM §42).

    When a person re-appears (compile_profile bump) or shows up in a
    cross-modal pattern, any OPEN tension whose question references
    that person — by person_id OR any display name — gets its freshness
    bumped. Source kind ``person_sighting`` distinguishes from topic
    patterns.

    Match strategy: case-insensitive substring against display names
    (or the person_id if no names are known). We require ≥3 chars to
    avoid spurious matches on short tokens; this means single-letter
    or two-letter display names won't match (acceptable trade-off)."""
    if not person_id:
        return 0
    needles: list[str] = []
    for nm in (display_names or []):
        nm_l = (nm or "").strip().lower()
        if len(nm_l) >= 3:
            needles.append(nm_l)
    # Fall back to the local-part of an email if no display name fits.
    if not needles:
        local = person_id.split("@", 1)[0].strip().lower()
        if len(local) >= 3:
            needles.append(local)
    if not needles:
        return 0
    boosted = 0
    seen_ids: set[str] = set()
    for t in list_tensions(status=STATUS_OPEN, base=base):
        q_lower = t.question.lower()
        if any(n in q_lower for n in needles):
            if t.id in seen_ids:
                continue
            update_tension(
                t.id,
                add_sources=[TensionSource(
                    kind="person_sighting",
                    ts=datetime.now(timezone.utc).isoformat(),
                    snippet=f"Person re-appearance: {person_id}",
                )],
                bump_touched=True, base=base,
            )
            seen_ids.add(t.id)
            boosted += 1
    return boosted


def decay_sweep(base: Path | None = None) -> dict[str, int]:
    """Idle-job task: transition OPEN tensions ≥90d untouched to
    DORMANT. Returns counts. Run from companion.loop.

    Never DELETES tensions — operator may want the history. DORMANT
    just removes them from the active list."""
    now = datetime.now(timezone.utc)
    summary = {"checked": 0, "transitioned_to_dormant": 0}
    for t in list_tensions(status=STATUS_OPEN, base=base):
        summary["checked"] += 1
        try:
            touched = datetime.fromisoformat(
                (t.last_touched_at or t.created_at).replace("Z", "+00:00"),
            )
        except ValueError:
            continue
        age_days = (now - touched).total_seconds() / 86400.0
        if age_days >= _DORMANT_AGE_DAYS:
            update_tension(
                t.id, status=STATUS_DORMANT, bump_touched=False, base=base,
            )
            summary["transitioned_to_dormant"] += 1
    return summary


# ── Detection (light heuristics; LLM-free) ───────────────────────────────


_OPEN_QUESTION_PATTERNS = [
    re.compile(r"\bI'?m (still )?wondering\b\s+(.{8,200}?)[.?!]", re.IGNORECASE),
    re.compile(r"\b(open question|unresolved):?\s+(.{8,200}?)[.?!]", re.IGNORECASE),
    re.compile(r"\bI (haven'?t|have not) decided\b\s+(.{8,200}?)[.?!]", re.IGNORECASE),
    re.compile(r"\bnot sure whether\b\s+(.{8,200}?)[.?!]", re.IGNORECASE),
    re.compile(r"\bneed to figure out\b\s+(.{8,200}?)[.?!]", re.IGNORECASE),
]


def detect_from_text(
    text: str, source_kind: str, source_ref: str | None = None,
    base: Path | None = None,
) -> list[Tension]:
    """Scan one text blob (a conversation chunk, an email body, a
    ticket description) for explicit open-question markers. Returns
    the tensions CREATED (skipping at the OPEN cap).

    Conservative: only triggers on explicit linguistic markers, not
    on inferred uncertainty. The cap + freshness decay both serve as
    additional brakes if detection over-fires.

    ``base`` overrides the default workspace path (useful for tests
    and ephemeral per-workspace stores)."""
    if not text:
        return []
    created: list[Tension] = []
    for pat in _OPEN_QUESTION_PATTERNS:
        for m in pat.finditer(text):
            # Last group has the question content.
            q = m.groups()[-1].strip()
            if not q:
                continue
            # Coerce "?" if absent so the question reads as a question
            if not q.endswith("?"):
                q = q + "?"
            t = create_tension(
                question=q[:300],
                sources=[TensionSource(
                    kind=source_kind,
                    ts=datetime.now(timezone.utc).isoformat(),
                    snippet=text[max(0, m.start() - 40):m.end() + 40][:200],
                    ref=source_ref,
                )],
                detection_source=f"text:{source_kind}",
                base=base,
            )
            if t is not None:
                created.append(t)
    return created


# ── Idle-job entry ───────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    """One pass of the housekeeping job: decay sweep. Cheap. Safe to
    call every companion loop tick — only OPEN tensions are walked
    and the file count is capped (≤30 OPEN + DORMANT + RESOLVED)."""
    try:
        summary = decay_sweep()
    except Exception:
        logger.debug("tensions: decay_sweep raised", exc_info=True)
        return {"ok": False}
    summary["ok"] = True
    return summary
