"""
salience.py — Loop 1 of the Narrative-Self track. INFRASTRUCTURE-level.

Filters the high-volume affect trace down to meaningful transitions.
Pure Python, no LLM, no governance interaction. Produces salience events
that the episode synthesizer (episodes.py) clusters into narratives.

Self-Improver permissions: read-only on this module. Modifying salience
triggers would let the system silently rewrite what counts as meaningful
about its own experience — a self-modeling integrity violation, alongside
welfare.py and consciousness_probe.py at the infrastructure level.

Triggers (any one fires an event):
    - attractor transition (different from previous step)
    - |ΔV| or |ΔA| > 0.4 within a 60s window
    - hard-envelope ≥80% near-miss (welfare bound proximity)
    - viability variable crosses out_of_band tolerance
    - novel attractor not seen in NOVELTY_WINDOW_S

Persistence: /app/workspace/affect/salience.jsonl  (append-only audit)
In-memory: a deque of unprocessed events, drained by episodes.synthesize_and_store.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.affect.schemas import AffectState, ViabilityFrame, utc_now_iso
from app.affect.welfare import HARD_ENVELOPE
from app.utils.jsonl_retention import append_with_archive_rotate, read_archive

logger = logging.getLogger(__name__)

from app.paths import AFFECT_SALIENCE as SALIENCE_FILE  # noqa: E402  workspace-aware path
_SALIENCE_LOCK = threading.Lock()
# Cap: ~50k salience events ≈ months-to-years at typical event rates.
# Older records rotate to workspace/affect/archive/<YYYY-MM>_salience.jsonl
# — preserved for narrative-self / decentered-reflection retrospection.
_SALIENCE_MAX_LINES = 50_000

NEAR_MISS_FRACTION = 0.80          # ≥80% of hard-envelope bound = near miss
DELTA_THRESHOLD = 0.40              # |ΔV| or |ΔA| above this = spike
DELTA_WINDOW_S = 60.0
NOVELTY_WINDOW_S = 24 * 3600
OOB_TOLERANCE = 0.20                # ViabilityFrame.out_of_band tolerance


@dataclass
class SalienceEvent:
    kind: str                       # transition | spike | near_miss | oob_cross | novel_attractor
    detail: str                     # short human-readable description
    valence: float = 0.0
    arousal: float = 0.0
    controllability: float = 0.5
    attractor: str = "neutral"
    prev_attractor: str | None = None
    out_of_band: list[str] = field(default_factory=list)
    severity: str = "info"          # info | warn | critical
    ts: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ── In-process state ────────────────────────────────────────────────────────

_recent_events: deque[SalienceEvent] = deque(maxlen=512)
_attractor_seen_at: dict[str, float] = {}     # attractor → unix ts last seen
_unprocessed: deque[SalienceEvent] = deque()  # drained by episodes.py
_unprocessed_lock = threading.Lock()
_last_event_ts: str | None = None


# ── Public entry ────────────────────────────────────────────────────────────


def evaluate(
    state: AffectState,
    frame: ViabilityFrame,
    prev: AffectState | None,
) -> list[SalienceEvent]:
    """Examine the current state vs the previous; return any salience events.

    Caller is responsible for invoking `record(event)` on each — that decouples
    detection from persistence so callers can choose to suppress (e.g. tests).
    """
    events: list[SalienceEvent] = []
    now_ts = state.ts or utc_now_iso()
    now_mono = _ts_to_unix(now_ts) or _now_unix()

    # 1. Attractor transition
    if prev is not None and prev.attractor != state.attractor:
        events.append(SalienceEvent(
            kind="transition",
            detail=f"{prev.attractor} → {state.attractor}",
            valence=state.valence, arousal=state.arousal,
            controllability=state.controllability,
            attractor=state.attractor, prev_attractor=prev.attractor,
            severity="info",
            ts=now_ts,
        ))

    # 2. V/A spike (within window)
    if prev is not None:
        prev_unix = _ts_to_unix(prev.ts)
        if prev_unix is not None and (now_mono - prev_unix) <= DELTA_WINDOW_S:
            dv = abs(state.valence - prev.valence)
            da = abs(state.arousal - prev.arousal)
            if dv > DELTA_THRESHOLD or da > DELTA_THRESHOLD:
                events.append(SalienceEvent(
                    kind="spike",
                    detail=f"|ΔV|={dv:.2f} |ΔA|={da:.2f} within {now_mono - prev_unix:.0f}s",
                    valence=state.valence, arousal=state.arousal,
                    controllability=state.controllability,
                    attractor=state.attractor, prev_attractor=prev.attractor,
                    severity="warn" if max(dv, da) > 0.6 else "info",
                    ts=now_ts,
                ))

    # 3. Hard-envelope near-miss (negative valence)
    threshold = HARD_ENVELOPE["negative_valence_threshold"]
    near_miss_zone = threshold * NEAR_MISS_FRACTION  # less negative than threshold
    if state.valence <= near_miss_zone and state.valence > threshold:
        events.append(SalienceEvent(
            kind="near_miss",
            detail=(
                f"valence {state.valence:.2f} approaching welfare threshold {threshold:.2f}"
            ),
            valence=state.valence, arousal=state.arousal,
            controllability=state.controllability,
            attractor=state.attractor,
            severity="warn",
            ts=now_ts,
        ))

    # 4. Out-of-band viability variables
    oob = frame.out_of_band(tolerance=OOB_TOLERANCE)
    if oob:
        events.append(SalienceEvent(
            kind="oob_cross",
            detail=f"viability out-of-band: {', '.join(oob)}",
            valence=state.valence, arousal=state.arousal,
            controllability=state.controllability,
            attractor=state.attractor,
            out_of_band=list(oob),
            severity="info",
            ts=now_ts,
        ))

    # 5. Novel attractor (not seen in the last NOVELTY_WINDOW_S)
    last_seen = _attractor_seen_at.get(state.attractor)
    if last_seen is not None and (now_mono - last_seen) > NOVELTY_WINDOW_S:
        events.append(SalienceEvent(
            kind="novel_attractor",
            detail=(
                f"attractor '{state.attractor}' not seen in "
                f"{(now_mono - last_seen) / 3600:.1f}h"
            ),
            valence=state.valence, arousal=state.arousal,
            controllability=state.controllability,
            attractor=state.attractor,
            severity="info",
            ts=now_ts,
        ))
    _attractor_seen_at[state.attractor] = now_mono

    return events


def record(event: SalienceEvent) -> None:
    """Persist event + buffer for episodes.py."""
    global _last_event_ts
    _recent_events.append(event)
    with _unprocessed_lock:
        _unprocessed.append(event)
    if event.ts:
        _last_event_ts = event.ts
    _append(event)


def emit_if_salient(
    state: AffectState,
    frame: ViabilityFrame,
    prev: AffectState | None = None,
) -> list[SalienceEvent]:
    """Convenience: evaluate + record any events. Called from core.compute_affect."""
    events = evaluate(state, frame, prev)
    for e in events:
        record(e)
    return events


def drain_unprocessed() -> list[SalienceEvent]:
    """Atomically pop all pending events for episode synthesis."""
    with _unprocessed_lock:
        items = list(_unprocessed)
        _unprocessed.clear()
    return items


def peek_unprocessed_count() -> int:
    return len(_unprocessed)


def recent(n: int = 64) -> list[SalienceEvent]:
    return list(_recent_events)[-n:]


def last_event_ts() -> str | None:
    """ISO timestamp of the most recently recorded event, or None."""
    return _last_event_ts


# ── Persistence ─────────────────────────────────────────────────────────────


def _append(event: SalienceEvent) -> None:
    """Append + rotate. Older entries persist in
    ``workspace/affect/archive/<YYYY-MM>_salience.jsonl`` rather than
    being lost — narrative-self / decentered-reflection probes can walk
    the full multi-year history."""
    try:
        line = json.dumps(event.to_dict(), default=str)
        with _SALIENCE_LOCK:
            append_with_archive_rotate(
                SALIENCE_FILE, line, max_lines=_SALIENCE_MAX_LINES,
            )
    except Exception:
        logger.debug("affect.salience: append failed", exc_info=True)


def load_recent(hours: int = 24) -> list[SalienceEvent]:
    """Read recent salience events from disk. Used by episodes/narrative on cold start.

    Q3.1 (2026-05-11) — extended to consult the archive when the live
    file's oldest entry is newer than the requested cutoff. Without this
    escalation, archive rotation silently truncates the visible window —
    defeating the purpose of preserving history. The escalation is
    early-exit-guarded: when the live file already covers the window
    (common case), the archive is never opened.
    """
    cutoff = datetime.now(timezone.utc).timestamp() - hours * 3600

    live_rows: list[SalienceEvent] = []
    live_oldest_ts: float | None = None
    if SALIENCE_FILE.exists():
        try:
            with SALIENCE_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts_unix = _ts_to_unix(row.get("ts", ""))
                    if ts_unix is None:
                        continue
                    if live_oldest_ts is None or ts_unix < live_oldest_ts:
                        live_oldest_ts = ts_unix
                    if ts_unix < cutoff:
                        continue
                    live_rows.append(SalienceEvent(**row))
        except Exception:
            logger.debug("affect.salience: live load failed", exc_info=True)

    # If the live file already starts before the cutoff, every in-window
    # entry is in live. Skip the archive — it can be many months of data.
    if live_oldest_ts is not None and live_oldest_ts <= cutoff:
        return live_rows

    # Otherwise, the window extends into rotated data. Walk archives.
    archive_rows: list[SalienceEvent] = []
    try:
        for line in read_archive(SALIENCE_FILE, include_live=False):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts_unix = _ts_to_unix(row.get("ts", ""))
            if ts_unix is None or ts_unix < cutoff:
                continue
            archive_rows.append(SalienceEvent(**row))
    except Exception:
        logger.debug("affect.salience: archive load failed", exc_info=True)

    # Archive rows are older than live rows (rotation guarantees this).
    return archive_rows + live_rows


# ── Helpers ─────────────────────────────────────────────────────────────────


def _ts_to_unix(ts: str) -> float | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _now_unix() -> float:
    return datetime.now(timezone.utc).timestamp()
