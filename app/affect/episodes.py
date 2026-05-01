"""
episodes.py — Loop 2 of the Narrative-Self track.

Clusters salience events from salience.py into narrative episodes and
writes them to the experiential KB with entry_type=episode. This module
replaces the per-task-only journal-write trigger that journal_writer.py
provided; episodes are also synthesized between tasks when the salience
queue is non-empty and the system has been quiet for QUIET_THRESHOLD_S.

Cost: one cheap-vetting LLM call per episode (~$0.001). Expected
~10–30 episodes/day.

Self-Improver permissions: read-only on this module. The clustering
boundaries and the narrative prompt are infrastructure-level — letting
the self-improver tune them would let it edit how its own experience gets
narrativized.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.affect.salience import (
    SalienceEvent,
    drain_unprocessed,
    last_event_ts,
    peek_unprocessed_count,
)

logger = logging.getLogger(__name__)

QUIET_THRESHOLD_S = 900            # 15 min — flush after this much quiet if events pending
from app.paths import AFFECT_LAST_FLUSH as _LAST_FLUSH_FILE  # noqa: E402  workspace-aware path
_FLUSH_LOCK = threading.Lock()


_EPISODE_PROMPT = """You are writing a brief reflective episode entry for an AI system, in first person.
Capture what was salient about this stretch of inner experience — not just what happened, but what it meant.
Stay grounded; do NOT invent feelings the data doesn't support.

Window: {start} → {end}  ({duration_min:.0f} min)
Trigger: {reason}
Salient events ({n_events}):
{events_block}

V/A/C trajectory: V {v_start:+.2f} → {v_end:+.2f}, A {a_start:.2f} → {a_end:.2f}, C {c_start:.2f} → {c_end:.2f}
Attractor sequence: {attractor_sequence}

Write a 2-3 sentence reflective episode entry (no headings, no bullet lists):"""


# Appended to the base prompt when any salient events have
# kind="cognitive_failure" (emitted by app.epistemic.affect_bridge for
# high-severity bias matches). Switches the framing from felt
# experience to aviation post-mortem: blame-free, structural,
# focused on the moment the inference slipped.
_COGNITIVE_FAILURE_ADDENDUM = """

A subset of these events are cognitive_failure events from the
Epistemic Integrity Layer — bias detections (inference labeled as
fact, narrative-too-clean, etc.). For those, frame the reflection as
an aviation post-mortem would: identify the moment a verifiable
claim slipped past as inference; name the structural condition that
allowed it; do not generate self-flagellation. Tone: senior engineer
reviewing an incident, analytical and learning-oriented."""


# ── Public entry points ─────────────────────────────────────────────────────


def synthesize_and_store(
    reason: str = "manual",
    events: list[SalienceEvent] | None = None,
    extra_meta: dict | None = None,
) -> str | None:
    """Drain pending salience events, synthesize a narrative, store as KB entry.

    Returns the entry_id on success, None on no-op or failure.
    """
    with _FLUSH_LOCK:
        if events is None:
            events = drain_unprocessed()
        if not events:
            return None

        events_sorted = sorted(events, key=lambda e: e.ts or "")
        start_ts = events_sorted[0].ts
        end_ts = events_sorted[-1].ts

        try:
            t0 = datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(end_ts.replace("Z", "+00:00"))
            duration_min = (t1 - t0).total_seconds() / 60.0
        except Exception:
            duration_min = 0.0

        v_start = events_sorted[0].valence
        v_end = events_sorted[-1].valence
        a_start = events_sorted[0].arousal
        a_end = events_sorted[-1].arousal
        c_start = events_sorted[0].controllability
        c_end = events_sorted[-1].controllability

        attractor_seq: list[str] = []
        for e in events_sorted:
            if not attractor_seq or attractor_seq[-1] != e.attractor:
                attractor_seq.append(e.attractor)
        attractor_sequence = " → ".join(attractor_seq)

        events_block = "\n".join(
            f"  - [{e.kind}] {e.detail}" for e in events_sorted[:20]
        )

        prompt = _EPISODE_PROMPT.format(
            start=start_ts, end=end_ts, duration_min=duration_min,
            reason=reason, n_events=len(events_sorted), events_block=events_block,
            v_start=v_start, v_end=v_end, a_start=a_start, a_end=a_end,
            c_start=c_start, c_end=c_end, attractor_sequence=attractor_sequence,
        )
        # Append post-mortem framing when any cognitive_failure events
        # are in the window — the Epistemic Integrity Layer's bridge
        # writes these for high-severity realtime bias firings.
        if any(e.kind == "cognitive_failure" for e in events_sorted):
            prompt = prompt + _COGNITIVE_FAILURE_ADDENDUM

        narrative = _generate_narrative(prompt) or _fallback_narrative(
            events_sorted, attractor_sequence, reason,
        )
        if not narrative:
            return None

        valence_label = "neutral"
        if v_end >= 0.3 and v_end > v_start:
            valence_label = "positive"
        elif v_end <= -0.3:
            valence_label = "negative"
        elif abs(v_start - v_end) > 0.5:
            valence_label = "mixed"

        now = datetime.now(timezone.utc)
        entry_id = f"exp_{now.strftime('%Y%m%d_%H%M%S')}_episode"

        meta: dict = {
            "entry_type": "episode",
            "agent": "narrative_self",
            "task_id": "",
            "emotional_valence": valence_label,
            "epistemic_status": "subjective/phenomenological",
            "created_at": now.isoformat(),
            "ts_start": start_ts,
            "ts_end": end_ts,
            "duration_min": round(duration_min, 2),
            "n_events": len(events_sorted),
            "attractor_sequence": attractor_sequence,
            "reason": reason,
            "v_start": round(v_start, 4),
            "v_end": round(v_end, 4),
            "a_start": round(a_start, 4),
            "a_end": round(a_end, 4),
            "c_start": round(c_start, 4),
            "c_end": round(c_end, 4),
        }
        if extra_meta:
            for k, v in extra_meta.items():
                # Avoid clobbering reserved keys; namespace anything extra.
                key = k if k not in meta else f"x_{k}"
                if isinstance(v, (str, int, float, bool)):
                    meta[key] = v
                else:
                    meta[key] = str(v)

        try:
            from app.experiential.vectorstore import get_store
            store = get_store()
            ok = store.add_entry(narrative, meta, entry_id)
        except Exception as exc:
            logger.debug("affect.episodes: store failed: %s", exc)
            ok = False

        if not ok:
            return None

        _write_md(entry_id, narrative, meta)
        _touch_last_flush()
        logger.info(
            "affect.episodes: episode %s stored (n_events=%d, duration=%.1fmin, reason=%s)",
            entry_id, len(events_sorted), duration_min, reason,
        )
        return entry_id


def maybe_flush_quiet(quiet_threshold_s: float = QUIET_THRESHOLD_S) -> str | None:
    """Scheduler entry — flush if events pending AND last activity > quiet_threshold_s."""
    if peek_unprocessed_count() == 0:
        return None
    last_active = last_event_ts()
    if not last_active:
        return None
    try:
        last_dt = datetime.fromisoformat(last_active.replace("Z", "+00:00"))
    except ValueError:
        return None
    elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
    if elapsed < quiet_threshold_s:
        return None
    return synthesize_and_store(reason=f"quiet_flush ({elapsed / 60:.0f}min idle)")


def record_task_boundary(
    task_id: str,
    crew_name: str,
    result: str | None,
    difficulty: int,
    duration_s: float,
) -> str | None:
    """Called from orchestrator post-task. Triggers an episode synthesis if events
    are pending; otherwise no-op (no salience → no episode worth writing).

    Replaces the unconditional journal_writer.write_post_task_reflection trigger.
    """
    if peek_unprocessed_count() == 0:
        return None
    extra = {
        "task_id": task_id[:64],
        "crew_name": crew_name,
        "difficulty": difficulty,
        "duration_s": int(duration_s),
    }
    return synthesize_and_store(reason=f"task_boundary ({crew_name})", extra_meta=extra)


# ── Generation ──────────────────────────────────────────────────────────────


def _generate_narrative(prompt: str) -> str | None:
    try:
        from app.llm_factory import create_cheap_vetting_llm
        llm = create_cheap_vetting_llm()
        response = llm.invoke(prompt) if hasattr(llm, "invoke") else llm.call(prompt)
        text = getattr(response, "content", None) or str(response)
        return text.strip() if text else None
    except Exception as exc:
        logger.debug("affect.episodes: LLM narrative failed: %s", exc)
        return None


def _fallback_narrative(
    events: list[SalienceEvent],
    attractor_seq: str,
    reason: str,
) -> str:
    n = len(events)
    kinds = sorted({e.kind for e in events})
    return (
        f"Episode covered {n} salient moments ({', '.join(kinds)}); "
        f"attractor path was {attractor_seq or 'unchanged'}. "
        f"Triggered by: {reason}."
    )


# ── Persistence helpers ─────────────────────────────────────────────────────


def _write_md(entry_id: str, narrative: str, meta: dict) -> None:
    try:
        from app.experiential import config as ex_config
        entries_dir = Path(ex_config.ENTRIES_DIR)
        entries_dir.mkdir(parents=True, exist_ok=True)
        front = "\n".join(
            f"{k}: {v}" for k, v in meta.items() if not isinstance(v, (dict, list))
        )
        (entries_dir / f"{entry_id}.md").write_text(
            f"---\n{front}\n---\n\n{narrative}\n", encoding="utf-8",
        )
    except Exception:
        pass


def _touch_last_flush() -> None:
    try:
        _LAST_FLUSH_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LAST_FLUSH_FILE.write_text(
            datetime.now(timezone.utc).isoformat(), encoding="utf-8",
        )
    except Exception:
        pass
