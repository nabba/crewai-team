"""Closed-loop feedback router (Phase B #3, 2026-05-09).

The IMMUTABLE ``app.feedback_pipeline.process_reaction`` writes one row
to ``feedback.events`` per inbound 👍 / 👎 reaction. Until this module
landed, those rows just sat there — no learning consumer drained them.

This router:

  1. Polls ``feedback.events`` for new rows (since last cursor).
  2. Looks up the source ID via ``app.companion.notify_meta.lookup`` —
     joining by Signal-cli message timestamp.
  3. Dispatches to three sinks based on what the source said:

        - ``skill_id``   → ``app.skills.registry.record_run_result``
        - ``recipe_id``  → ``app.self_improvement.meta_agent.recorder.record``
        - ``idea_id``    → ``app.companion.feedback.record``

     (All three callable; sink failure is non-fatal — the next sink
     still runs.)

  4. Persists the cursor + summary to ``workspace/companion/feedback_router_state.json``.

Master switch: ``FEEDBACK_ROUTER_ENABLED`` (defaults ON). Disabling it
freezes the loop without touching the IMMUTABLE writer.

Idle-job cadence: 10 minutes. The pipeline writes events synchronously
on reaction, so a 10-min drain lag is tolerable (and avoids a per-
event hot path).
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_STATE_PATH = Path("/app/workspace/companion/feedback_router_state.json")


def _enabled() -> bool:
    return os.getenv("FEEDBACK_ROUTER_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _read_state() -> dict[str, Any]:
    if not _STATE_PATH.exists():
        return {"last_event_id_seen": "", "last_run_at": 0.0,
                "last_dispatched_count": 0, "last_error": ""}
    try:
        return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("feedback_router: state unreadable; starting fresh", exc_info=True)
        return {"last_event_id_seen": "", "last_run_at": 0.0,
                "last_dispatched_count": 0, "last_error": ""}


def _write_state(state: dict[str, Any]) -> None:
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _STATE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(_STATE_PATH)
    except Exception:
        logger.debug("feedback_router: state write failed", exc_info=True)


def _fetch_new_events(since_id: str, limit: int = 200) -> list[dict[str, Any]]:
    """Pull recent rows from ``feedback.events``. Returns [] on error.

    Cursor is the row UUID. We sort by ``recorded_at`` desc, ``id`` desc
    to absorb in-batch insert ordering. The cursor stops scanning once
    we hit the previously-seen id.
    """
    try:
        from sqlalchemy import create_engine, text  # type: ignore[import-not-found]
        from app.config import get_settings
    except Exception:
        return []
    s = get_settings()
    db_url = getattr(s, "mem0_postgres_url", None)
    if not db_url:
        return []
    try:
        engine = create_engine(db_url, pool_size=1, max_overflow=1)
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT id, sender_id, feedback_type, raw_signal, "
                "       original_task, original_response, crew_used, "
                "       recorded_at, target_role "
                "FROM feedback.events "
                "ORDER BY recorded_at DESC, id DESC "
                "LIMIT :lim"
            ), {"lim": limit}).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            row_id = str(row[0])
            if row_id == since_id:
                break  # seen everything from here back
            out.append({
                "id": row_id,
                "sender_id": row[1],
                "feedback_type": row[2],
                "raw_signal": row[3],
                "original_task": row[4] or "",
                "original_response": row[5] or "",
                "crew_used": row[6] or "",
                "recorded_at": row[7],
                "target_role": row[8] or "",
                # The PG schema doesn't include target_timestamp directly;
                # we rely on the notify_meta sidechannel for ts→source.
                # See _resolve_send_ts below.
            })
        # Reverse so we process oldest→newest (so the cursor advances cleanly).
        out.reverse()
        return out
    except Exception:
        logger.debug("feedback_router: PG read failed", exc_info=True)
        return []


def _resolve_send_ts(event: dict[str, Any]) -> Optional[int]:
    """Find the Signal-cli send timestamp this event reacted to.

    The IMMUTABLE feedback_pipeline doesn't currently surface the
    ``target_timestamp`` directly through ``feedback.events`` — it
    looks up response metadata internally and stores ``original_task``
    / ``original_response`` strings. To get the ts, we cross-reference
    ``feedback.responses`` (the same writer's response-table) by
    response_id.

    If the response table is unavailable / schema differs, we degrade
    silently — the dispatcher will fall through to the
    ``original_response``-text-based heuristics below.
    """
    try:
        from sqlalchemy import create_engine, text  # type: ignore[import-not-found]
        from app.config import get_settings
    except Exception:
        return None
    s = get_settings()
    db_url = getattr(s, "mem0_postgres_url", None)
    if not db_url:
        return None
    try:
        engine = create_engine(db_url, pool_size=1, max_overflow=1)
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT target_timestamp FROM feedback.responses "
                "WHERE response_text = :resp "
                "ORDER BY recorded_at DESC LIMIT 1"
            ), {"resp": (event.get("original_response") or "")[:2000]}).fetchone()
        return int(row[0]) if row and row[0] else None
    except Exception:
        return None


def _polarity_for(event: dict[str, Any]) -> str:
    """Map feedback_type → "👍" / "👎"."""
    ft = event.get("feedback_type") or ""
    if ft == "explicit_positive":
        return "👍"
    if ft == "explicit_negative":
        return "👎"
    return ""


def _dispatch(event: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Fan event out to whichever sinks the metadata identifies.

    Each sink is best-effort: a single sink failure logs at DEBUG and
    leaves the others to run. Returns the per-sink delivery status.
    """
    polarity = _polarity_for(event)
    success = polarity == "👍"
    delivered = {"skill": False, "recipe": False, "companion": False}

    # ── Skill registry counter ────────────────────────────────────────
    skill_id = metadata.get("skill_id")
    if skill_id:
        try:
            from app.skills.registry import record_run_result
            record_run_result(skill_id, success=success)
            delivered["skill"] = True
        except Exception:
            logger.debug("feedback_router: skill sink failed", exc_info=True)

    # ── Meta-agent recipe-outcome ledger ─────────────────────────────
    recipe_id = metadata.get("recipe_id")
    if recipe_id:
        try:
            from app.self_improvement.meta_agent.recorder import record_outcome
            record_outcome(
                recipe_id=recipe_id,
                crew_name=metadata.get("crew_name") or event.get("crew_used") or "",
                task_id=metadata.get("task_id") or event.get("id"),
                success=success,
                user_feedback=polarity,
                confidence="high",
            )
            delivered["recipe"] = True
        except Exception:
            logger.debug("feedback_router: recipe sink failed", exc_info=True)

    # ── Companion event-log ─────────────────────────────────────────
    idea_id = metadata.get("idea_id")
    workspace_id = metadata.get("workspace_id")
    if idea_id and workspace_id:
        try:
            from app.companion.feedback import record as _record_companion, Polarity, Source
            _record_companion(
                idea_id=idea_id,
                workspace_id=workspace_id,
                polarity=Polarity.UP if success else Polarity.DOWN,
                comment=(event.get("original_response") or "")[:500],
                source=Source.SIGNAL,
            )
            delivered["companion"] = True
        except Exception:
            logger.debug("feedback_router: companion sink failed", exc_info=True)

    # ── Companion-scheduler weight downgrade (Phase D #4) ───────────
    # Bias the next-pick away from the workspace whose idea was 👎'd.
    # 👍 partially counteracts past downvotes (see feedback_weights).
    if workspace_id:
        try:
            from app.companion.feedback_weights import (
                record_negative, record_positive,
            )
            if success:
                record_positive(workspace_id)
            else:
                record_negative(workspace_id)
        except Exception:
            logger.debug(
                "feedback_router: scheduler-weight sink failed",
                exc_info=True,
            )

    return delivered


def run() -> dict[str, Any]:
    """One drain pass. Returns a small summary dict for tests + audit."""
    summary: dict[str, Any] = {
        "ran": False, "events_seen": 0, "events_resolved": 0,
        "events_dispatched": 0, "skill_hits": 0, "recipe_hits": 0,
        "companion_hits": 0, "error": "",
    }
    if not _enabled():
        return summary

    # Prune the sidechannel so it doesn't grow forever.
    try:
        from app.companion.notify_meta import prune as _prune_meta
        _prune_meta()
    except Exception:
        logger.debug("feedback_router: meta prune failed", exc_info=True)

    state = _read_state()
    since = state.get("last_event_id_seen", "")

    events = _fetch_new_events(since_id=since)
    summary["ran"] = True
    summary["events_seen"] = len(events)
    if not events:
        state["last_run_at"] = time.time()
        _write_state(state)
        return summary

    new_cursor = since
    for ev in events:
        try:
            send_ts = _resolve_send_ts(ev)
            if send_ts is None:
                continue
            from app.companion.notify_meta import lookup as _lookup_meta
            metadata = _lookup_meta(send_ts)
            if not metadata:
                continue
            summary["events_resolved"] += 1
            delivered = _dispatch(ev, metadata)
            if any(delivered.values()):
                summary["events_dispatched"] += 1
                if delivered["skill"]:
                    summary["skill_hits"] += 1
                if delivered["recipe"]:
                    summary["recipe_hits"] += 1
                if delivered["companion"]:
                    summary["companion_hits"] += 1
        except Exception:
            logger.debug("feedback_router: per-event dispatch failed", exc_info=True)
        new_cursor = ev["id"]

    state["last_event_id_seen"] = new_cursor
    state["last_run_at"] = time.time()
    state["last_dispatched_count"] = summary["events_dispatched"]
    _write_state(state)

    # Best-effort audit.
    try:
        from app.life_companion._common import audit_event
        audit_event(
            "feedback_router_pass",
            events_seen=summary["events_seen"],
            events_resolved=summary["events_resolved"],
            events_dispatched=summary["events_dispatched"],
            skill_hits=summary["skill_hits"],
            recipe_hits=summary["recipe_hits"],
            companion_hits=summary["companion_hits"],
        )
    except Exception:
        logger.debug("feedback_router: audit failed", exc_info=True)
    return summary
