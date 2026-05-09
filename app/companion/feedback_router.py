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
    """Pull recent rows from ``feedback.events`` joined with
    ``feedback.response_metadata`` so each event carries its
    ``msg_timestamp``. Returns [] on error.

    Phase F #8 (2026-05-09): the prior implementation ran ONE query
    per event via ``_resolve_send_ts``. With limit=200 that's 200
    round-trips per pass. Now folded into a single LEFT JOIN — N+1
    becomes 1.

    Cursor is the row UUID. We sort by ``recorded_at`` desc, ``id``
    desc to absorb in-batch insert ordering. The cursor stops scanning
    once we hit the previously-seen id.
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
                "SELECT e.id, e.sender_id, e.feedback_type, e.raw_signal, "
                "       e.original_task, e.original_response, e.crew_used, "
                "       e.recorded_at, e.target_role, rm.msg_timestamp "
                "FROM feedback.events e "
                "LEFT JOIN feedback.response_metadata rm "
                "  ON rm.response_text = e.original_response "
                " AND (rm.sender_id = e.sender_id OR rm.sender_id IS NULL) "
                "ORDER BY e.recorded_at DESC, e.id DESC "
                "LIMIT :lim"
            ), {"lim": limit}).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            row_id = str(row[0])
            if row_id == since_id:
                break
            msg_ts = row[9]
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
                # ``msg_timestamp`` joined in via the JOIN above; consumers
                # (the dispatcher) read this directly. None when no
                # response_metadata row exists for this event yet.
                "msg_timestamp": int(msg_ts) if msg_ts else None,
            })
        out.reverse()
        return out
    except Exception:
        logger.debug("feedback_router: PG read failed", exc_info=True)
        return []


def _resolve_send_ts(event: dict[str, Any]) -> Optional[int]:
    """Return the Signal-cli send timestamp pre-joined onto the event.

    Phase F #8 collapsed the per-event SQL lookup into the
    ``_fetch_new_events`` JOIN — this function now just reads the
    ``msg_timestamp`` field that the JOIN populated. Kept as a
    function (rather than inlining) so existing test stubs that
    monkeypatch ``_resolve_send_ts`` continue to work.
    """
    ts = event.get("msg_timestamp")
    return int(ts) if ts else None


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
    delivered = {"skill": False, "recipe": False, "companion": False, "job": False}

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

    # ── Topic-level downweight (Phase G #2) ─────────────────────────
    # Extract topic mentions from the 👎/👍 comment + bias topic
    # selection in interest_model. Workspace and topic weights are
    # complementary: workspace narrows WHICH workspace runs next;
    # topic narrows WHICH topic surfaces inside the cycle.
    comment = event.get("original_response") or ""
    if comment and not success:
        try:
            from app.companion.topic_weights import (
                record_negative_from_comment as _record_topic_neg,
            )
            _record_topic_neg(comment)
        except Exception:
            logger.debug(
                "feedback_router: topic-weight sink failed",
                exc_info=True,
            )
    elif comment and success:
        try:
            from app.companion.topic_weights import (
                record_positive,
                _candidate_topics_in_comment,
            )
            for topic in _candidate_topics_in_comment(comment):
                record_positive(topic)
        except Exception:
            logger.debug(
                "feedback_router: topic-positive sink failed",
                exc_info=True,
            )

    # ── Background-job feedback log (Phase F #3) ────────────────────
    # When a completion-ping carries ``job_id`` metadata (set via
    # ``notify_on_complete(metadata={"job_id": ...})``), record the
    # reaction so we can build a per-job satisfaction metric over time.
    # Sink is a simple JSONL — separate from companion event log
    # because the scope is system jobs, not workspace ideas.
    job_id = metadata.get("job_id")
    if job_id:
        try:
            from datetime import datetime, timezone
            from pathlib import Path
            log_path = Path("/app/workspace/companion/job_feedback.jsonl")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "job_id": job_id,
                "polarity": polarity,
                "raw_signal": event.get("raw_signal", ""),
                "event_id": event.get("id", ""),
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True))
                f.write("\n")
            delivered["job"] = True
        except Exception:
            logger.debug("feedback_router: job sink failed", exc_info=True)

    return delivered


def run() -> dict[str, Any]:
    """One drain pass. Returns a small summary dict for tests + audit."""
    summary: dict[str, Any] = {
        "ran": False, "events_seen": 0, "events_resolved": 0,
        "events_dispatched": 0, "skill_hits": 0, "recipe_hits": 0,
        "companion_hits": 0, "job_hits": 0, "error": "",
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
                if delivered.get("job"):
                    summary["job_hits"] += 1
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
