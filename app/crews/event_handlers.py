"""
event_handlers.py — Default subscribers for crew lifecycle events.

These are the sinks that used to be inlined in ``lifecycle.py``'s
context-manager body.  Each is now an independent handler keyed to
the event shape in ``crews.events``.  Adding a new sink is a single
``register_defaults()``-adjacent line — no more editing the envelope.

Ordering
--------
Handlers are registered in the same order the old inline code fired:

    on_crew_started:
        1. update_belief("working")
        2. crew_started (Firebase)

    on_crew_completed:
        1. update_belief("completed")   — retrospective reads this first
        2. record_metric("task_completion_time")
        3. crew_completed (Firebase)
        4. journal_append(success)
        5. maybe auto-skill distillation (threshold-gated, daemon thread)

    on_crew_failed:
        1. update_belief("failed")
        2. crew_failed (Firebase)
        3. journal_append(failed)

When SubIA activates, its handlers register LAST and thus fire after
the defaults — it sees the state after belief_state has been written.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.benchmarks import record_metric
from app.crews import events as crew_events
from app.crews.events import CrewEventContext
from app.firebase_reporter import crew_completed, crew_failed, crew_started
from app.memory.belief_state import update_belief

logger = logging.getLogger(__name__)


_JOURNAL_PATH = Path("/app/workspace/journal.jsonl")


# ── on_crew_started handlers ────────────────────────────────────────


def _handler_belief_working(ctx: CrewEventContext) -> None:
    """Belief-state write — first sink on start so downstream readers
    (retrospective_crew especially) observe the "working" state if
    they poll during the run."""
    update_belief(ctx.agent_role, "working", current_task=ctx.task_title[:100])


def _handler_firebase_started(ctx: CrewEventContext) -> None:
    """Firebase 'started' record.  Populates ``ctx.task_id`` for the
    lifecycle manager so later events can reference the same task —
    ``crew_started`` is the only sink that mints the id.
    """
    ctx.task_id = crew_started(
        ctx.crew_name,
        ctx.task_title,
        eta_seconds=_estimate_eta(ctx.crew_name),
        parent_task_id=ctx.parent_task_id,
        model=ctx.model,
    )


# ── on_crew_completed handlers ─────────────────────────────────────


def _handler_belief_completed(ctx: CrewEventContext) -> None:
    update_belief(ctx.agent_role, "completed",
                  current_task=ctx.task_title[:100])


def _handler_record_metric(ctx: CrewEventContext) -> None:
    labels = {"crew": ctx.crew_name}
    if ctx.mode:
        labels["mode"] = ctx.mode
    record_metric("task_completion_time", ctx.duration_s, labels)


def _handler_firebase_completed(ctx: CrewEventContext) -> None:
    crew_completed(
        ctx.crew_name, ctx.task_id, ctx.outcome[:2000],
        tokens_used=ctx.tokens_used,
        model=ctx.cost_model,
        cost_usd=ctx.cost_usd,
    )


def _handler_journal_success(ctx: CrewEventContext) -> None:
    _journal_append(ctx, result_label="success")


def _handler_maybe_auto_skill(ctx: CrewEventContext) -> None:
    """Threshold-gated auto-skill distillation.  Runs in a daemon
    thread so the slow LLM call to distill the skill doesn't block
    the crew from returning.

    The task_id is forwarded so the daemon thread can check the
    vetting outcome (recorded later by the orchestrator) before
    persisting the distilled skill — Phase 3 of the 2026-05-02
    audit found this handler was creating skills from FAILED
    dispatches, polluting the experiential KB.
    """
    # Lazy import: base_crew owns the excluded-crews list + the LLM
    # that does the distillation.  Import here, not at module scope,
    # to avoid a circular import (base_crew imports lifecycle which
    # would transitively import this handler registry).
    from app.crews import base_crew as _bc

    if ctx.crew_name in _bc._SKILL_EXCLUDED_CREWS:
        return
    tool_calls = (
        ctx.tool_call_count
        if ctx.tool_call_count is not None
        else _bc._estimate_tool_calls(ctx.outcome)
    )
    if tool_calls < _bc._SKILL_CREATION_THRESHOLD:
        return
    threading.Thread(
        target=_bc._auto_create_skill,
        args=(ctx.crew_name, ctx.task_description or ctx.task_title,
              ctx.outcome, tool_calls, ctx.task_id),
        daemon=True,
        name=f"skill-{ctx.crew_name}",
    ).start()


# ── on_crew_failed handlers ────────────────────────────────────────


def _handler_belief_failed(ctx: CrewEventContext) -> None:
    update_belief(ctx.agent_role, "failed",
                  current_task=ctx.task_title[:100])


def _handler_firebase_failed(ctx: CrewEventContext) -> None:
    err = str(ctx.error)[:200] if ctx.error else "unknown"
    crew_failed(ctx.crew_name, ctx.task_id, err)


def _handler_journal_failure(ctx: CrewEventContext) -> None:
    err = str(ctx.error)[:100] if ctx.error else "unknown"
    _journal_append(ctx, result_label="failed", extra={"error": err})


# ── Registration ────────────────────────────────────────────────────


_installed: bool = False


def install_defaults() -> None:
    """Register every built-in crew-event handler in the canonical
    order.  Call once at gateway startup.  Safe to call again —
    guarded against double-install so the handler lists don't stack
    duplicates.
    """
    global _installed
    if _installed:
        return
    _installed = True

    # Order matters: see module docstring.
    crew_events.on_crew_started(_handler_belief_working)
    crew_events.on_crew_started(_handler_firebase_started)

    crew_events.on_crew_completed(_handler_belief_completed)
    crew_events.on_crew_completed(_handler_record_metric)
    crew_events.on_crew_completed(_handler_firebase_completed)
    crew_events.on_crew_completed(_handler_journal_success)
    crew_events.on_crew_completed(_handler_maybe_auto_skill)

    crew_events.on_crew_failed(_handler_belief_failed)
    crew_events.on_crew_failed(_handler_firebase_failed)
    crew_events.on_crew_failed(_handler_journal_failure)


# ── Shared helpers ──────────────────────────────────────────────────


def _estimate_eta(crew_name: str) -> int:
    """Thin wrapper so the handler doesn't have to know where the
    EMA-backed ETA store lives."""
    try:
        from app.conversation_store import estimate_eta
        return estimate_eta(crew_name)
    except Exception:
        return 60


def _journal_append(
    ctx: CrewEventContext,
    *,
    result_label: str,
    extra: dict | None = None,
) -> None:
    """Append a single JSONL row to the autobiographical journal.
    Silent on I/O failure — journal is advisory telemetry, we never
    fail a crew run because the disk is full or the file is locked.
    """
    try:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "crew": ctx.crew_name,
            "task": (ctx.task_description or ctx.task_title)[:200],
            "result": result_label,
            "duration_s": round(ctx.duration_s, 1),
        }
        if extra:
            row.update(extra)
        with open(_JOURNAL_PATH, "a") as jf:
            jf.write(json.dumps(row) + "\n")
    except Exception:
        pass
