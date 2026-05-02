"""
events.py — Crew lifecycle events.

A small, typed observer bus that decouples the lifecycle envelope
(``app.crews.lifecycle``) from the concrete side-effects that fire
around a crew run (Firebase, belief state, metric, journal, auto-skill
distillation, and — once it goes live — SubIA's pre/post hooks).

Why a bus rather than direct calls?
-----------------------------------
The lifecycle manager previously called five different sinks inline:

    update_belief(...)             # ChromaDB belief-state write
    crew_started/completed(...)    # Firebase
    record_metric(...)             # benchmarks
    _journal_append(...)           # journal.jsonl
    maybe_auto_create_skill(...)   # skill distillation

Adding a sixth sink (SubIA's world-model writer, a per-crew audit log,
whatever) meant editing ``lifecycle.py``.  Testing one sink in
isolation meant importing the whole module.  That's exactly the same
"parallel paths stitched into one emitter" pattern we consolidated for
LLM observability — so the solution shape matches:

* Define a typed ``CrewEventContext`` carrying every datum any sink
  could possibly want.
* Expose ``on_crew_started`` / ``on_crew_completed`` / ``on_crew_failed``
  decorators to register handlers.
* ``fire_*`` helpers invoke all handlers under their own try/except so
  a crashing handler can't drop the rest, and log the exception with
  the handler's qualified name so failures are diagnosable.

Ordering
--------
Handlers run in **registration order**.  That's deliberate: the default
set needs a specific order (belief → metric → Firebase → journal), and
using a dict or a set would lose it.  When ``subia.hooks`` registers
its handlers later in startup it naturally fires after the defaults,
which is what we want — SubIA sees state after belief_state has been
updated.

Non-goals
---------
* Not a general-purpose pub-sub. Crew events are the only events here.
  For LLM events we delegate to CrewAI's event bus
  (``app.observability.llm_events``); crew events are ours to define.
* No async dispatch, no per-handler threading.  Handlers run
  synchronously on the caller thread (the crew's thread-pool worker).
  If a handler wants to do heavy work, it's its responsibility to
  hand that off to a daemon thread (see the auto-skill handler for
  the established pattern).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CrewEventContext:
    """All the data any handler could want about a crew run.

    The lifecycle manager builds one of these at start, mutates the
    post-execution fields (``outcome``, ``duration_s``, ``error``,
    cost/tokens) before firing the terminal event, and hands it to
    each registered handler.

    Fields are public and may be read by any handler — they're
    intentionally mutable during lifecycle's own flow, but handlers
    should treat them as read-only (mutating them makes the handler
    order-dependent in a way that's hard to reason about).
    """
    # ── Known at "started" time ──
    crew_name: str
    agent_role: str
    task_title: str
    task_description: str
    task_id: str
    mode: Optional[str] = None           # "delegated" or None
    parent_task_id: Optional[str] = None
    model: str = ""

    # ── Filled in by lifecycle before completed / failed events ──
    duration_s: float = 0.0
    outcome: str = ""
    error: Optional[BaseException] = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    cost_model: str = ""
    # Optional explicit tool-call count (crews can set this; otherwise
    # auto-skill handler heuristically estimates it).
    tool_call_count: Optional[int] = None


# ── Handler registries ──────────────────────────────────────────────

CrewEventHandler = Callable[[CrewEventContext], None]

_on_start: List[CrewEventHandler] = []
_on_complete: List[CrewEventHandler] = []
_on_fail: List[CrewEventHandler] = []


# ── Vetting-outcome registry (Week 1 audit fix for H6) ─────────────
#
# `on_crew_completed` fires before the orchestrator runs vetting, so
# handlers like `_handler_maybe_auto_skill` that need to know "did this
# crew's output pass vetting?" can't read it from CrewEventContext at
# fire time.  But the auto-skill handler runs in a daemon thread that
# does an LLM distillation call (~10-30s), and by the time it gets to
# integrate(), the orchestrator has typically finished vetting.  So we
# use a shared in-process registry: orchestrator writes the outcome
# after vetting, the handler reads it just before persisting the skill.
#
# Keyed by ``crew_name`` (not task_id) for two reasons:
#   1. The orchestrator's task_id is the commander's row, while the
#      auto-skill handler sees the per-crew lifecycle task_id.  These
#      diverge and reconciling them needs orchestrator plumbing we're
#      deferring to Week 2.
#   2. The gateway typically processes one user request at a time, so
#      "last vetting outcome for this crew_name within the TTL window"
#      is a reliable proxy.
#
# Conservative default — if the outcome is unknown or stale when the
# handler checks, treat as "not passed" and drop the draft.  This
# stops failure-shaped outputs from polluting the experiential KB
# even in race-condition cases.
#
# 5-minute TTL — vetting always completes within seconds; an entry
# older than 5 min is from a different dispatch and shouldn't gate
# anything.

_VETTING_OUTCOMES_CAP = 64
_VETTING_OUTCOMES_TTL_S = 300
_vetting_outcomes: "OrderedDict[str, tuple[bool, float]]" = None  # type: ignore[assignment]


def _vetting_dict() -> "OrderedDict[str, tuple[bool, float]]":
    """Lazy-init the registry to keep import cost low."""
    global _vetting_outcomes
    if _vetting_outcomes is None:
        from collections import OrderedDict
        _vetting_outcomes = OrderedDict()
    return _vetting_outcomes


def set_vetting_outcome(crew_name: str, passed: bool) -> None:
    """Record the vetting verdict for *crew_name*.  Called by the
    orchestrator immediately after vetting completes (pass or fail).

    Idempotent on re-call — last write wins, which is what we want
    for retries and reroutes (the most recent verdict is the true one).
    """
    if not crew_name:
        return
    import time
    d = _vetting_dict()
    d[crew_name] = (bool(passed), time.monotonic())
    d.move_to_end(crew_name)
    while len(d) > _VETTING_OUTCOMES_CAP:
        d.popitem(last=False)


def get_vetting_outcome(crew_name: str) -> bool | None:
    """Read the most recent vetting verdict for *crew_name*.  Returns
    None when unknown (crew not yet vetted in this session, or entry
    evicted by TTL/cap).

    Callers that gate on this should treat None as "not passed" —
    refusing to act on uncertainty is the conservative behaviour for
    persistence sinks like auto-skill creation.
    """
    if not crew_name:
        return None
    import time
    d = _vetting_dict()
    entry = d.get(crew_name)
    if entry is None:
        return None
    passed, recorded_at = entry
    if time.monotonic() - recorded_at > _VETTING_OUTCOMES_TTL_S:
        d.pop(crew_name, None)
        return None
    return passed


def on_crew_started(fn: CrewEventHandler) -> CrewEventHandler:
    """Decorator: register ``fn`` to run when a crew starts."""
    _on_start.append(fn)
    logger.info(
        "crews.events: registered start handler '%s' (total=%d)",
        _handler_name(fn), len(_on_start),
    )
    return fn


def on_crew_completed(fn: CrewEventHandler) -> CrewEventHandler:
    """Decorator: register ``fn`` to run when a crew completes
    successfully."""
    _on_complete.append(fn)
    logger.info(
        "crews.events: registered complete handler '%s' (total=%d)",
        _handler_name(fn), len(_on_complete),
    )
    return fn


def on_crew_failed(fn: CrewEventHandler) -> CrewEventHandler:
    """Decorator: register ``fn`` to run when a crew fails."""
    _on_fail.append(fn)
    logger.info(
        "crews.events: registered fail handler '%s' (total=%d)",
        _handler_name(fn), len(_on_fail),
    )
    return fn


def _handler_name(fn: CrewEventHandler) -> str:
    return getattr(fn, "__qualname__", None) or getattr(fn, "__name__", repr(fn))


# ── Fire helpers (called only by lifecycle.py) ─────────────────────


def fire_crew_started(ctx: CrewEventContext) -> None:
    """Run every registered start handler under isolated try/except."""
    _fire(_on_start, ctx, "start")


def fire_crew_completed(ctx: CrewEventContext) -> None:
    """Run every registered completion handler under isolated try/except."""
    _fire(_on_complete, ctx, "complete")


def fire_crew_failed(ctx: CrewEventContext) -> None:
    """Run every registered failure handler under isolated try/except."""
    _fire(_on_fail, ctx, "fail")


def _fire(
    handlers: List[CrewEventHandler],
    ctx: CrewEventContext,
    label: str,
) -> None:
    for h in handlers:
        try:
            h(ctx)
        except Exception:
            logger.debug(
                "crews.events: %s handler '%s' raised (crew=%s, task_id=%s)",
                label, _handler_name(h), ctx.crew_name, ctx.task_id,
                exc_info=True,
            )


# ── Introspection helpers ───────────────────────────────────────────


def registered_handlers() -> dict[str, list[str]]:
    """Observability: the names of every handler on each event, in
    execution order.  Handy for startup diagnostics and tests."""
    return {
        "on_crew_started":   [_handler_name(h) for h in _on_start],
        "on_crew_completed": [_handler_name(h) for h in _on_complete],
        "on_crew_failed":    [_handler_name(h) for h in _on_fail],
    }
