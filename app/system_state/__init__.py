"""system_state — read-side service for deployment context.

Phase 5.1 of the post-PIM-incident program (see docs/SYSTEM_STATE.md).
This package answers questions like:

  * What's the current git state of the source repo? (host-side)
  * How long has the gateway been running?
  * What crews ran recently and how did they fare?
  * How many tools are registered? Which files are TIER_IMMUTABLE?

Phases 5.2 (Commander routing fix) and 5.3 (change-request system)
both consume this. Two consumers ground their decisions in factual
state instead of LLM inference from conversation history:

  * Commander routing — auto-injected into the routing prompt for
    "fix" / "broken" / file-mentioning queries.
  * `request_restricted_write` (Phase 5.3) — uses the git state to
    write the auto-PR's branch off the right base SHA.

Public API::

    from app.system_state import get_system_state, record_crew_run

    state = get_system_state(window_hours=24)
    # → dict with git / gateway / recent_crew_runs / tier_immutable_count / tools_registered

    record_crew_run("pim", ok=True, error=None)
    # called by base_crew on every crew completion
"""
from app.system_state.crew_runs import record_crew_run, recent_runs
from app.system_state.state import get_system_state

__all__ = [
    "get_system_state",
    "record_crew_run",
    "recent_runs",
]
