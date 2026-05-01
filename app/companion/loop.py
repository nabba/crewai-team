"""Companion tick — picks one eligible workspace and runs (Phase 1: no-op).

Phase 2 will replace the no-op cycle with a real Creative MAS pass over
``WorkspaceKB`` context. Registered with ``app.idle_scheduler`` as a
MEDIUM-weight job (180 s wall-clock cap, cooperative yield).
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from app.companion import scheduler as _scheduler

logger = logging.getLogger(__name__)


def companion_tick() -> None:
    """One Companion iteration."""
    started = time.monotonic()
    try:
        cand = _scheduler.select_next()
    except Exception as exc:
        logger.warning("companion.loop: select_next failed: %s", exc)
        return

    if cand is None:
        logger.debug("companion.loop: no eligible workspace this tick")
        return

    logger.info(
        "companion.loop: tick — workspace=%s budget=$%.4f vruntime=%.2fs "
        "cycles_total=%d",
        cand.project_id, cand.config.daily_budget_usd,
        cand.state.vruntime_s, cand.state.cycles_total,
    )

    # Phase 2 replaces this no-op with the actual ideation cycle. The tick
    # is still recorded so fairness/budget mechanisms are exercised end-to-end.
    cycle_cost_s = time.monotonic() - started
    _scheduler.record_tick(cand.project_id, cycle_cost_s, cand.weight)


def get_idle_jobs() -> list[tuple[str, Callable[[], None], str]]:
    """Idle-scheduler job tuples — appended in ``_default_jobs()``."""
    from app.idle_scheduler import JobWeight
    return [("companion-tick", companion_tick, JobWeight.MEDIUM)]
