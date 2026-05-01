"""Companion tick — picks one eligible workspace and runs one cycle.

Registered with ``app.idle_scheduler`` as a MEDIUM-weight job (180 s
wall-clock cap, cooperative yield). The cycle itself is in
``app.companion.cycle``; this module only orchestrates selection,
scheduler bookkeeping, and budget charging.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from app.companion import budget as _budget
from app.companion import cycle as _cycle
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
        "companion.loop: tick start — workspace=%s budget=$%.4f vruntime=%.2fs "
        "cycles_total=%d",
        cand.project_id, cand.config.daily_budget_usd,
        cand.state.vruntime_s, cand.state.cycles_total,
    )

    try:
        result = _cycle.run_cycle(cand.project_id, cand.config)
    except Exception as exc:
        logger.warning("companion.loop: cycle raised for %s: %s",
                       cand.project_id, exc)
        cycle_cost_s = time.monotonic() - started
        _scheduler.record_tick(cand.project_id, cycle_cost_s, cand.weight)
        return

    if result.cost_usd > 0:
        _budget.charge(cand.project_id, result.cost_usd)

    cycle_cost_s = time.monotonic() - started
    _scheduler.record_tick(cand.project_id, cycle_cost_s, cand.weight)

    logger.info(
        "companion.loop: tick done — workspace=%s phase1=%d phase2=%d "
        "final=%dch cost=$%.4f dur=%.1fs aborted=%s",
        cand.project_id, result.phase_1_count, result.phase_2_count,
        result.final_output_chars, result.cost_usd, result.duration_s,
        result.aborted_reason,
    )


def get_idle_jobs() -> list[tuple[str, Callable[[], None], str]]:
    """Idle-scheduler job tuples — appended in ``_default_jobs()``."""
    from app.idle_scheduler import JobWeight
    return [("companion-tick", companion_tick, JobWeight.MEDIUM)]
