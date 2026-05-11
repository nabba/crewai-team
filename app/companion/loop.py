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
    """Idle-scheduler job tuples — appended in ``_default_jobs()``.

    Companion ideation jobs:
      - ``companion-tick``       — MEDIUM, the ideation cycle
      - ``companion-ingest``     — LIGHT,  fetches external sources daily
      - ``companion-grand-task`` — MEDIUM, 12 h workspace-grand-task synthesis
      - ``companion-xworkspace`` — LIGHT,  cross-workspace transfer proposals

    Life-companion proactive jobs (added 2026-05-09 — see
    ``app/life_companion/`` for full design):
      - ``life-companion-email``    — LIGHT, 10-min unread-inbox triage
      - ``life-companion-briefing`` — LIGHT, morning/evening/weekly digest
      - ``life-companion-routines`` — LIGHT, DOW+TOD pattern detection

    All life-companion jobs cadence-check internally and respect the
    ``LIFE_COMPANION_ENABLED`` master switch + per-feature flags.
    """
    from app.companion import cross_workspace as _xw
    from app.companion import grand_task as _grand_task
    from app.companion import ingest as _ingest
    from app.idle_scheduler import JobWeight
    jobs: list[tuple[str, Callable[[], None], str]] = [
        ("companion-tick", companion_tick, JobWeight.MEDIUM),
        *_ingest.get_idle_jobs(),
        *_grand_task.get_idle_jobs(),
        *_xw.get_idle_jobs(),
    ]
    # Life-companion jobs are best-effort: any import failure (e.g.
    # missing optional Google libraries on a slim deployment) must
    # never block the rest of the companion pipeline.
    try:
        from app.life_companion import get_idle_jobs as _lc_get_idle_jobs
        jobs.extend(_lc_get_idle_jobs())
    except Exception:
        logger.debug("companion.loop: life_companion jobs skipped", exc_info=True)

    # Phase B (2026-05-09) — closed-loop feedback router.
    try:
        from app.companion.feedback_router import run as _feedback_router_run
        jobs.append(("feedback-router", _feedback_router_run, JobWeight.LIGHT))
    except Exception:
        logger.debug("companion.loop: feedback_router job skipped", exc_info=True)

    # Phase B (2026-05-09) — user-interest model.
    try:
        from app.companion.interest_model import run as _interest_run
        jobs.append(("interest-model", _interest_run, JobWeight.LIGHT))
    except Exception:
        logger.debug("companion.loop: interest_model job skipped", exc_info=True)

    # Q4 Phase A (PROGRAM §41.1) — companion tensions housekeeping.
    # Decay sweep transitions OPEN ≥90d to DORMANT.
    try:
        from app.companion.tensions import run as _tensions_run
        jobs.append(("companion-tensions", _tensions_run, JobWeight.LIGHT))
    except Exception:
        logger.debug("companion.loop: tensions job skipped", exc_info=True)

    # Q4 Phase B (PROGRAM §41.2) — cross-modal pattern detector.
    # Reads interest_profile + control_plane.tickets; emits convergence
    # signals and boosts matching open tensions.
    try:
        from app.companion.cross_modal_patterns import run as _xmp_run
        jobs.append(("cross-modal-patterns", _xmp_run, JobWeight.LIGHT))
    except Exception:
        logger.debug(
            "companion.loop: cross_modal_patterns job skipped", exc_info=True,
        )

    # Phase C (2026-05-09) — adapter performance / paper pipeline / governance auto-propose.
    try:
        from app.training.adapter_performance import run as _ap_run
        jobs.append(("adapter-performance", _ap_run, JobWeight.LIGHT))
    except Exception:
        logger.debug("companion.loop: adapter_performance job skipped", exc_info=True)
    try:
        from app.episteme.paper_pipeline import run as _pp_run
        jobs.append(("paper-pipeline", _pp_run, JobWeight.LIGHT))
    except Exception:
        logger.debug("companion.loop: paper_pipeline job skipped", exc_info=True)
    try:
        from app.governance_ratchet.auto_propose import run as _gov_run
        jobs.append(("governance-auto-propose", _gov_run, JobWeight.LIGHT))
    except Exception:
        logger.debug("companion.loop: governance_auto_propose job skipped", exc_info=True)

    # Phase D (2026-05-09).
    try:
        from app.governance_ratchet.goodhart_enforcing_proposer import run as _ghe_run
        jobs.append(
            ("goodhart-enforcing-proposer", _ghe_run, JobWeight.LIGHT),
        )
    except Exception:
        logger.debug(
            "companion.loop: goodhart_enforcing_proposer job skipped",
            exc_info=True,
        )
    try:
        from app.companion.lessons_learned import run as _ll_run
        jobs.append(("lessons-learned", _ll_run, JobWeight.LIGHT))
    except Exception:
        logger.debug(
            "companion.loop: lessons_learned job skipped", exc_info=True,
        )

    # Identity-continuity yearly passes (§8.2 annual reflection + §8.5
    # legacy essay). Both cadence-check internally; daily fire is a no-op
    # 364 days of the year.
    try:
        from app.identity import get_idle_jobs as _identity_get_idle_jobs
        jobs.extend(_identity_get_idle_jobs())
    except Exception:
        logger.debug("companion.loop: identity jobs skipped", exc_info=True)

    # Health summary (§5.1; default-OFF behind HEALTH_INGESTION_ENABLED).
    try:
        from app.health.idle_job import get_idle_jobs as _health_get_idle_jobs
        jobs.extend(_health_get_idle_jobs())
    except Exception:
        logger.debug("companion.loop: health jobs skipped", exc_info=True)

    # Multi-modal inbox watcher (§5.4; default-OFF behind
    # INBOX_INGESTION_ENABLED).
    try:
        from app.inbox import get_idle_jobs as _inbox_get_idle_jobs
        jobs.extend(_inbox_get_idle_jobs())
    except Exception:
        logger.debug("companion.loop: inbox jobs skipped", exc_info=True)
    return jobs
