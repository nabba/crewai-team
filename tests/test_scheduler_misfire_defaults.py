"""Regression: AsyncIOScheduler must be created with a generous
``misfire_grace_time`` and coalesce=True at the job-defaults level.

Pre-fix shape (the operator-reported bug):

  pattern_learner saw 613 occurrences/week of:
    "Run time of job '...' was missed by 0:00:03.389716"
  All from ``apscheduler.executors.default`` at WARNING level (so
  they landed in errors.jsonl). The default ``misfire_grace_time``
  is 1 second; any cron job delayed by 3+ seconds under normal
  load triggers the warning.

Post-fix:
  ``AsyncIOScheduler(job_defaults={"misfire_grace_time": 60,
                                    "coalesce": True})``
  → 60 s grace absorbs routine scheduling jitter; coalesce means
  catch-up after a pause fires once, not N times.

These tests assert the configuration without booting the actual
gateway (which depends on Postgres / Firebase / etc.).
"""
from __future__ import annotations

import pytest


def test_main_scheduler_has_generous_misfire_grace() -> None:
    """The shared ``app.main.scheduler`` instance must be configured
    with misfire_grace_time ≥ 60 s and coalesce=True."""
    # Source-grep the literal so we don't have to import app.main
    # (which has heavy startup side effects).
    from pathlib import Path
    src = (
        Path(__file__).resolve().parent.parent / "app" / "main.py"
    ).read_text(encoding="utf-8")

    # The fix must appear in the AsyncIOScheduler() construction.
    assert 'AsyncIOScheduler(' in src, "scheduler must still exist"
    # Find the constructor call and surrounding lines.
    idx = src.index("AsyncIOScheduler(")
    block = src[idx:idx + 400]
    assert '"misfire_grace_time": 60' in block, (
        "AsyncIOScheduler must be constructed with "
        "job_defaults={'misfire_grace_time': 60, ...}"
    )
    assert '"coalesce": True' in block, (
        "AsyncIOScheduler must be constructed with coalesce=True "
        "in job_defaults"
    )


def test_apscheduler_defaults_actually_take_effect_when_constructed() -> None:
    """Sanity: APScheduler honors the kwargs we pass.

    Skipped when apscheduler isn't installed in the host pytest env
    (the package is available in the gateway container only).
    """
    pytest.importorskip("apscheduler")
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    sched = AsyncIOScheduler(
        job_defaults={"misfire_grace_time": 60, "coalesce": True},
    )
    # APScheduler stores the resolved defaults on _job_defaults (private
    # but stable for years).
    assert sched._job_defaults["misfire_grace_time"] == 60
    assert sched._job_defaults["coalesce"] is True
