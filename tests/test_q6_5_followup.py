"""PROGRAM §44.5 — Q6.5 second-cycle audit follow-ups.

Covers all 4 code items + 3 doc items from the second-cycle audit:

  P1#1 — Scheduler emits Signal on FAIL/ERROR drill
  P2#2 — Double-registration warning (different module)
  P2#3 — backup_freshness healing monitor
  doc#4 — Audit-log corruption recovery procedure documented
  doc#5 — Annual reflection consumption documented
  doc#6 — Q6 closure criteria documented
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _load_isolated(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────
#   P1#1 — Scheduler emits Signal on FAIL/ERROR
# ─────────────────────────────────────────────────────────────────────────


def test_scheduler_notifies_on_drill_fail(monkeypatch, tmp_path):
    """When an auto-run drill returns FAIL, scheduler fires a Signal
    notification immediately (not just counter increment)."""
    sch = _load_isolated("sch_q65", "app/resilience_drills/scheduler.py")
    # Stub master switch + drill_enabled.
    monkeypatch.setattr(sch, "master_enabled", lambda: True)
    # Force registry to be populated.
    import app.resilience_drills.drills  # noqa: F401
    from app.resilience_drills.protocol import get_registry
    reg = get_registry()
    # Snapshot prior registrations to restore after the test.
    saved_specs = list(reg._specs.values())
    saved_runners = dict(reg._runners)
    try:
        # Make every drill report past-due.
        monkeypatch.setattr(sch, "days_since_last_success", lambda name: 200.0)
        # Stub drill_enabled True for all.
        monkeypatch.setattr(sch, "drill_enabled", lambda spec: True)
        # Replace one runner with a FAIL-returning stub.
        from app.resilience_drills.protocol import DrillResult, DrillStatus, DrillSpec, DrillRisk
        fail_spec = DrillSpec(name="failing_test_drill", cadence_days=90, risk=DrillRisk.LOW)
        def fail_runner(*, dry_run=True):
            return DrillResult(
                drill_name="failing_test_drill",
                status=DrillStatus.FAIL,
                started_at="2026-05-13T10:00:00+00:00",
                completed_at="2026-05-13T10:01:00+00:00",
                duration_s=60.0, dry_run=True,
                errors=["synthesized failure for test"],
            )
        # Clear other registrations so only this fail_runner is exercised.
        reg.clear_for_tests()
        reg.register(fail_spec, fail_runner)
        # Stub the notify chain.
        captured: list[dict] = []
        monkeypatch.setattr(
            "app.notify.notify",
            lambda **kw: (captured.append(kw), True)[1],
        )
        # Run the scheduler.
        result = sch.run_once()
        # An "drill due" notification AND a "drill failed" notification both fire.
        assert result["errors"] >= 1
        titles = [c.get("title", "") for c in captured]
        assert any("Resilience drill failed" in t for t in titles), (
            f"expected drill-failed alert; got titles: {titles}"
        )
        # The failed alert body must mention the drill name + error.
        failed_alerts = [c for c in captured if "failed" in c.get("title", "").lower()]
        assert failed_alerts
        assert "failing_test_drill" in failed_alerts[0]["body"]
        assert "synthesized failure" in failed_alerts[0]["body"]
        # Topic-keyed for arbiter dedup.
        assert failed_alerts[0]["topic"] == "resilience_drill_failed:failing_test_drill"
    finally:
        # Restore the registry so subsequent tests have the real drills.
        reg.clear_for_tests()
        for spec in saved_specs:
            runner = saved_runners.get(spec.name) or (lambda **kw: None)
            reg.register(spec, runner)


def test_scheduler_no_failure_notify_on_pass(monkeypatch):
    """Routine PASS results don't trigger the failure-alert path."""
    sch = _load_isolated("sch_q65b", "app/resilience_drills/scheduler.py")
    monkeypatch.setattr(sch, "master_enabled", lambda: True)
    import app.resilience_drills.drills  # noqa: F401
    from app.resilience_drills.protocol import get_registry, DrillResult, DrillStatus, DrillSpec, DrillRisk
    reg = get_registry()
    saved_specs = list(reg._specs.values())
    saved_runners = dict(reg._runners)
    try:
        monkeypatch.setattr(sch, "days_since_last_success", lambda name: 200.0)
        monkeypatch.setattr(sch, "drill_enabled", lambda spec: True)
        pass_spec = DrillSpec(name="passing_test_drill", cadence_days=90, risk=DrillRisk.LOW)
        def pass_runner(*, dry_run=True):
            return DrillResult(
                drill_name="passing_test_drill",
                status=DrillStatus.PASS,
                started_at="x", completed_at="y", duration_s=1.0, dry_run=True,
            )
        reg.clear_for_tests()
        reg.register(pass_spec, pass_runner)
        captured: list[dict] = []
        monkeypatch.setattr(
            "app.notify.notify",
            lambda **kw: (captured.append(kw), True)[1],
        )
        result = sch.run_once()
        # "drill due" notification still fires for past-due, but NO
        # "drill failed" notification.
        titles = [c.get("title", "") for c in captured]
        assert not any("Resilience drill failed" in t for t in titles)
        assert result["errors"] == 0
    finally:
        reg.clear_for_tests()
        for spec in saved_specs:
            reg.register(spec, saved_runners.get(spec.name) or (lambda **kw: None))


# ─────────────────────────────────────────────────────────────────────────
#   P2#2 — Double-registration warning
# ─────────────────────────────────────────────────────────────────────────


def test_double_registration_warns_on_different_module(caplog):
    """Registering the same name from a DIFFERENT module + different
    runner logs a warning. Same-module re-registration (hot-reload)
    does NOT warn."""
    from app.resilience_drills.protocol import DrillRegistry, DrillSpec
    reg = DrillRegistry()
    spec = DrillSpec(name="dup_test", cadence_days=90)
    def runner_a(*, dry_run=True): return None
    runner_a.__module__ = "module_one"
    def runner_b(*, dry_run=True): return None
    runner_b.__module__ = "module_two"
    reg.register(spec, runner_a)
    with caplog.at_level(logging.WARNING, logger="app.resilience_drills.protocol"):
        reg.register(spec, runner_b)
    assert any(
        "name 'dup_test' being re-registered" in r.message
        or "name collision" in r.message
        for r in caplog.records
    )


def test_double_registration_silent_for_same_runner(caplog):
    """Same runner object → no warning (hot-reload case)."""
    from app.resilience_drills.protocol import DrillRegistry, DrillSpec
    reg = DrillRegistry()
    spec = DrillSpec(name="hr_test", cadence_days=90)
    def runner(*, dry_run=True): return None
    reg.register(spec, runner)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="app.resilience_drills.protocol"):
        reg.register(spec, runner)
    assert not any(
        "name collision" in r.message
        or "re-registered" in r.message
        for r in caplog.records
    )


# ─────────────────────────────────────────────────────────────────────────
#   P2#3 — backup_freshness healing monitor
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def bf():
    return _load_isolated(
        "bf_q65", "app/healing/monitors/backup_freshness.py",
    )


def test_backup_freshness_alerts_when_directory_missing(bf, monkeypatch, tmp_path):
    """Missing backup directory → alert."""
    monkeypatch.setattr(bf, "_default_backup_dir", lambda: tmp_path / "absent")
    try:
        import app.runtime_settings  # noqa: F401
        monkeypatch.setattr(
            "app.runtime_settings.get_backup_freshness_monitor_enabled",
            lambda: True,
        )
    except Exception:
        pass
    captured: list[dict] = []
    monkeypatch.setattr(
        "app.notify.notify",
        lambda **kw: (captured.append(kw), True)[1],
    )
    result = bf.run()
    assert result["stale"] is True
    assert result["alerts"] == 1
    assert "No local DR tarball" in captured[0]["body"]


def test_backup_freshness_alerts_when_stale(bf, monkeypatch, tmp_path):
    """Tarball older than 14d → alert."""
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    old_tarball = backup_dir / "drill_20250101T120000Z.tar.gz"
    old_tarball.write_bytes(b"fake")
    # Backdate to 30d ago.
    thirty_days_ago = time.time() - 30 * 86_400
    import os
    os.utime(old_tarball, (thirty_days_ago, thirty_days_ago))
    monkeypatch.setattr(bf, "_default_backup_dir", lambda: backup_dir)
    try:
        import app.runtime_settings  # noqa: F401
        monkeypatch.setattr(
            "app.runtime_settings.get_backup_freshness_monitor_enabled",
            lambda: True,
        )
    except Exception:
        pass
    captured: list[dict] = []
    monkeypatch.setattr(
        "app.notify.notify",
        lambda **kw: (captured.append(kw), True)[1],
    )
    result = bf.run()
    assert result["stale"] is True
    assert result["newest_tarball_age_days"] > 14
    assert result["alerts"] == 1


def test_backup_freshness_silent_when_fresh(bf, monkeypatch, tmp_path):
    """Recent tarball → no alert."""
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    fresh = backup_dir / "drill_recent.tar.gz"
    fresh.write_bytes(b"fake")  # mtime defaults to now
    monkeypatch.setattr(bf, "_default_backup_dir", lambda: backup_dir)
    try:
        import app.runtime_settings  # noqa: F401
        monkeypatch.setattr(
            "app.runtime_settings.get_backup_freshness_monitor_enabled",
            lambda: True,
        )
    except Exception:
        pass
    captured: list[dict] = []
    monkeypatch.setattr(
        "app.notify.notify",
        lambda **kw: (captured.append(kw), True)[1],
    )
    result = bf.run()
    assert result["stale"] is False
    assert result["alerts"] == 0
    assert captured == []


def test_backup_freshness_skipped_when_master_off(bf, monkeypatch):
    try:
        import app.runtime_settings  # noqa: F401
        monkeypatch.setattr(
            "app.runtime_settings.get_backup_freshness_monitor_enabled",
            lambda: False,
        )
    except Exception:
        pytest.skip("runtime_settings unavailable")
    result = bf.run()
    assert result.get("skipped") is True


def test_backup_freshness_ignores_non_tarballs(bf, monkeypatch, tmp_path):
    """Files that aren't .tar.gz / .tar are ignored (e.g. manifest.json)."""
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    # Recent JSON manifest but old tarball.
    (backup_dir / "manifest.json").write_bytes(b"{}")
    old_tar = backup_dir / "old.tar.gz"
    old_tar.write_bytes(b"x")
    import os
    thirty_days = time.time() - 30 * 86_400
    os.utime(old_tar, (thirty_days, thirty_days))
    monkeypatch.setattr(bf, "_default_backup_dir", lambda: backup_dir)
    try:
        import app.runtime_settings  # noqa: F401
        monkeypatch.setattr(
            "app.runtime_settings.get_backup_freshness_monitor_enabled",
            lambda: True,
        )
    except Exception:
        pass
    monkeypatch.setattr("app.notify.notify", lambda **kw: None)
    result = bf.run()
    # Newest TARBALL is 30d old; manifest.json is recent but ignored.
    assert result["newest_tarball_age_days"] > 14
    assert result["stale"] is True


def test_backup_freshness_monitor_registered():
    """Source-level: backup_freshness is in the healing-monitor list."""
    src = Path("app/healing/monitors/__init__.py").read_text()
    assert '"backup_freshness"' in src
    assert "from app.healing.monitors import backup_freshness" in src


# ─────────────────────────────────────────────────────────────────────────
#   Docs
# ─────────────────────────────────────────────────────────────────────────


def test_resilience_drills_doc_covers_q65_additions():
    doc = Path("docs/RESILIENCE_DRILLS.md").read_text()
    # doc#4 — audit corruption recovery.
    assert "Audit-log corruption recovery" in doc
    assert "drill_audit.jsonl" in doc
    assert ".corrupt-" in doc or "mv workspace/resilience" in doc
    # doc#5 — annual reflection consumption.
    assert "annual reflection" in doc.lower()
    assert "summarise_drift" in doc
    # doc#6 — closure criteria.
    assert "Q6 closure criteria" in doc
    assert "Q6 declared closed" in doc or "Q6 is\ndeclared CLOSED" in doc or "declared CLOSED" in doc
    # Names the re-open conditions.
    assert "Posture violation" in doc
    assert "Recovery-time excess" in doc


def test_q6_closure_criteria_names_specific_conditions():
    """Closure doc names FIVE specific re-open conditions, not vague
    'when something seems wrong'."""
    doc = Path("docs/RESILIENCE_DRILLS.md").read_text()
    section_start = doc.find("Q6 closure criteria")
    section = doc[section_start:section_start + 4000]
    # Look for the numbered list items.
    for n in range(1, 6):
        assert f"{n}." in section, f"missing condition #{n}"
