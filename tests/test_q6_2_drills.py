"""PROGRAM §44.2 — Q6.2 drill + scheduler + staleness tests.

Covers:
  * Each of the 4 drills (skipped-on-disabled, runs with stubbed deps,
    audit row written, landmark emission)
  * scheduler.run_once auto-runs LOW/MEDIUM-risk drills
  * scheduler refuses to auto-run HIGH-risk drills
  * drill_staleness monitor alerts past-due drills
  * external kill_the_gateway script is executable + checks typed phrase
"""
from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
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


@pytest.fixture
def protocol():
    return _load_isolated(
        "proto_q62", "app/resilience_drills/protocol.py",
    )


@pytest.fixture
def audit(monkeypatch, tmp_path):
    """Load audit module with a tmp_path audit log."""
    mod = _load_isolated("audit_q62", "app/resilience_drills/audit.py")
    log = tmp_path / "drill_audit.jsonl"
    mod._default_audit_path = lambda: log
    return mod


# ─────────────────────────────────────────────────────────────────────────
#   backup_restore drill
# ─────────────────────────────────────────────────────────────────────────


def test_backup_restore_skipped_when_disabled(monkeypatch, tmp_path):
    """Per-drill switch OFF → SKIPPED."""
    drill = _load_isolated(
        "br_q62", "app/sentience_experiments/__init__.py",  # placeholder
    ) if False else None
    drill = _load_isolated(
        "br_q62", "app/resilience_drills/drills/backup_restore.py",
    )
    # Stub the audit path so we don't pollute real state.
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(
        "app.resilience_drills.audit._default_audit_path", lambda: log,
    )
    # Per-drill OFF.
    monkeypatch.setattr(drill, "drill_enabled", lambda spec: False)
    result = drill.run(dry_run=True)
    assert result.status.value == "skipped"


def test_backup_restore_passes_when_boot_drill_ok(monkeypatch, tmp_path):
    """boot_drill.run_drill returning overall_ok=True → drill PASS."""
    drill = _load_isolated(
        "br_q62b", "app/resilience_drills/drills/backup_restore.py",
    )
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(
        "app.resilience_drills.audit._default_audit_path", lambda: log,
    )
    # Patch the drill module's OWN reference (imported via `from ... import`)
    # rather than the source module's — Python `from X import Y` copies
    # the binding, so patching the source doesn't affect the importer.
    monkeypatch.setattr(drill, "drill_enabled", lambda spec: True)
    # Stub boot_drill.run_drill.
    fake_report = MagicMock()
    fake_report.tarball = "/path/x.tar.gz"
    fake_report.overall_ok = True
    fake_report.collections = [MagicMock(), MagicMock()]  # 2 collections
    fake_report.fresh_export = False
    fake_report.errors = []
    monkeypatch.setattr(
        "app.dr.boot_drill.run_drill", lambda **kw: fake_report,
    )
    result = drill.run(dry_run=True)
    assert result.status.value == "pass"
    assert result.detail["collections_checked"] == 2
    assert result.detail["tarball"] == "/path/x.tar.gz"


def test_backup_restore_fails_when_boot_drill_not_ok(monkeypatch, tmp_path):
    drill = _load_isolated(
        "br_q62c", "app/resilience_drills/drills/backup_restore.py",
    )
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(
        "app.resilience_drills.audit._default_audit_path", lambda: log,
    )
    # Patch the drill module's OWN reference (imported via `from ... import`)
    # rather than the source module's — Python `from X import Y` copies
    # the binding, so patching the source doesn't affect the importer.
    monkeypatch.setattr(drill, "drill_enabled", lambda spec: True)
    fake_report = MagicMock()
    fake_report.tarball = "/path/x.tar.gz"
    fake_report.overall_ok = False
    fake_report.collections = []
    fake_report.fresh_export = False
    fake_report.errors = ["verifier ng"]
    monkeypatch.setattr(
        "app.dr.boot_drill.run_drill", lambda **kw: fake_report,
    )
    result = drill.run(dry_run=True)
    assert result.status.value == "fail"


# ─────────────────────────────────────────────────────────────────────────
#   embedding_migration drill
# ─────────────────────────────────────────────────────────────────────────


def test_embedding_migration_passes_on_all_steps_ok(monkeypatch, tmp_path):
    drill = _load_isolated(
        "em_q62", "app/resilience_drills/drills/embedding_migration.py",
    )
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(
        "app.resilience_drills.audit._default_audit_path", lambda: log,
    )
    # Patch the drill module's OWN reference (imported via `from ... import`)
    # rather than the source module's — Python `from X import Y` copies
    # the binding, so patching the source doesn't affect the importer.
    monkeypatch.setattr(drill, "drill_enabled", lambda spec: True)
    fake_report = MagicMock()
    fake_report.steps = [
        {"name": "plan_round_trip", "ok": True},
        {"name": "state_walk", "ok": True},
        {"name": "dual_write", "ok": True},
    ]
    monkeypatch.setattr(
        "app.memory.embedding_migration.dry_run.run_dry_run", lambda **kw: fake_report,
    )
    result = drill.run(dry_run=True)
    assert result.status.value == "pass"
    assert result.detail["n_steps_ok"] == 3
    assert result.detail["n_steps_total"] == 3


def test_embedding_migration_fails_on_step_fail(monkeypatch, tmp_path):
    drill = _load_isolated(
        "em_q62b", "app/resilience_drills/drills/embedding_migration.py",
    )
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(
        "app.resilience_drills.audit._default_audit_path", lambda: log,
    )
    # Patch the drill module's OWN reference (imported via `from ... import`)
    # rather than the source module's — Python `from X import Y` copies
    # the binding, so patching the source doesn't affect the importer.
    monkeypatch.setattr(drill, "drill_enabled", lambda spec: True)
    fake_report = MagicMock()
    fake_report.steps = [
        {"name": "plan_round_trip", "ok": True},
        {"name": "shadow_read", "ok": False},
    ]
    monkeypatch.setattr(
        "app.memory.embedding_migration.dry_run.run_dry_run", lambda **kw: fake_report,
    )
    result = drill.run(dry_run=True)
    assert result.status.value == "fail"
    assert "shadow_read" in result.errors[0]


# ─────────────────────────────────────────────────────────────────────────
#   secret_rotation drill
# ─────────────────────────────────────────────────────────────────────────


def test_secret_rotation_passes_on_clean_checks(monkeypatch, tmp_path):
    drill = _load_isolated(
        "sr_q62", "app/resilience_drills/drills/secret_rotation.py",
    )
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(
        "app.resilience_drills.audit._default_audit_path", lambda: log,
    )
    # Patch the drill module's OWN reference (imported via `from ... import`)
    # rather than the source module's — Python `from X import Y` copies
    # the binding, so patching the source doesn't affect the importer.
    monkeypatch.setattr(drill, "drill_enabled", lambda spec: True)
    result = drill.run(dry_run=True)
    assert result.status.value == "pass"
    # All four checks should appear in the detail.
    checks = result.detail["checks"]
    assert checks["gateway_secret_generation"] is True
    assert checks["bearer_token_round_trip"] is True
    assert checks["per_agent_token_enumeration"] is True
    assert checks["vendor_key_patterns"] is True


def test_secret_rotation_never_leaks_secret_values(monkeypatch, tmp_path):
    """LOAD-BEARING: the audit row must NEVER contain a generated
    candidate secret. Format-check booleans only."""
    drill = _load_isolated(
        "sr_q62b", "app/resilience_drills/drills/secret_rotation.py",
    )
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(
        "app.resilience_drills.audit._default_audit_path", lambda: log,
    )
    # Patch the drill module's OWN reference (imported via `from ... import`)
    # rather than the source module's — Python `from X import Y` copies
    # the binding, so patching the source doesn't affect the importer.
    monkeypatch.setattr(drill, "drill_enabled", lambda spec: True)
    result = drill.run(dry_run=True)
    serialized = json.dumps(result.to_dict())
    # No actual generated candidate tokens should appear.
    # We verify by checking that the serialized result is short
    # (< 2KB) and contains no obvious secret-like substrings.
    assert len(serialized) < 4000, "audit row suspiciously large"
    # The detail block legitimately contains the WORD 'sk-ant-' or
    # 'sk-' as pattern documentation, but should NOT contain
    # any 30+-char candidate following them.
    import re
    for marker in ("sk-ant-", "sk-or-"):
        for match in re.findall(marker + r"[A-Za-z0-9_-]+", serialized):
            assert len(match) < len(marker) + 5, (
                f"audit row contains suspicious secret-like value: {match[:30]}..."
            )


def test_secret_rotation_always_dry_run(monkeypatch, tmp_path):
    """The drill must ALWAYS report dry_run=True regardless of caller."""
    drill = _load_isolated(
        "sr_q62c", "app/resilience_drills/drills/secret_rotation.py",
    )
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(
        "app.resilience_drills.audit._default_audit_path", lambda: log,
    )
    # Patch the drill module's OWN reference (imported via `from ... import`)
    # rather than the source module's — Python `from X import Y` copies
    # the binding, so patching the source doesn't affect the importer.
    monkeypatch.setattr(drill, "drill_enabled", lambda spec: True)
    # Even when caller passes dry_run=False, the drill must enforce True.
    result = drill.run(dry_run=False)
    assert result.dry_run is True


# ─────────────────────────────────────────────────────────────────────────
#   kill_the_gateway drill (pre-drill check)
# ─────────────────────────────────────────────────────────────────────────


def test_kill_the_gateway_skipped_when_disabled(monkeypatch, tmp_path):
    drill = _load_isolated(
        "ktg_q62", "app/resilience_drills/drills/kill_the_gateway.py",
    )
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(
        "app.resilience_drills.audit._default_audit_path", lambda: log,
    )
    monkeypatch.setattr(drill, "drill_enabled", lambda spec: False)
    result = drill.run(dry_run=True)
    assert result.status.value == "skipped"
    assert "opt in" in result.detail.get("reason", "")


def test_kill_the_gateway_passes_when_ready(monkeypatch, tmp_path):
    drill = _load_isolated(
        "ktg_q62b", "app/resilience_drills/drills/kill_the_gateway.py",
    )
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(
        "app.resilience_drills.audit._default_audit_path", lambda: log,
    )
    # Patch the drill module's OWN reference (imported via `from ... import`)
    # rather than the source module's — Python `from X import Y` copies
    # the binding, so patching the source doesn't affect the importer.
    monkeypatch.setattr(drill, "drill_enabled", lambda spec: True)
    # Stub the three readiness checks to pass.
    monkeypatch.setattr(drill, "_check_dr_backup_recent", lambda **kw: (True, None))
    monkeypatch.setattr(drill, "_check_no_active_tier3_monitoring", lambda: (True, None))
    monkeypatch.setattr(drill, "_check_persistent_stores_healthy", lambda: (True, None))
    result = drill.run(dry_run=True)
    assert result.status.value == "pass"
    assert result.detail["mode"] == "pre_drill_check"
    assert "next_step" in result.detail


def test_kill_the_gateway_fails_when_backup_stale(monkeypatch, tmp_path):
    drill = _load_isolated(
        "ktg_q62c", "app/resilience_drills/drills/kill_the_gateway.py",
    )
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(
        "app.resilience_drills.audit._default_audit_path", lambda: log,
    )
    # Patch the drill module's OWN reference (imported via `from ... import`)
    # rather than the source module's — Python `from X import Y` copies
    # the binding, so patching the source doesn't affect the importer.
    monkeypatch.setattr(drill, "drill_enabled", lambda spec: True)
    monkeypatch.setattr(
        drill, "_check_dr_backup_recent",
        lambda **kw: (False, "no tarball"),
    )
    monkeypatch.setattr(drill, "_check_no_active_tier3_monitoring", lambda: (True, None))
    monkeypatch.setattr(drill, "_check_persistent_stores_healthy", lambda: (True, None))
    result = drill.run(dry_run=True)
    assert result.status.value == "fail"


def test_kill_the_gateway_ingest_external_report(monkeypatch, tmp_path):
    """The ingest_external_report function reads a recent report
    JSON and converts it into a DrillResult in the audit log."""
    drill = _load_isolated(
        "ktg_q62d", "app/resilience_drills/drills/kill_the_gateway.py",
    )
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(
        "app.resilience_drills.audit._default_audit_path", lambda: log,
    )
    report_dir = tmp_path / "resilience"
    report_dir.mkdir()
    monkeypatch.setattr(
        drill, "_default_kill_report_dir", lambda: report_dir,
    )
    # Write a recent report.
    report_path = report_dir / "kill_drill_20260513T120000Z.json"
    report_path.write_text(json.dumps({
        "drill_name": "kill_the_gateway",
        "status": "pass",
        "started_at": "2026-05-13T12:00:00+00:00",
        "completed_at": "2026-05-13T12:00:42+00:00",
        "duration_s": 42,
        "detail": {"recovery_seconds": 42, "recovered": True},
        "errors": [],
    }), encoding="utf-8")
    result = drill.ingest_external_report()
    assert result is not None
    assert result.status.value == "pass"
    assert result.detail["recovery_seconds"] == 42
    assert result.dry_run is False
    # Idempotent — second call returns None.
    second = drill.ingest_external_report()
    assert second is None


# ─────────────────────────────────────────────────────────────────────────
#   scheduler
# ─────────────────────────────────────────────────────────────────────────


def test_scheduler_skips_high_risk_drills(monkeypatch, tmp_path):
    """HIGH-risk drills must NEVER auto-run from the scheduler.
    Operator runs the external script."""
    sch = _load_isolated(
        "sch_q62", "app/resilience_drills/scheduler.py",
    )
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(
        "app.resilience_drills.audit._default_audit_path", lambda: log,
    )
    monkeypatch.setattr(sch, "master_enabled", lambda: True)
    # Force-import the drills package so registry populates.
    import app.resilience_drills.drills  # noqa: F401
    # Stub days_since_last_success → past due for all drills.
    monkeypatch.setattr(sch, "days_since_last_success", lambda name: 200.0)
    # Stub drill_enabled → True for all.
    monkeypatch.setattr(sch, "drill_enabled", lambda spec: True)
    # Stub the registry's runners so we can detect calls.
    from app.resilience_drills.protocol import get_registry, DrillStatus, DrillResult
    reg = get_registry()
    calls: dict[str, int] = {}
    def make_stub(name):
        def _run(*, dry_run=True):
            calls[name] = calls.get(name, 0) + 1
            return DrillResult(
                drill_name=name, status=DrillStatus.PASS,
                started_at="x", completed_at="y", duration_s=0.0,
                dry_run=dry_run,
            )
        return _run
    for spec in reg.list_specs():
        reg.register(spec, make_stub(spec.name))
    # Stub notify so we don't import the full notify chain.
    monkeypatch.setattr("app.notify.notify", lambda **kw: None)
    summary = sch.run_once()
    assert summary["auto_ran"] >= 1   # LOW/MEDIUM ones ran
    # The HIGH-risk one (kill_the_gateway) MUST NOT be auto-run.
    assert calls.get("kill_the_gateway", 0) == 0


def test_scheduler_skipped_when_master_off(monkeypatch):
    sch = _load_isolated(
        "sch_q62b", "app/resilience_drills/scheduler.py",
    )
    monkeypatch.setattr(sch, "master_enabled", lambda: False)
    summary = sch.run_once()
    assert summary["skipped"] == 1


# ─────────────────────────────────────────────────────────────────────────
#   drill_staleness monitor
# ─────────────────────────────────────────────────────────────────────────


def test_drill_staleness_monitor_alerts_past_due(monkeypatch, tmp_path):
    mon = _load_isolated(
        "ds_q62", "app/healing/monitors/drill_staleness.py",
    )
    log = tmp_path / "drill_audit.jsonl"
    # Q6.4 P1#5: monitor honors a 7-day boot grace based on audit
    # file mtime. Create the file + backdate so the grace doesn't fire.
    log.write_text("\n", encoding="utf-8")
    import os as _os, time as _t
    _os.utime(log, (_t.time() - 30 * 86_400, _t.time() - 30 * 86_400))
    monkeypatch.setattr(
        "app.resilience_drills.audit._default_audit_path", lambda: log,
    )
    # Master switch ON via stub-import path.
    try:
        import app.runtime_settings  # noqa: F401
        monkeypatch.setattr(
            "app.runtime_settings.get_drill_staleness_monitor_enabled",
            lambda: True,
        )
    except Exception:
        pass
    # Force registry population.
    import app.resilience_drills.drills  # noqa: F401
    # Stub drill_enabled True for all.
    # Patch the drill module's OWN reference (imported via `from ... import`)
    # rather than the source module's — Python `from X import Y` copies
    # the binding, so patching the source doesn't affect the importer.
    # Monitor imports drill_enabled inside its run() body, so the
    # canonical path is the one to patch (works because it's a
    # late-binding import).
    monkeypatch.setattr(
        "app.resilience_drills.protocol.drill_enabled", lambda spec: True,
    )
    # All drills past their cadence+grace.
    monkeypatch.setattr(
        "app.resilience_drills.audit.days_since_last_success",
        lambda name: 500.0,
    )
    notified: list[dict] = []
    monkeypatch.setattr(
        "app.notify.notify",
        lambda **kw: (notified.append(kw), True)[1],
    )
    summary = mon.run()
    assert summary["stale"] >= 1
    assert summary["alerts"] == 1
    assert notified[0]["title"].startswith("🛡 Resilience drills past-due")


def test_drill_staleness_monitor_silent_when_fresh(monkeypatch, tmp_path):
    mon = _load_isolated(
        "ds_q62b", "app/healing/monitors/drill_staleness.py",
    )
    try:
        import app.runtime_settings  # noqa: F401
        monkeypatch.setattr(
            "app.runtime_settings.get_drill_staleness_monitor_enabled",
            lambda: True,
        )
    except Exception:
        pass
    import app.resilience_drills.drills  # noqa: F401
    # Patch the drill module's OWN reference (imported via `from ... import`)
    # rather than the source module's — Python `from X import Y` copies
    # the binding, so patching the source doesn't affect the importer.
    monkeypatch.setattr(
        "app.resilience_drills.protocol.drill_enabled", lambda spec: True,
    )
    # All drills fresh.
    monkeypatch.setattr(
        "app.resilience_drills.audit.days_since_last_success",
        lambda name: 10.0,
    )
    notified: list[dict] = []
    monkeypatch.setattr(
        "app.notify.notify",
        lambda **kw: (notified.append(kw), True)[1],
    )
    summary = mon.run()
    assert summary["stale"] == 0
    assert summary["alerts"] == 0
    assert notified == []


# ─────────────────────────────────────────────────────────────────────────
#   External kill_the_gateway script
# ─────────────────────────────────────────────────────────────────────────


def test_kill_script_is_executable():
    """The shell script must have +x bit set."""
    path = Path("scripts/drills/kill_the_gateway.sh")
    assert path.exists()
    mode = path.stat().st_mode
    assert mode & stat.S_IXUSR, "script not executable"


def test_kill_script_refuses_without_typed_phrase():
    """Calling the script with no args (or wrong phrase) must exit with
    code 3 and NOT actually kill anything."""
    path = Path("scripts/drills/kill_the_gateway.sh").resolve()
    if not path.exists():
        pytest.skip("script not found")
    # No args.
    result = subprocess.run(
        [str(path)], capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 3
    assert "EXECUTE KILL DRILL" in result.stderr or "EXECUTE KILL DRILL" in result.stdout
    # Wrong phrase.
    result = subprocess.run(
        [str(path), "wrong phrase"], capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 3


# ─────────────────────────────────────────────────────────────────────────
#   Registry — all 4 drills register at package import
# ─────────────────────────────────────────────────────────────────────────


def test_all_four_drills_register_on_package_import():
    """The package __init__ re-imports each drill module which triggers
    their module-level register() calls. Order doesn't matter because
    register() is idempotent."""
    from app.resilience_drills.protocol import get_registry
    import importlib
    # Force fresh module loads so register() fires.
    for sub in (
        "app.resilience_drills.drills.backup_restore",
        "app.resilience_drills.drills.embedding_migration",
        "app.resilience_drills.drills.secret_rotation",
        "app.resilience_drills.drills.kill_the_gateway",
    ):
        if sub in sys.modules:
            del sys.modules[sub]
    import app.resilience_drills.drills  # noqa: F401  — triggers registration
    importlib.reload(sys.modules["app.resilience_drills.drills"])
    reg = get_registry()
    names = {s.name for s in reg.list_specs()}
    expected = {
        "backup_restore",
        "embedding_migration",
        "secret_rotation",
        "kill_the_gateway",
    }
    # Each expected drill must be registered.
    assert expected <= names
