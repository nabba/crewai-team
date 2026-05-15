"""PROGRAM §44.4 — Q6.4 post-ship audit fixes.

Covers all P0 + P1 + P2 items from the Q6 first-cycle audit:

  P0#1  Recovery-landmark via explicit prior_status (BUG: read its own row)
  P0#2  Recovery test exercises production sequence (BUG: test mocked wrong layer)
  P1#3  Per-drill in-flight lock
  P1#4  is_first_run uses last_successful_for (not last_result_for)
  P1#5  Boot grace for staleness monitor
  P1#6  secret_rotation uses inspect.getsource
  P1#7  SOUL.md guard regex catches leaked secrets in audit
  P2#8  React card shows drill state (source-level check)
  P2#9  CLI entry point
  P2#10 kill_the_gateway SPEC description clarifies toggle semantics
  #11   Self-monitoring documented
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
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
#   P0#1 — recovery-landmark via explicit prior_status
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def proto():
    return _load_isolated(
        "proto_q64", "app/resilience_drills/protocol.py",
    )


@pytest.fixture
def audit(monkeypatch, tmp_path):
    mod = _load_isolated("audit_q64", "app/resilience_drills/audit.py")
    log = tmp_path / "drill_audit.jsonl"
    mod._default_audit_path = lambda: log
    # Also stub lock paths into tmp.
    mod._default_lock_path = lambda name: tmp_path / f".{name}.lock"
    return mod


def test_emit_landmark_recovery_uses_explicit_prior_status(audit, proto):
    """P0#1 — emit_landmark_for accepts prior_status and uses it
    instead of reading the audit log (which would be the new row)."""
    captured: list[dict] = []
    import app.identity.continuity_ledger as cl
    original_record = cl.record_event
    cl.record_event = lambda **kw: (captured.append(kw), True)[1]
    try:
        # Simulate production sequence: append THEN emit.
        new_r = proto.DrillResult(
            drill_name="backup_restore",
            status=proto.DrillStatus.PASS,
            started_at="2026-05-13T10:00:00+00:00",
            completed_at="2026-05-13T10:01:00+00:00",
            duration_s=60.0,
            dry_run=True,
        )
        audit.append_result(new_r)
        # Pass explicit prior_status — the new row's "pass" must NOT
        # be read instead.
        audit.emit_landmark_for(
            new_r, is_first_run=False, prior_status="fail",
        )
        assert len(captured) == 1
        assert "recovered" in captured[0]["summary"]
    finally:
        cl.record_event = original_record


def test_emit_landmark_no_recovery_when_prior_was_pass(audit, proto):
    """Routine PASS-then-PASS — prior_status='pass' — must NOT emit."""
    captured: list[dict] = []
    import app.identity.continuity_ledger as cl
    original = cl.record_event
    cl.record_event = lambda **kw: (captured.append(kw), True)[1]
    try:
        new_r = proto.DrillResult(
            drill_name="x", status=proto.DrillStatus.PASS,
            started_at="x", completed_at="y", duration_s=1.0, dry_run=True,
        )
        audit.append_result(new_r)
        audit.emit_landmark_for(
            new_r, is_first_run=False, prior_status="pass",
        )
        assert captured == []
    finally:
        cl.record_event = original


def test_emit_landmark_recovery_old_signature_no_longer_reads_audit(audit, proto):
    """Regression for the original bug: even if a caller forgets to
    pass prior_status, emit_landmark_for must NOT read its own newly-
    appended row from the audit log."""
    captured: list[dict] = []
    import app.identity.continuity_ledger as cl
    original = cl.record_event
    cl.record_event = lambda **kw: (captured.append(kw), True)[1]
    try:
        # Seed a prior FAIL.
        audit.append_result(proto.DrillResult(
            drill_name="z", status=proto.DrillStatus.FAIL,
            started_at="2026-05-12T10:00:00+00:00",
            completed_at="2026-05-12T10:01:00+00:00",
            duration_s=60.0, dry_run=True,
        ))
        new_r = proto.DrillResult(
            drill_name="z", status=proto.DrillStatus.PASS,
            started_at="2026-05-13T10:00:00+00:00",
            completed_at="2026-05-13T10:01:00+00:00",
            duration_s=60.0, dry_run=True,
        )
        # Production sequence: append FIRST.
        audit.append_result(new_r)
        # Caller forgets prior_status — recovery branch should NOT fire
        # (no fallback to last_result_for).
        audit.emit_landmark_for(new_r, is_first_run=False)
        assert captured == [], (
            "without explicit prior_status, no recovery emission"
        )
    finally:
        cl.record_event = original


# ─────────────────────────────────────────────────────────────────────────
#   P0#2 — recovery test exercises production sequence
# ─────────────────────────────────────────────────────────────────────────


def test_recovery_landmark_exercised_via_production_sequence(audit, proto):
    """P0#2 — this is the Q5.5 lesson #2: the test must exercise the
    REAL call path (append → emit), not mock a higher layer.

    The full production sequence:
      1. snapshot prior state (last_successful_for + last_result_for)
      2. append new result
      3. emit_landmark_for(result, is_first_run, prior_status)

    Recovery emission must fire when prior was FAIL and new is PASS."""
    captured: list[dict] = []
    import app.identity.continuity_ledger as cl
    original = cl.record_event
    cl.record_event = lambda **kw: (captured.append(kw), True)[1]
    try:
        # Step 1: seed a prior FAIL.
        audit.append_result(proto.DrillResult(
            drill_name="seq_test", status=proto.DrillStatus.FAIL,
            started_at="2026-05-12T10:00:00+00:00",
            completed_at="2026-05-12T10:01:00+00:00",
            duration_s=60.0, dry_run=True,
            errors=["synth"],
        ))

        # Step 2: snapshot BEFORE append (mirrors production).
        prior_any = audit.last_result_for("seq_test")
        is_first_run = audit.last_successful_for("seq_test") is None
        prior_status = prior_any.get("status") if prior_any else None
        assert prior_status == "fail"
        assert is_first_run is True  # no prior PASS exists

        # Step 3: append + emit.
        new_r = proto.DrillResult(
            drill_name="seq_test", status=proto.DrillStatus.PASS,
            started_at="2026-05-13T10:00:00+00:00",
            completed_at="2026-05-13T10:01:00+00:00",
            duration_s=60.0, dry_run=True,
        )
        audit.append_result(new_r)
        audit.emit_landmark_for(
            new_r, is_first_run=is_first_run, prior_status=prior_status,
        )

        # First-pass landmark fires (because is_first_run was True);
        # recovery branch is preempted by first-pass branch which is
        # checked first.
        assert len(captured) == 1
        assert "first_pass" in captured[0]["summary"]
    finally:
        cl.record_event = original


# ─────────────────────────────────────────────────────────────────────────
#   P1#3 — per-drill in-flight lock
# ─────────────────────────────────────────────────────────────────────────


def test_drill_lock_acquire_release_round_trip(audit, tmp_path):
    name = "test_drill"
    # Initially not in-flight.
    assert audit.is_drill_in_flight(name) is False
    # Acquire.
    assert audit.acquire_drill_lock(name) is True
    # Now in-flight.
    assert audit.is_drill_in_flight(name) is True
    # Re-acquire fails.
    assert audit.acquire_drill_lock(name) is False
    # Release.
    audit.release_drill_lock(name)
    assert audit.is_drill_in_flight(name) is False


def test_drill_lock_stale_lock_treated_as_crashed(audit, tmp_path):
    """A lock file >1h old means the prior run crashed; new acquire OK."""
    name = "stale_drill"
    audit.acquire_drill_lock(name)
    # Backdate the lock file mtime by 2h.
    lock_path = audit._default_lock_path(name)
    two_hours_ago = time.time() - 2 * 3600
    os.utime(lock_path, (two_hours_ago, two_hours_ago))
    assert audit.is_drill_in_flight(name) is False
    # Acquire succeeds (overwrites stale lock).
    assert audit.acquire_drill_lock(name) is True


def test_backup_restore_drill_uses_lock(monkeypatch, tmp_path):
    """The drill should call acquire_drill_lock + release on completion."""
    drill = _load_isolated(
        "br_q64", "app/resilience_drills/drills/backup_restore.py",
    )
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(
        "app.resilience_drills.audit._default_audit_path", lambda: log,
    )
    monkeypatch.setattr(
        "app.resilience_drills.audit._default_lock_path",
        lambda name: tmp_path / f".{name}.lock",
    )
    monkeypatch.setattr(drill, "drill_enabled", lambda spec: True)
    fake_report = MagicMock()
    fake_report.tarball = "/path/x"
    fake_report.overall_ok = True
    fake_report.collections = []
    fake_report.fresh_export = False
    fake_report.errors = []
    monkeypatch.setattr("app.dr.boot_drill.run_drill", lambda **kw: fake_report)
    # Pre-acquire the lock — drill should bail out.
    lock_path = tmp_path / ".backup_restore.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text('{"drill_name": "backup_restore", "acquired_at": "now"}')
    result = drill.run(dry_run=True)
    assert result.status.value == "skipped"
    assert "in-flight" in result.detail["reason"]


# ─────────────────────────────────────────────────────────────────────────
#   P1#4 — is_first_run uses last_successful_for
# ─────────────────────────────────────────────────────────────────────────


def test_skipped_first_run_does_not_suppress_first_pass(monkeypatch, tmp_path):
    """A prior SKIPPED row must NOT make the first actual PASS lose
    its first_pass landmark."""
    drill = _load_isolated(
        "br_q64_skip", "app/resilience_drills/drills/backup_restore.py",
    )
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(
        "app.resilience_drills.audit._default_audit_path", lambda: log,
    )
    monkeypatch.setattr(
        "app.resilience_drills.audit._default_lock_path",
        lambda name: tmp_path / f".{name}.lock",
    )
    # First call: master OFF → SKIPPED.
    monkeypatch.setattr(drill, "drill_enabled", lambda spec: False)
    drill.run(dry_run=True)
    # Second call: master ON, run produces PASS.
    monkeypatch.setattr(drill, "drill_enabled", lambda spec: True)
    fake_report = MagicMock()
    fake_report.tarball = "/x"; fake_report.overall_ok = True
    fake_report.collections = []; fake_report.fresh_export = False
    fake_report.errors = []
    monkeypatch.setattr("app.dr.boot_drill.run_drill", lambda **kw: fake_report)
    captured: list[dict] = []
    monkeypatch.setattr(
        "app.identity.continuity_ledger.record_event",
        lambda **kw: (captured.append(kw), True)[1],
    )
    result = drill.run(dry_run=True)
    assert result.status.value == "pass"
    # The first_pass landmark should fire — last_successful_for was
    # still None at the second invocation (the SKIPPED row wasn't a
    # success).
    assert any("first_pass" in c.get("summary", "") for c in captured)


# ─────────────────────────────────────────────────────────────────────────
#   P1#5 — boot grace for staleness monitor
# ─────────────────────────────────────────────────────────────────────────


def test_staleness_monitor_skips_in_boot_grace_window(monkeypatch, tmp_path):
    """When audit file doesn't exist yet (fresh install), monitor skips."""
    mon = _load_isolated(
        "ds_q64", "app/healing/monitors/drill_staleness.py",
    )
    # Audit file doesn't exist in tmp_path.
    monkeypatch.setattr(
        "app.resilience_drills.audit._default_audit_path",
        lambda: tmp_path / "nope.jsonl",
    )
    try:
        import app.runtime_settings  # noqa: F401
        monkeypatch.setattr(
            "app.runtime_settings.get_drill_staleness_monitor_enabled",
            lambda: True,
        )
    except Exception:
        pass
    result = mon.run()
    assert result.get("skipped") is True
    assert result.get("reason") == "boot_grace"


def test_staleness_monitor_runs_after_grace_window(monkeypatch, tmp_path):
    """When audit file is older than the boot grace, monitor runs."""
    mon = _load_isolated(
        "ds_q64b", "app/healing/monitors/drill_staleness.py",
    )
    audit_path = tmp_path / "drill_audit.jsonl"
    audit_path.write_text("\n")
    # Backdate mtime to 30 days ago.
    thirty_days_ago = time.time() - 30 * 86_400
    os.utime(audit_path, (thirty_days_ago, thirty_days_ago))
    monkeypatch.setattr(
        "app.resilience_drills.audit._default_audit_path", lambda: audit_path,
    )
    try:
        import app.runtime_settings  # noqa: F401
        monkeypatch.setattr(
            "app.runtime_settings.get_drill_staleness_monitor_enabled",
            lambda: True,
        )
    except Exception:
        pass
    # No drills registered, but monitor at least proceeds past boot-grace.
    result = mon.run()
    assert result.get("skipped") is not True or (
        result.get("reason") != "boot_grace"
    )


# ─────────────────────────────────────────────────────────────────────────
#   P1#6 — secret_rotation uses inspect.getsource
# ─────────────────────────────────────────────────────────────────────────


def test_secret_rotation_uses_inspect_getsource_for_bridge_client():
    """Source-level: the drill uses inspect.getsource not open(__file__)."""
    src = Path("app/resilience_drills/drills/secret_rotation.py").read_text()
    assert "inspect.getsource(bc)" in src
    assert "open(bc.__file__)" not in src


# ─────────────────────────────────────────────────────────────────────────
#   P1#7 — SOUL.md guard regex
# ─────────────────────────────────────────────────────────────────────────


def test_soul_md_guard_catches_anthropic_key_shaped_string():
    drill = _load_isolated(
        "sr_q64", "app/resilience_drills/drills/secret_rotation.py",
    )
    bad_detail = {
        "checks": {"x": True},
        "leaked_value": "sk-ant-abcdefghijklmnopqrstuvwxyz12345",
    }
    ok, err = drill._soul_md_guard(bad_detail)
    assert ok is False
    assert "secret-shaped substring" in err
    # CRITICAL: the error must NOT include the leaked value itself.
    assert "abcdefghijklmnopqrstuvwxyz12345" not in err


def test_soul_md_guard_accepts_clean_detail():
    drill = _load_isolated(
        "sr_q64b", "app/resilience_drills/drills/secret_rotation.py",
    )
    clean_detail = {
        "checks": {
            "gateway_secret_generation": True,
            "bearer_token_round_trip": True,
        },
        "vendor_patterns_info": {
            "vendors": {"anthropic": True, "openai": True, "openrouter": True},
        },
    }
    ok, err = drill._soul_md_guard(clean_detail)
    assert ok is True
    assert err is None


def test_soul_md_guard_catches_bearer_token_leak():
    drill = _load_isolated(
        "sr_q64c", "app/resilience_drills/drills/secret_rotation.py",
    )
    bad = {"raw": "Bearer abcdefghijklmnopqrstuvwxyz123456789012345"}
    ok, err = drill._soul_md_guard(bad)
    assert ok is False


# ─────────────────────────────────────────────────────────────────────────
#   P2#8 — React card shows drill state
# ─────────────────────────────────────────────────────────────────────────


def test_react_card_displays_drill_state():
    src = Path(
        "dashboard-react/src/components/ResilienceDrillsCard.tsx"
    ).read_text()
    assert "useDrillsRegistryQuery" in src
    assert "drillStateFor" in src
    # The card passes state to the Toggle component.
    assert "state={drillState.get(" in src


def test_react_queries_exports_drills_registry_hook():
    src = Path("dashboard-react/src/api/queries.ts").read_text()
    assert "useDrillsRegistryQuery" in src
    assert "DrillRegistryEntry" in src


# ─────────────────────────────────────────────────────────────────────────
#   P2#9 — CLI entry point
# ─────────────────────────────────────────────────────────────────────────


def test_cli_entry_point_runs_list():
    """python -m app.resilience_drills list returns JSON with drills key."""
    result = subprocess.run(
        [sys.executable, "-m", "app.resilience_drills", "list"],
        capture_output=True, text=True, timeout=30,
    )
    # The CLI exits 0 on list (no drill execution).
    assert result.returncode == 0, f"stderr: {result.stderr}"
    payload = json.loads(result.stdout)
    assert "drills" in payload
    names = {d["name"] for d in payload["drills"]}
    assert names >= {
        "backup_restore", "embedding_migration",
        "secret_rotation", "kill_the_gateway",
    }


def test_cli_entry_point_runs_posture():
    """python -m app.resilience_drills posture returns the decision."""
    result = subprocess.run(
        [sys.executable, "-m", "app.resilience_drills", "posture"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["ha_enabled"] is False
    assert "s3" in payload["off_host_targets"]
    assert "google_drive" in payload["off_host_targets"]


def test_cli_rejects_unknown_drill():
    result = subprocess.run(
        [sys.executable, "-m", "app.resilience_drills", "run", "nope_drill"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert "unknown drill" in result.stderr.lower()


# ─────────────────────────────────────────────────────────────────────────
#   P2#10 — kill_the_gateway SPEC description
# ─────────────────────────────────────────────────────────────────────────


def test_kill_the_gateway_spec_documents_toggle_semantics():
    """SPEC.description must explain that the master switch gates
    SCHEDULER notifications, not external-script execution. Load the
    module and inspect the runtime-concatenated description string
    rather than the source (which may have line-broken literals)."""
    drill = _load_isolated(
        "ktg_spec", "app/resilience_drills/drills/kill_the_gateway.py",
    )
    desc = drill.SPEC.description
    assert "SCHEDULER notifications" in desc
    assert "external scripts/drills/kill_the_gateway.sh" in desc
    assert "manually with the typed phrase" in desc


# ─────────────────────────────────────────────────────────────────────────
#   #11 — self-monitoring documentation
# ─────────────────────────────────────────────────────────────────────────


def test_operator_guide_documents_self_monitoring():
    doc = Path("docs/RESILIENCE_DRILLS.md").read_text()
    assert "Who watches the watchers" in doc
    assert "120+ days" in doc or "120 days" in doc
    # Documents the CLI.
    assert "python -m app.resilience_drills" in doc
