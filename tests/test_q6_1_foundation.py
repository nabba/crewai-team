"""PROGRAM §44.1 — Q6.1 foundation tests.

Covers:
  * protocol.DrillSpec / DrillResult / DrillRegistry
  * audit.append + iter + last_successful_for + days_since_last_success
  * audit.emit_landmark_for emits on FAIL / first-pass / recovered
  * posture.POSTURE constants + is_ha_proposed_for_subsystem guard
  * continuity_ledger accepts resilience_drill kind
"""
from __future__ import annotations

import importlib.util
import json
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


# ─────────────────────────────────────────────────────────────────────────
#   Protocol (DrillSpec / DrillResult / Registry)
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def protocol():
    return _load_isolated(
        "drill_proto_q61", "app/resilience_drills/protocol.py",
    )


def test_drill_spec_defaults(protocol):
    spec = protocol.DrillSpec(name="x", cadence_days=90)
    assert spec.cadence_days == 90
    assert spec.grace_days == 30
    assert spec.risk == protocol.DrillRisk.LOW
    assert spec.requires_typed_phrase is None
    assert spec.requires_master_switch is None


def test_drill_result_to_dict(protocol):
    r = protocol.DrillResult(
        drill_name="backup_restore",
        status=protocol.DrillStatus.PASS,
        started_at="2026-05-13T10:00:00+00:00",
        completed_at="2026-05-13T10:14:00+00:00",
        duration_s=840.0,
        dry_run=True,
        detail={"tarball": "/path/x.tar.gz"},
    )
    d = r.to_dict()
    assert d["drill_name"] == "backup_restore"
    assert d["status"] == "pass"
    assert d["duration_s"] == 840.0
    assert d["dry_run"] is True
    assert d["detail"]["tarball"] == "/path/x.tar.gz"


def test_registry_register_and_get(protocol):
    reg = protocol.DrillRegistry()
    spec = protocol.DrillSpec(name="foo", cadence_days=90)
    reg.register(spec, lambda *, dry_run=True: protocol.DrillResult(
        drill_name="foo", status=protocol.DrillStatus.PASS,
        started_at="x", completed_at="y", duration_s=0.0, dry_run=dry_run,
    ))
    assert reg.get("foo") is not None
    assert reg.runner_for("foo") is not None
    assert len(reg.list_specs()) == 1


def test_registry_register_is_idempotent(protocol):
    reg = protocol.DrillRegistry()
    spec = protocol.DrillSpec(name="foo", cadence_days=90)
    reg.register(spec, lambda *, dry_run=True: None)
    new_runner = lambda *, dry_run=True: "new"
    reg.register(spec, new_runner)
    assert reg.runner_for("foo") is new_runner
    assert len(reg.list_specs()) == 1


def test_registry_singleton_accessor(protocol):
    a = protocol.get_registry()
    b = protocol.get_registry()
    assert a is b


def test_protocol_master_enabled_default(protocol, monkeypatch):
    """Master switch defaults ON when runtime_settings unavailable."""
    # On dev environments without pydantic_settings, the import fails
    # and we fall through to default True.
    try:
        import app.runtime_settings  # noqa: F401
    except Exception:
        assert protocol.master_enabled() is True
        return
    monkeypatch.setattr(
        "app.runtime_settings.get_resilience_drills_enabled", lambda: True,
    )
    assert protocol.master_enabled() is True


def test_protocol_drill_enabled_respects_per_drill_switch(protocol, monkeypatch):
    """Per-drill switch gates execution even when master is ON."""
    spec = protocol.DrillSpec(
        name="kill_the_gateway", cadence_days=90,
        risk=protocol.DrillRisk.HIGH,
        requires_master_switch="drill_kill_the_gateway_enabled",
    )
    try:
        import app.runtime_settings  # noqa: F401
    except Exception:
        pytest.skip("runtime_settings unavailable")
    monkeypatch.setattr(
        "app.runtime_settings.get_resilience_drills_enabled", lambda: True,
    )
    # Per-drill OFF → disabled.
    monkeypatch.setattr(
        "app.runtime_settings.get_drill_kill_the_gateway_enabled",
        lambda: False,
    )
    assert protocol.drill_enabled(spec) is False
    # Per-drill ON → enabled.
    monkeypatch.setattr(
        "app.runtime_settings.get_drill_kill_the_gateway_enabled",
        lambda: True,
    )
    assert protocol.drill_enabled(spec) is True


# ─────────────────────────────────────────────────────────────────────────
#   Audit (append + iter + landmark emission)
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def audit():
    return _load_isolated(
        "drill_audit_q61", "app/resilience_drills/audit.py",
    )


@pytest.fixture
def proto():
    return _load_isolated(
        "drill_proto_for_audit_q61", "app/resilience_drills/protocol.py",
    )


def test_audit_append_round_trip(audit, proto, monkeypatch, tmp_path):
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(audit, "_default_audit_path", lambda: log)
    r = proto.DrillResult(
        drill_name="backup_restore",
        status=proto.DrillStatus.PASS,
        started_at="2026-05-13T10:00:00+00:00",
        completed_at="2026-05-13T10:14:00+00:00",
        duration_s=840.0,
        dry_run=True,
    )
    assert audit.append_result(r) is True
    rows = list(audit.iter_results())
    assert len(rows) == 1
    assert rows[0]["drill_name"] == "backup_restore"
    assert rows[0]["status"] == "pass"


def test_audit_last_successful_returns_latest_pass(audit, proto, monkeypatch, tmp_path):
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(audit, "_default_audit_path", lambda: log)
    # Mix of PASS and FAIL for the same drill.
    for i, status in enumerate([
        proto.DrillStatus.PASS,
        proto.DrillStatus.FAIL,
        proto.DrillStatus.PASS,  # latest pass
        proto.DrillStatus.FAIL,
    ]):
        r = proto.DrillResult(
            drill_name="backup_restore",
            status=status,
            started_at=f"2026-05-1{i+1}T10:00:00+00:00",
            completed_at=f"2026-05-1{i+1}T10:14:00+00:00",
            duration_s=10.0,
            dry_run=True,
        )
        audit.append_result(r)
    # FAIL is the latest result overall; the latest PASS is the i=2 row.
    last_pass = audit.last_successful_for("backup_restore")
    assert last_pass is not None
    assert last_pass["started_at"] == "2026-05-13T10:00:00+00:00"
    last_any = audit.last_result_for("backup_restore")
    assert last_any["status"] == "fail"


def test_audit_days_since_last_success(audit, proto, monkeypatch, tmp_path):
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(audit, "_default_audit_path", lambda: log)
    five_days_ago = datetime.now(timezone.utc) - timedelta(days=5)
    audit.append_result(proto.DrillResult(
        drill_name="x", status=proto.DrillStatus.PASS,
        started_at=five_days_ago.isoformat(),
        completed_at=five_days_ago.isoformat(),
        duration_s=1.0, dry_run=True,
    ))
    days = audit.days_since_last_success("x")
    assert days is not None
    assert 4.5 <= days <= 5.5


def test_audit_days_since_returns_none_when_no_run(audit, monkeypatch, tmp_path):
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(audit, "_default_audit_path", lambda: log)
    assert audit.days_since_last_success("never_run") is None


def test_audit_emit_landmark_on_fail(audit, proto, monkeypatch, tmp_path):
    """FAIL outcome should emit a continuity-ledger event."""
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(audit, "_default_audit_path", lambda: log)
    captured: list[dict] = []
    def fake_record(**kwargs):
        captured.append(kwargs)
        return True
    monkeypatch.setattr(
        "app.identity.continuity_ledger.record_event", fake_record,
    )
    r = proto.DrillResult(
        drill_name="backup_restore",
        status=proto.DrillStatus.FAIL,
        started_at="2026-05-13T10:00:00+00:00",
        completed_at="2026-05-13T10:14:00+00:00",
        duration_s=840.0,
        dry_run=True,
        errors=["verifier returned ok=false"],
    )
    emitted = audit.emit_landmark_for(r)
    assert emitted is True
    assert len(captured) == 1
    assert captured[0]["kind"] == "resilience_drill"
    assert captured[0]["actor"] == "drill:backup_restore"
    assert "failed" in captured[0]["summary"]
    assert captured[0]["detail"]["status"] == "fail"
    assert captured[0]["detail"]["n_errors"] == 1


def test_audit_emit_landmark_on_first_pass(audit, proto, monkeypatch, tmp_path):
    """First-ever PASS run emits landmark."""
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(audit, "_default_audit_path", lambda: log)
    captured: list[dict] = []
    monkeypatch.setattr(
        "app.identity.continuity_ledger.record_event",
        lambda **kw: (captured.append(kw), True)[1],
    )
    r = proto.DrillResult(
        drill_name="x", status=proto.DrillStatus.PASS,
        started_at="2026-05-13T10:00:00+00:00",
        completed_at="2026-05-13T10:00:01+00:00",
        duration_s=1.0, dry_run=True,
    )
    audit.emit_landmark_for(r, is_first_run=True)
    assert len(captured) == 1
    assert "first_pass" in captured[0]["summary"]


def test_audit_emit_landmark_on_recovery(audit, proto, monkeypatch, tmp_path):
    """PASS after a previous FAIL emits a 'recovered' landmark."""
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(audit, "_default_audit_path", lambda: log)
    # Seed audit with a prior FAIL.
    audit.append_result(proto.DrillResult(
        drill_name="backup_restore", status=proto.DrillStatus.FAIL,
        started_at="2026-05-12T10:00:00+00:00",
        completed_at="2026-05-12T10:01:00+00:00",
        duration_s=60.0, dry_run=True,
    ))
    captured: list[dict] = []
    monkeypatch.setattr(
        "app.identity.continuity_ledger.record_event",
        lambda **kw: (captured.append(kw), True)[1],
    )
    # Now PASS — should emit recovery landmark.
    new_r = proto.DrillResult(
        drill_name="backup_restore", status=proto.DrillStatus.PASS,
        started_at="2026-05-13T10:00:00+00:00",
        completed_at="2026-05-13T10:01:00+00:00",
        duration_s=60.0, dry_run=True,
    )
    audit.emit_landmark_for(new_r)
    assert len(captured) == 1
    assert "recovered" in captured[0]["summary"]


def test_audit_routine_pass_emits_no_landmark(audit, proto, monkeypatch, tmp_path):
    """PASS-then-PASS is routine — no ledger emission."""
    log = tmp_path / "drill_audit.jsonl"
    monkeypatch.setattr(audit, "_default_audit_path", lambda: log)
    audit.append_result(proto.DrillResult(
        drill_name="backup_restore", status=proto.DrillStatus.PASS,
        started_at="2026-05-12T10:00:00+00:00",
        completed_at="2026-05-12T10:01:00+00:00",
        duration_s=60.0, dry_run=True,
    ))
    captured: list[dict] = []
    monkeypatch.setattr(
        "app.identity.continuity_ledger.record_event",
        lambda **kw: (captured.append(kw), True)[1],
    )
    new_r = proto.DrillResult(
        drill_name="backup_restore", status=proto.DrillStatus.PASS,
        started_at="2026-05-13T10:00:00+00:00",
        completed_at="2026-05-13T10:01:00+00:00",
        duration_s=60.0, dry_run=True,
    )
    emitted = audit.emit_landmark_for(new_r)
    assert emitted is False
    assert captured == []


# ─────────────────────────────────────────────────────────────────────────
#   Posture
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def posture():
    return _load_isolated(
        "drill_posture_q61", "app/resilience_drills/posture.py",
    )


def test_posture_ha_disabled_by_default(posture):
    assert posture.POSTURE.ha_enabled is False
    assert "identity is data" in posture.POSTURE.rationale_short


def test_posture_off_host_targets(posture):
    assert posture.POSTURE.off_host_targets == ("s3", "google_drive")


def test_posture_quarterly_drills_list(posture):
    expected = {
        "backup_restore",
        "embedding_migration",
        "secret_rotation",
        "kill_the_gateway",
    }
    assert set(posture.POSTURE.quarterly_drills) == expected


def test_posture_ha_guard_detects_ha_keywords(posture):
    """Names containing 'replica', 'failover', etc. trigger the guard."""
    assert posture.is_ha_proposed_for_subsystem("gateway_replica") is not None
    assert posture.is_ha_proposed_for_subsystem("failover_controller") is not None
    assert posture.is_ha_proposed_for_subsystem("leader_election_pool") is not None
    # Non-HA names pass.
    assert posture.is_ha_proposed_for_subsystem("backup_runner") is None
    assert posture.is_ha_proposed_for_subsystem("drill_scheduler") is None


# ─────────────────────────────────────────────────────────────────────────
#   Continuity ledger event kind
# ─────────────────────────────────────────────────────────────────────────


def test_continuity_ledger_accepts_resilience_drill_kind():
    cl = _load_isolated("cl_q61", "app/identity/continuity_ledger.py")
    assert "resilience_drill" in cl.IDENTITY_EVENT_KINDS


# ─────────────────────────────────────────────────────────────────────────
#   Posture document exists + names the off-host policy
# ─────────────────────────────────────────────────────────────────────────


def test_posture_doc_exists_and_names_decision():
    doc = Path("docs/RESILIENCE_POSTURE.md").read_text()
    assert "identity is data, not uptime" in doc.lower()
    assert "s3" in doc.lower() and "google drive" in doc.lower()
    # Names all four drills.
    for name in ("backup_restore", "embedding_migration",
                 "secret_rotation", "kill_the_gateway"):
        assert name in doc
    # Documents escape conditions.
    assert "escape condition" in doc.lower()
