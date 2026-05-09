"""Tests for the Goodhart hard gate (Wave 3 #2)."""
from __future__ import annotations

import json
import time

import pytest


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Redirect the goodhart report file + ratchet state file.

    The new reader hierarchy in ``governance.py`` prefers
    ``runtime_settings`` over env vars. We patch the runtime-settings
    getters directly so tests can flip the gate without depending on
    runtime-settings init / cache state.
    """
    from app import goodhart_guard
    from app.governance_ratchet import store as ratchet_store
    from app.governance_ratchet import audit as ratchet_audit

    monkeypatch.setattr(
        goodhart_guard, "GAMING_REPORT_PATH", tmp_path / "goodhart_reports.json",
    )
    monkeypatch.setattr(ratchet_store, "_STATE_PATH", tmp_path / "ratchet_state.json")
    monkeypatch.setattr(ratchet_audit, "_AUDIT_PATH", tmp_path / "ratchet_audit.jsonl")

    # Stub Postgres-touching paths in governance.py.
    monkeypatch.setattr("app.governance._check_rate_limit", lambda _s: True)
    monkeypatch.setattr("app.governance._record_promotion", lambda *_a: None)

    # Patch runtime-settings getters so tests control the gate directly.
    # The default state is advisory (both flags False).
    flags = {"disabled": False, "enforcing": False}

    def _set_disabled(v: bool) -> None:
        flags["disabled"] = v

    def _set_enforcing(v: bool) -> None:
        flags["enforcing"] = v

    monkeypatch.setattr(
        "app.runtime_settings.get_goodhart_hard_gate_disabled",
        lambda: flags["disabled"],
    )
    monkeypatch.setattr(
        "app.runtime_settings.get_goodhart_hard_gate_enforcing",
        lambda: flags["enforcing"],
    )

    # Also clear env so the env-fallback path is consistent.
    monkeypatch.delenv("GOODHART_HARD_GATE_DISABLED", raising=False)
    monkeypatch.delenv("GOODHART_HARD_GATE_ENFORCING", raising=False)

    yield tmp_path, _set_disabled, _set_enforcing


def _write_signals(path, severities_with_offsets):
    """Write a goodhart_reports.json with given severities + age offsets (s)."""
    now = time.time()
    payload = []
    for sev, offset_s in severities_with_offsets:
        payload.append({
            "signal_type": "kept_ratio_spike",
            "severity": sev,
            "description": f"test signal {sev}",
            "metric_value": 0.9,
            "threshold": 0.85,
            "detected_at": now - offset_s,
        })
    path.write_text(json.dumps(payload, default=str))


def _passing_request():
    from app.governance import PromotionRequest
    return PromotionRequest(
        system="evolution",
        target="researcher_v3",
        proposed_by="self_improver",
        quality_score=0.90,
        safety_score=0.99,
    )


# ── recent_severity / recent_signal_summary ──────────────────────────────


def test_recent_severity_none_when_no_file(isolated):
    from app.goodhart_guard import recent_severity
    assert recent_severity() == "none"


def test_recent_severity_picks_highest(isolated):
    from app.goodhart_guard import recent_severity, GAMING_REPORT_PATH
    _write_signals(GAMING_REPORT_PATH, [
        ("low", 100),
        ("high", 200),
        ("medium", 300),
    ])
    assert recent_severity() == "high"


def test_recent_severity_outside_window_excluded(isolated):
    from app.goodhart_guard import recent_severity, GAMING_REPORT_PATH
    _write_signals(GAMING_REPORT_PATH, [
        ("high", 100 * 3600),  # 100 hours ago — outside default 24h
        ("low", 60),
    ])
    assert recent_severity(lookback_hours=24) == "low"


def test_recent_signal_summary_counts(isolated):
    from app.goodhart_guard import recent_signal_summary, GAMING_REPORT_PATH
    _write_signals(GAMING_REPORT_PATH, [
        ("high", 60),
        ("medium", 120),
        ("medium", 180),
        ("low", 200),
    ])
    summary = recent_signal_summary()
    assert summary["highest_severity"] == "high"
    assert summary["counts"]["high"] == 1
    assert summary["counts"]["medium"] == 2
    assert summary["counts"]["low"] == 1


# ── Hard-gate behaviour in evaluate_promotion ─────────────────────────────


def test_advisory_phase_does_not_block(isolated):
    """Default (no enforcing flag) — high severity is RECORDED but the
    promotion still proceeds.
    """
    from app.governance import evaluate_promotion
    from app.goodhart_guard import GAMING_REPORT_PATH

    _write_signals(GAMING_REPORT_PATH, [("high", 60)])
    result = evaluate_promotion(_passing_request())
    assert result.approved is True
    goodhart = result.gate_results.get("goodhart") or {}
    assert goodhart.get("phase") == "advisory"
    assert goodhart.get("severity") == "high"
    assert goodhart.get("block") is False


def test_enforcing_phase_blocks_on_high(isolated, monkeypatch):
    """With enforcing flag ON, severity=high blocks the promotion."""
    isolated[2](True)  # enforcing=True via the runtime-settings stub

    from app.governance import evaluate_promotion
    from app.goodhart_guard import GAMING_REPORT_PATH

    _write_signals(GAMING_REPORT_PATH, [("high", 60)])
    result = evaluate_promotion(_passing_request())
    assert result.approved is False
    assert "Goodhart hard gate blocked" in result.reason
    assert "'high'" in result.reason
    goodhart = result.gate_results.get("goodhart") or {}
    assert goodhart.get("phase") == "enforcing"
    assert goodhart.get("block") is True


def test_enforcing_phase_passes_on_medium(isolated, monkeypatch):
    """Only severity=high blocks — medium and low are allowed even
    when enforcing is on. (Otherwise we'd block on every category-
    concentration warning.)
    """
    isolated[2](True)  # enforcing=True via the runtime-settings stub

    from app.governance import evaluate_promotion
    from app.goodhart_guard import GAMING_REPORT_PATH

    _write_signals(GAMING_REPORT_PATH, [("medium", 60), ("low", 120)])
    result = evaluate_promotion(_passing_request())
    assert result.approved is True


def test_emergency_disable_skips_gate(isolated, monkeypatch):
    """GOODHART_HARD_GATE_DISABLED=true bypasses both phases — for
    incident response when the detector is buggy.
    """
    isolated[2](True)  # enforcing=True via the runtime-settings stub
    isolated[1](True)  # disabled=True via the runtime-settings stub

    from app.governance import evaluate_promotion
    from app.goodhart_guard import GAMING_REPORT_PATH

    _write_signals(GAMING_REPORT_PATH, [("high", 60)])
    result = evaluate_promotion(_passing_request())
    assert result.approved is True
    goodhart = result.gate_results.get("goodhart") or {}
    assert goodhart.get("phase") == "disabled"


def test_no_signals_no_block(isolated, monkeypatch):
    """Empty / missing report file → severity=none → never blocks."""
    isolated[2](True)  # enforcing=True via the runtime-settings stub

    from app.governance import evaluate_promotion

    result = evaluate_promotion(_passing_request())
    assert result.approved is True
    goodhart = result.gate_results.get("goodhart") or {}
    assert goodhart.get("severity") == "none"


def test_detector_failure_fails_open(isolated, monkeypatch):
    """If recent_signal_summary raises, the gate fails OPEN — we don't
    let a buggy detector halt every promotion.
    """
    isolated[2](True)  # enforcing=True via the runtime-settings stub

    def _explode(*_a, **_kw):
        raise RuntimeError("detector broken")

    monkeypatch.setattr("app.goodhart_guard.recent_signal_summary", _explode)

    from app.governance import evaluate_promotion
    result = evaluate_promotion(_passing_request())
    assert result.approved is True
    goodhart = result.gate_results.get("goodhart") or {}
    assert "detector unavailable" in goodhart.get("description", "")


def test_gate_evaluated_before_safety_minimum(isolated, monkeypatch):
    """Gate-0 (goodhart) runs before gate-1 (safety). A high-severity
    block PRE-EMPTS the safety check — the result reason mentions
    goodhart, not safety.
    """
    isolated[2](True)  # enforcing=True via the runtime-settings stub

    from app.governance import evaluate_promotion, PromotionRequest
    from app.goodhart_guard import GAMING_REPORT_PATH

    _write_signals(GAMING_REPORT_PATH, [("high", 60)])
    # A request with safety_score < FLOOR — would normally fail safety,
    # but goodhart block fires first.
    req = PromotionRequest(
        system="evolution",
        target="x",
        proposed_by="y",
        quality_score=0.90,
        safety_score=0.50,  # WAY below FLOOR
    )
    result = evaluate_promotion(req)
    assert not result.approved
    assert "Goodhart" in result.reason
    assert "Safety gate failed" not in result.reason


def test_runtime_settings_overrides_env_for_disabled(monkeypatch):
    """When runtime_settings says ``disabled=True``, the gate is OFF
    even with ``GOODHART_HARD_GATE_DISABLED=false`` in env. The React
    toggle is canonical on a live system.
    """
    from app.governance import _goodhart_hard_gate_disabled

    monkeypatch.delenv("GOODHART_HARD_GATE_DISABLED", raising=False)
    monkeypatch.setattr(
        "app.runtime_settings.get_goodhart_hard_gate_disabled",
        lambda: True,
    )
    assert _goodhart_hard_gate_disabled() is True


def test_runtime_settings_overrides_env_for_enforcing(monkeypatch):
    from app.governance import _goodhart_hard_gate_enforcing

    monkeypatch.delenv("GOODHART_HARD_GATE_ENFORCING", raising=False)
    monkeypatch.setattr(
        "app.runtime_settings.get_goodhart_hard_gate_enforcing",
        lambda: True,
    )
    assert _goodhart_hard_gate_enforcing() is True


def test_env_fallback_when_runtime_settings_unavailable(monkeypatch):
    """If ``runtime_settings`` raises (corrupted JSON / degraded boot),
    the env-var fallback keeps the gate switchable.
    """
    from app.governance import (
        _goodhart_hard_gate_disabled, _goodhart_hard_gate_enforcing,
    )

    def _explode():
        raise RuntimeError("runtime_settings unavailable")

    monkeypatch.setattr(
        "app.runtime_settings.get_goodhart_hard_gate_disabled", _explode,
    )
    monkeypatch.setattr(
        "app.runtime_settings.get_goodhart_hard_gate_enforcing", _explode,
    )

    monkeypatch.setenv("GOODHART_HARD_GATE_DISABLED", "true")
    monkeypatch.setenv("GOODHART_HARD_GATE_ENFORCING", "true")
    assert _goodhart_hard_gate_disabled() is True
    assert _goodhart_hard_gate_enforcing() is True

    monkeypatch.delenv("GOODHART_HARD_GATE_DISABLED")
    monkeypatch.delenv("GOODHART_HARD_GATE_ENFORCING")
    assert _goodhart_hard_gate_disabled() is False
    assert _goodhart_hard_gate_enforcing() is False


def test_runbooks_runtime_settings_priority(monkeypatch):
    """``app.healing.runbooks.runbooks_enabled`` reads runtime_settings
    first, falls back to env.
    """
    from app.healing import runbooks

    monkeypatch.setattr(
        "app.runtime_settings.get_error_runbooks_enabled", lambda: True,
    )
    monkeypatch.delenv("ERROR_RUNBOOKS_ENABLED", raising=False)
    assert runbooks.runbooks_enabled() is True

    monkeypatch.setattr(
        "app.runtime_settings.get_error_runbooks_enabled", lambda: False,
    )
    monkeypatch.setenv("ERROR_RUNBOOKS_ENABLED", "true")  # ignored
    assert runbooks.runbooks_enabled() is False


def test_supervisor_runtime_settings_priority(monkeypatch):
    """``app.tool_runtime.supervisor.is_enabled`` follows the same
    runtime-settings → env hierarchy.
    """
    from app.tool_runtime import supervisor

    monkeypatch.setattr(
        "app.runtime_settings.get_tool_supervisor_enabled", lambda: True,
    )
    monkeypatch.delenv("TOOL_SUPERVISOR_ENABLED", raising=False)
    assert supervisor.is_enabled() is True


def test_recovery_loop_runtime_settings_priority(monkeypatch):
    from app.recovery import loop

    monkeypatch.setattr(
        "app.runtime_settings.get_recovery_loop_enabled", lambda: True,
    )
    monkeypatch.delenv("RECOVERY_LOOP_ENABLED", raising=False)
    assert loop.is_enabled() is True


def test_advisory_records_severity_for_audit(isolated):
    """Even in advisory mode, the gate records the severity in
    gate_results['goodhart'] so the audit trail captures it.
    """
    from app.governance import evaluate_promotion
    from app.goodhart_guard import GAMING_REPORT_PATH

    _write_signals(GAMING_REPORT_PATH, [
        ("high", 60),
        ("medium", 120),
        ("medium", 180),
    ])
    result = evaluate_promotion(_passing_request())
    assert result.approved is True
    goodhart = result.gate_results.get("goodhart") or {}
    assert goodhart.get("counts") == {"low": 0, "medium": 2, "high": 1}
