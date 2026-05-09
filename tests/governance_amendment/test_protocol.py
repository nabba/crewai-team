"""Tests for the Tier-3 amendment protocol."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Redirect store + audit paths to tmp; force protocol ON.

    Stubs runtime_settings to avoid touching the persisted JSON state
    file — the env-fallback path is the simplest way to do this.
    """
    from app.governance_amendment import store, audit, eligibility, protocol

    monkeypatch.setattr(store, "_STATE_DIR", tmp_path / "tier3")
    monkeypatch.setattr(audit, "_AUDIT_PATH", tmp_path / "tier3" / "audit.jsonl")

    # Make ``amendment_protocol_enabled`` read from env (fallback) by
    # raising inside the runtime_settings import. The protocol's
    # try/except routes to the env-var branch.
    def _broken_runtime_settings(*_a, **_kw):
        raise RuntimeError("runtime_settings stubbed off in tests")
    monkeypatch.setattr(
        "app.runtime_settings.get_tier3_amendment_enabled",
        _broken_runtime_settings,
        raising=False,
    )

    # Force the env switch on for the duration of the test.
    monkeypatch.setenv("TIER3_AMENDMENT_ENABLED", "true")
    monkeypatch.setattr(audit, "get_audit",
                        lambda: type("A", (), {"log": lambda **kw: None})(),
                        raising=False)
    yield tmp_path


def _good_eligibility(monkeypatch):
    """Stub eligibility to a passing result."""
    from app.governance_amendment import eligibility, protocol

    def _ok():
        return eligibility.EligibilityResult(
            ok=True, failures=[],
            evidence={"promotion_stats": {"approved": 250, "rolled_back": 5}},
        )

    monkeypatch.setattr(protocol._eligibility, "check_eligibility", _ok)


def _bad_eligibility(monkeypatch):
    from app.governance_amendment import eligibility, protocol

    def _fail():
        return eligibility.EligibilityResult(
            ok=False,
            failures=["insufficient_approved_promotions: 12 < 200"],
            evidence={"promotion_stats": {"approved": 12, "rolled_back": 0}},
        )

    monkeypatch.setattr(protocol._eligibility, "check_eligibility", _fail)


def _allow_target_in_tier3(monkeypatch, *targets: str):
    """Make the protocol's TIER_IMMUTABLE check accept ``targets``."""
    from app.governance_amendment import protocol

    real_isimm = protocol._is_tier_immutable

    def patched(path):
        norm = (path or "").replace("\\", "/").lstrip("/")
        if norm in targets:
            return True, norm
        return real_isimm(path)

    monkeypatch.setattr(protocol, "_is_tier_immutable", patched)


# ── propose_amendment refusals ────────────────────────────────────────────


def test_propose_refused_when_disabled(isolated, monkeypatch):
    from app.governance_amendment import (
        ProtocolDisabled, propose_amendment,
    )
    monkeypatch.setenv("TIER3_AMENDMENT_ENABLED", "false")
    with pytest.raises(ProtocolDisabled):
        propose_amendment(
            target_path="app/version_manifest.py",
            new_content="x", old_content="y",
            citation="raise quality minimum to 0.75 because audit shows headroom",
            proposer="self_improver",
        )


def test_runtime_settings_overrides_env_when_available(monkeypatch):
    """The React toggle wins: when ``runtime_settings`` is importable and
    returns ``True``, ``amendment_protocol_enabled`` returns True even if
    the env var is unset.
    """
    from app.governance_amendment import protocol

    monkeypatch.delenv("TIER3_AMENDMENT_ENABLED", raising=False)
    import app.runtime_settings as rs
    monkeypatch.setattr(rs, "get_tier3_amendment_enabled", lambda: True)
    assert protocol.amendment_protocol_enabled() is True


def test_runtime_settings_off_blocks_protocol(monkeypatch):
    """Conversely, the React toggle OFF blocks the protocol even if the
    env var is set — runtime_settings is the canonical read path on a
    live system.
    """
    from app.governance_amendment import protocol

    monkeypatch.setenv("TIER3_AMENDMENT_ENABLED", "true")
    import app.runtime_settings as rs
    monkeypatch.setattr(rs, "get_tier3_amendment_enabled", lambda: False)
    assert protocol.amendment_protocol_enabled() is False


def test_env_fallback_kicks_in_when_runtime_settings_raises(monkeypatch):
    """If runtime_settings explodes (e.g. corrupted state file in a
    degraded boot), the env-var fallback keeps the protocol switchable.
    """
    from app.governance_amendment import protocol

    def _explode():
        raise RuntimeError("runtime_settings unavailable")

    import app.runtime_settings as rs
    monkeypatch.setattr(rs, "get_tier3_amendment_enabled", _explode)
    monkeypatch.setenv("TIER3_AMENDMENT_ENABLED", "true")
    assert protocol.amendment_protocol_enabled() is True
    monkeypatch.setenv("TIER3_AMENDMENT_ENABLED", "false")
    assert protocol.amendment_protocol_enabled() is False


def test_propose_refuses_short_citation(isolated, monkeypatch):
    from app.governance_amendment import propose_amendment
    _good_eligibility(monkeypatch)
    _allow_target_in_tier3(monkeypatch, "app/version_manifest.py")
    with pytest.raises(ValueError, match="citation"):
        propose_amendment(
            target_path="app/version_manifest.py",
            new_content="x", old_content="y",
            citation="too short",
            proposer="self_improver",
        )


def test_propose_refuses_non_tier3(isolated, monkeypatch):
    from app.governance_amendment import propose_amendment
    _good_eligibility(monkeypatch)
    with pytest.raises(ValueError, match="TIER_IMMUTABLE"):
        propose_amendment(
            target_path="app/some/random/non_tier3_file.py",
            new_content="x", old_content="y",
            citation="this is a long-enough citation explaining the target",
            proposer="self_improver",
        )


def test_propose_refuses_quarantined(isolated, monkeypatch):
    from app.governance_amendment import propose_amendment
    _good_eligibility(monkeypatch)
    # safety_guardian.py is in BOTH TIER_IMMUTABLE and QUARANTINED_FILES.
    _allow_target_in_tier3(monkeypatch, "app/safety_guardian.py")
    with pytest.raises(ValueError, match="self-quarantine"):
        propose_amendment(
            target_path="app/safety_guardian.py",
            new_content="x", old_content="y",
            citation="this is a long-enough citation explaining the target",
            proposer="self_improver",
        )


def test_governance_thresholds_are_quarantined(isolated, monkeypatch):
    """Sanity: ``app/governance.py`` itself MUST be quarantined.

    This is the protocol's own guard against an agent earning amendment
    privileges and then quietly lowering its own safety/quality minimums.
    Threshold ratcheting (raising the floor over time) requires a
    separate, monotonic-only protocol — out of scope for V1.
    """
    from app.governance_amendment import propose_amendment, is_quarantined
    _good_eligibility(monkeypatch)
    _allow_target_in_tier3(monkeypatch, "app/governance.py")
    assert is_quarantined("app/governance.py")
    with pytest.raises(ValueError, match="self-quarantine"):
        propose_amendment(
            target_path="app/governance.py",
            new_content="SAFETY_MINIMUM = 0.50",
            old_content="SAFETY_MINIMUM = 0.95",
            citation="this is a long-enough citation explaining the target",
            proposer="self_improver",
        )


# ── Eligibility-driven branching ──────────────────────────────────────────


def test_propose_with_failing_eligibility_records_terminal_state(isolated, monkeypatch):
    from app.governance_amendment import State, propose_amendment
    _bad_eligibility(monkeypatch)
    _allow_target_in_tier3(monkeypatch, "app/version_manifest.py")
    p = propose_amendment(
        target_path="app/version_manifest.py",
        new_content="QUALITY_MINIMUM = 0.75",
        old_content="QUALITY_MINIMUM = 0.70",
        citation="raise quality floor — audit shows last 12 promotions all >0.85",
        proposer="self_improver",
    )
    assert p.state == State.ELIGIBILITY_FAILED
    assert any("insufficient_approved_promotions" in f for f in p.eligibility_failures)


def test_propose_with_passing_eligibility_goes_to_staged(isolated, monkeypatch):
    from app.governance_amendment import State, propose_amendment
    _good_eligibility(monkeypatch)
    _allow_target_in_tier3(monkeypatch, "app/version_manifest.py")
    p = propose_amendment(
        target_path="app/version_manifest.py",
        new_content="QUALITY_MINIMUM = 0.75",
        old_content="QUALITY_MINIMUM = 0.70",
        citation="raise quality floor — audit shows headroom; rationale...",
        proposer="self_improver",
    )
    assert p.state == State.STAGED
    assert p.staged_at != ""
    assert p.cooldown_started_at != ""


# ── Cooldown advance ──────────────────────────────────────────────────────


def test_cooldown_rollback_signal_aborts_immediately(isolated, monkeypatch):
    from app.governance_amendment import (
        State, propose_amendment, advance_cooldown,
    )
    _good_eligibility(monkeypatch)
    _allow_target_in_tier3(monkeypatch, "app/version_manifest.py")
    p = propose_amendment(
        target_path="app/version_manifest.py",
        new_content="x", old_content="y",
        citation="long-enough rationale that explains the change for review",
        proposer="self_improver",
    )
    p2 = advance_cooldown(p.id, rollback_signal="alignment_audit_warning")
    assert p2.state == State.COOLDOWN_FAILED
    assert p2.rollback_signal == "alignment_audit_warning"


def test_cooldown_passes_after_seven_days(isolated, monkeypatch):
    from app.governance_amendment import (
        State, propose_amendment, advance_cooldown, store,
    )
    _good_eligibility(monkeypatch)
    _allow_target_in_tier3(monkeypatch, "app/version_manifest.py")
    p = propose_amendment(
        target_path="app/version_manifest.py",
        new_content="x", old_content="y",
        citation="long-enough rationale that explains the change for review",
        proposer="self_improver",
    )
    # Backdate cooldown_started_at by 8 days.
    p.cooldown_started_at = (
        datetime.now(timezone.utc) - timedelta(days=8)
    ).isoformat()
    store.save(p)

    p2 = advance_cooldown(p.id)
    assert p2.state == State.COOLDOWN_OK
    assert p2.cooldown_passed_at != ""


def test_cooldown_no_op_under_seven_days(isolated, monkeypatch):
    from app.governance_amendment import (
        State, propose_amendment, advance_cooldown,
    )
    _good_eligibility(monkeypatch)
    _allow_target_in_tier3(monkeypatch, "app/version_manifest.py")
    p = propose_amendment(
        target_path="app/version_manifest.py",
        new_content="x", old_content="y",
        citation="long-enough rationale that explains the change for review",
        proposer="self_improver",
    )
    # Don't backdate — cooldown is fresh.
    p2 = advance_cooldown(p.id)
    assert p2.state == State.STAGED  # unchanged


# ── Operator decisions ────────────────────────────────────────────────────


def _to_cooldown_ok(monkeypatch, target="app/version_manifest.py"):
    from app.governance_amendment import (
        propose_amendment, advance_cooldown, store,
    )
    _good_eligibility(monkeypatch)
    _allow_target_in_tier3(monkeypatch, target)
    p = propose_amendment(
        target_path=target,
        new_content="x", old_content="y",
        citation="long-enough rationale that explains the change for review",
        proposer="self_improver",
    )
    p.cooldown_started_at = (
        datetime.now(timezone.utc) - timedelta(days=8)
    ).isoformat()
    store.save(p)
    return advance_cooldown(p.id)


def test_operator_approve_transitions_to_approved(isolated, monkeypatch):
    from app.governance_amendment import State, operator_approve
    p = _to_cooldown_ok(monkeypatch)
    p2 = operator_approve(p.id, source="signal_thumbs_up", reason="ok")
    assert p2.state == State.APPROVED
    assert p2.approved_at != ""


def test_operator_reject_with_reason_terminal(isolated, monkeypatch):
    from app.governance_amendment import State, operator_reject
    p = _to_cooldown_ok(monkeypatch)
    p2 = operator_reject(p.id, source="signal_thumbs_down", reason="too risky")
    assert p2.state == State.REJECTED
    assert p2.operator_decision_reason == "too risky"


def test_operator_reject_requires_reason(isolated, monkeypatch):
    from app.governance_amendment import operator_reject
    p = _to_cooldown_ok(monkeypatch)
    with pytest.raises(ValueError, match="reason"):
        operator_reject(p.id, source="x", reason="")


# ── Apply + monitoring ────────────────────────────────────────────────────


def test_full_lifecycle_to_stable(isolated, monkeypatch):
    from app.governance_amendment import (
        State, operator_approve, mark_applied, advance_monitoring, store,
    )
    p = _to_cooldown_ok(monkeypatch)
    p = operator_approve(p.id, source="signal_thumbs_up")
    p = mark_applied(p.id, applied_by="host_bridge")
    assert p.state == State.APPLIED

    # Backdate applied_at by 31 days.
    p.applied_at = (
        datetime.now(timezone.utc) - timedelta(days=31)
    ).isoformat()
    store.save(p)

    p = advance_monitoring(p.id)
    assert p.state == State.STABLE
    assert p.stable_at != ""


def test_monitoring_rollback_signal_reverts(isolated, monkeypatch):
    from app.governance_amendment import (
        State, operator_approve, mark_applied, advance_monitoring,
    )
    p = _to_cooldown_ok(monkeypatch)
    p = operator_approve(p.id, source="signal_thumbs_up")
    p = mark_applied(p.id, applied_by="host_bridge")
    p = advance_monitoring(p.id, reverted_signal="goodhart_severity_high")
    assert p.state == State.REVERTED
    assert p.rollback_signal == "goodhart_severity_high"


# ── State machine illegal transitions ─────────────────────────────────────


def test_illegal_transition_raises(isolated, monkeypatch):
    from app.governance_amendment import (
        operator_approve, propose_amendment, InvalidStateTransition,
    )
    _good_eligibility(monkeypatch)
    _allow_target_in_tier3(monkeypatch, "app/version_manifest.py")
    p = propose_amendment(
        target_path="app/version_manifest.py",
        new_content="x", old_content="y",
        citation="long-enough rationale that explains the change for review",
        proposer="self_improver",
    )
    # STAGED → APPROVED is not legal (must pass through COOLDOWN_OK first).
    with pytest.raises(InvalidStateTransition):
        operator_approve(p.id, source="x")


# ── Audit chain integrity ─────────────────────────────────────────────────


def test_audit_chain_intact_after_full_lifecycle(isolated, monkeypatch):
    from app.governance_amendment import (
        operator_approve, mark_applied, advance_monitoring,
        verify_audit_chain, store,
    )
    p = _to_cooldown_ok(monkeypatch)
    p = operator_approve(p.id, source="signal_thumbs_up")
    p = mark_applied(p.id, applied_by="host_bridge")
    p.applied_at = (
        datetime.now(timezone.utc) - timedelta(days=31)
    ).isoformat()
    store.save(p)
    p = advance_monitoring(p.id)

    ok, broken = verify_audit_chain()
    assert ok
    assert broken == []
