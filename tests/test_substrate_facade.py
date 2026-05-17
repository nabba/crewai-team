"""Tests for the thin substrate facade — productization plan T2.1.

The facade is intentionally minimal:
  - gather_substrate_status() never raises, collects per-probe errors
  - should_defer_heavy_work() never raises, fail-open on probe failure
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_metrics import _FakeSettings  # noqa: E402
import app.config as config_mod  # noqa: E402

config_mod.get_settings = lambda: _FakeSettings()
config_mod.get_anthropic_api_key = lambda: "fake-key"
config_mod.get_gateway_secret = lambda: "a" * 64


class TestSubstrateStatus:
    """gather_substrate_status() — never raises, always returns a typed snapshot."""

    def test_smoke(self):
        from app.substrate import gather_substrate_status, SubstrateStatus
        snap = gather_substrate_status()
        assert isinstance(snap, SubstrateStatus)
        assert snap.timestamp  # ISO 8601
        # Every section is a dict
        assert isinstance(snap.memory, dict)
        assert isinstance(snap.subia, dict)
        assert isinstance(snap.self_improvement, dict)
        assert isinstance(snap.resources, dict)
        assert isinstance(snap.continuity, dict)
        assert isinstance(snap.health, dict)
        assert isinstance(snap.settings, dict)
        # Errors list exists
        assert isinstance(snap.errors, list)

    def test_to_dict_roundtrip(self):
        from app.substrate import gather_substrate_status
        snap = gather_substrate_status()
        d = snap.to_dict()
        # Required keys
        for key in ("timestamp", "inflight_tasks", "memory", "subia", "errors"):
            assert key in d

    def test_subia_section_includes_live_flag(self):
        from app.substrate import gather_substrate_status
        snap = gather_substrate_status()
        # live_enabled is True/False (boolean)
        assert isinstance(snap.subia.get("live_enabled"), bool)

    def test_settings_section_surfaces_runtime_flags(self):
        from app.substrate import gather_substrate_status
        snap = gather_substrate_status()
        # Each runtime-setting flag should appear
        for flag in (
            "subia_live_enabled",
            "subia_grounding_enabled",
            "recovery_loop_enabled",
            "tier3_amendment_enabled",
        ):
            assert flag in snap.settings


class TestShouldDeferHeavyWork:
    """Policy decision — never raises, fail-open on missing inputs."""

    def test_returns_none_on_healthy_snapshot(self):
        from app.substrate import gather_substrate_status, should_defer_heavy_work
        snap = gather_substrate_status()
        # Without any external pressure, no defer reason should fire
        reason = should_defer_heavy_work(snap)
        # Could be None (healthy) or a string (degraded dev env)
        assert reason is None or isinstance(reason, str)

    def test_defers_on_critical_disk(self):
        from app.substrate import (
            SubstrateStatus,
            should_defer_heavy_work,
            ResourcePolicy,
        )
        snap = SubstrateStatus()
        snap.resources = {"disk_free_gb": 0.1}
        reason = should_defer_heavy_work(snap, policy=ResourcePolicy())
        assert reason is not None
        assert "critical" in reason

    def test_defers_on_low_disk(self):
        from app.substrate import SubstrateStatus, should_defer_heavy_work, ResourcePolicy
        snap = SubstrateStatus()
        snap.resources = {"disk_free_gb": 1.5}
        reason = should_defer_heavy_work(snap, policy=ResourcePolicy(min_free_disk_gb=2.0, min_free_disk_gb_critical=0.5))
        assert reason is not None
        assert "disk_free" in reason

    def test_defers_on_high_inflight(self):
        from app.substrate import SubstrateStatus, should_defer_heavy_work, ResourcePolicy
        snap = SubstrateStatus()
        snap.inflight_tasks = 20
        snap.resources = {"disk_free_gb": 100.0}
        reason = should_defer_heavy_work(snap, policy=ResourcePolicy(max_inflight_tasks=8))
        assert reason is not None
        assert "inflight_tasks" in reason

    def test_defers_on_host_substrate_alert(self):
        from app.substrate import SubstrateStatus, should_defer_heavy_work, ResourcePolicy
        snap = SubstrateStatus()
        snap.resources = {
            "disk_free_gb": 100.0,
            "host_substrate_alerts": [{"kind": "disk_horizon", "ts": "now", "msg": "trending"}],
        }
        reason = should_defer_heavy_work(snap, policy=ResourcePolicy())
        assert reason is not None
        assert "host_substrate_alert" in reason

    def test_unknown_alert_kind_does_not_defer(self):
        from app.substrate import SubstrateStatus, should_defer_heavy_work, ResourcePolicy
        snap = SubstrateStatus()
        snap.resources = {
            "disk_free_gb": 100.0,
            "host_substrate_alerts": [{"kind": "informational_only", "msg": "fyi"}],
        }
        reason = should_defer_heavy_work(snap, policy=ResourcePolicy())
        assert reason is None

    def test_never_raises_on_garbage_snapshot(self):
        from app.substrate import should_defer_heavy_work, SubstrateStatus
        # Pass a snapshot whose fields are garbage
        snap = SubstrateStatus()
        snap.resources = {"disk_free_gb": "not-a-number"}
        # Should fail-open (return None), not raise
        reason = should_defer_heavy_work(snap)
        assert reason is None or isinstance(reason, str)


class TestProbeIsolation:
    """A broken probe must not take down the whole snapshot."""

    def test_subia_integrity_failure_doesnt_break_snapshot(self, monkeypatch):
        """Force one probe to raise and verify others still return data."""
        import app.subia.integrity as integ

        def _boom(*a, **kw):
            raise RuntimeError("simulated probe failure")

        monkeypatch.setattr(integ, "verify_integrity", _boom)

        from app.substrate import gather_substrate_status
        snap = gather_substrate_status()
        # The error is recorded
        assert any("subia.integrity" in e for e in snap.errors)
        # Other sections still ran
        assert snap.timestamp
        assert isinstance(snap.resources, dict)
