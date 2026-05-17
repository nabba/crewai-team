"""Tests for idle_scheduler ↔ substrate.policy wiring (T2.5).

Verifies that:
  - LIGHT jobs are never deferred.
  - HEAVY/MEDIUM jobs consult substrate.policy.should_defer_heavy_work().
  - When a defer reason fires, the job is skipped AND a visible event is
    emitted (no silent suppression).
  - When the substrate package is broken, the scheduler degrades to
    pre-T2.5 behavior (runs jobs normally).
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


class TestSubstrateDeferReason:
    """The _substrate_defer_reason helper applies LIGHT-job exemption."""

    def test_light_never_deferred(self):
        from app.idle_scheduler import _substrate_defer_reason, JobWeight
        assert _substrate_defer_reason(JobWeight.LIGHT) is None

    def test_medium_consults_policy(self, monkeypatch):
        from app.idle_scheduler import _substrate_defer_reason, JobWeight

        def fake_policy(*a, **kw):
            return "disk_free=0.3GB < critical=0.5GB"

        monkeypatch.setattr(
            "app.substrate.policy.should_defer_heavy_work", fake_policy
        )
        reason = _substrate_defer_reason(JobWeight.MEDIUM)
        assert reason is not None
        assert "disk_free" in reason

    def test_heavy_consults_policy(self, monkeypatch):
        from app.idle_scheduler import _substrate_defer_reason, JobWeight

        monkeypatch.setattr(
            "app.substrate.policy.should_defer_heavy_work",
            lambda *a, **kw: "host_substrate_alert=disk_horizon",
        )
        reason = _substrate_defer_reason(JobWeight.HEAVY)
        assert reason is not None
        assert "host_substrate_alert" in reason

    def test_clean_state_returns_none(self, monkeypatch):
        from app.idle_scheduler import _substrate_defer_reason, JobWeight

        monkeypatch.setattr(
            "app.substrate.policy.should_defer_heavy_work",
            lambda *a, **kw: None,
        )
        assert _substrate_defer_reason(JobWeight.HEAVY) is None

    def test_broken_substrate_is_fail_safe(self, monkeypatch):
        """A broken substrate module must NOT stall the scheduler."""
        from app.idle_scheduler import _substrate_defer_reason, JobWeight

        def boom(*a, **kw):
            raise RuntimeError("simulated substrate failure")

        monkeypatch.setattr(
            "app.substrate.policy.should_defer_heavy_work", boom
        )
        # Returns None (fail-open) so jobs continue to run.
        assert _substrate_defer_reason(JobWeight.HEAVY) is None


class TestDeferralPublish:
    """A deferral must always emit a visible event."""

    def test_publish_deferral_calls_workspace_publish(self, monkeypatch):
        from app import idle_scheduler

        captured = {}

        def fake_publish(*args, **kwargs):
            captured["called"] = True
            captured.update(kwargs)

        monkeypatch.setattr(
            "app.workspace_publish.publish_idle_outcome", fake_publish
        )

        idle_scheduler._publish_deferral(
            "test-heavy-job", idle_scheduler.JobWeight.HEAVY, "disk pressure"
        )

        assert captured.get("called") is True
        assert captured.get("source") == "idle_scheduler"
        assert "test-heavy-job" in captured.get("content_template", "")
        assert "disk pressure" in captured.get("content_template", "")

    def test_publish_deferral_swallows_broken_publisher(self, monkeypatch):
        """If publish itself fails, deferral logic must not crash."""
        from app import idle_scheduler

        def boom(*a, **kw):
            raise RuntimeError("simulated publish failure")

        monkeypatch.setattr(
            "app.workspace_publish.publish_idle_outcome", boom
        )

        # Should NOT raise — silent log instead.
        idle_scheduler._publish_deferral(
            "test-job", idle_scheduler.JobWeight.MEDIUM, "reason"
        )
