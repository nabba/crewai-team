"""Tests for ``app.healing.monitors.listener_heartbeat`` (Wave 0/1 #A3).

Two-layer design exercised: per-listener heartbeats first; workspace
proxy fallback when none exist.
"""
from __future__ import annotations

import os
import time

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.healing import listener_heartbeats
    from app.healing.monitors import listener_heartbeat
    from app.healing.handlers import _common as _h_common

    # listener_heartbeat reads/writes state via app.healing.handlers._common
    monkeypatch.setattr(_h_common, "_STATE_DIR", tmp_path / "self_heal")
    monkeypatch.setattr(listener_heartbeats, "_HEARTBEAT_DIR", tmp_path / "hb")

    sent: list[str] = []
    monkeypatch.setattr(listener_heartbeat, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(listener_heartbeat, "audit_event", lambda *a, **k: None)

    yield tmp_path, sent


def test_per_listener_path_no_alerts_when_fresh(isolated):
    """All KNOWN listeners present + fresh = no alerts.

    Phase F #9 added missing-heartbeat alerts when SOME heartbeats
    exist but a known listener has none — so the test now touches
    every listener in KNOWN_LISTENERS to suppress that path."""
    tmp_path, sent = isolated
    from app.healing import listener_heartbeats
    from app.healing.monitors import listener_heartbeat

    for name in listener_heartbeats.KNOWN_LISTENERS:
        listener_heartbeats.touch(name)

    listener_heartbeat.run()
    assert sent == []


def test_per_listener_path_alerts_on_stale(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.healing import listener_heartbeats
    from app.healing.monitors import listener_heartbeat

    monkeypatch.setenv("HEALING_LISTENER_STALE_MIN", "5")  # 5-min threshold

    listener_heartbeats.touch("firebase-kb-poll")
    p = tmp_path / "hb" / "firebase-kb-poll.heartbeat"
    old = time.time() - 600  # 10 min stale
    os.utime(p, (old, old))

    listener_heartbeats.touch("firebase-mode-poll")  # this one fresh

    listener_heartbeat.run()
    assert any("firebase-kb-poll" in s and "stale" in s.lower() for s in sent)
    assert not any("firebase-mode-poll" in s for s in sent)


def test_per_listener_alert_dedup(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.healing import listener_heartbeats
    from app.healing.monitors import listener_heartbeat

    monkeypatch.setenv("HEALING_LISTENER_STALE_MIN", "5")

    listener_heartbeats.touch("firebase-kb-poll")
    p = tmp_path / "hb" / "firebase-kb-poll.heartbeat"
    old = time.time() - 600
    os.utime(p, (old, old))

    listener_heartbeat.run()
    listener_heartbeat.run()  # within cooldown window
    # Only one alert.
    assert sum("firebase-kb-poll" in s for s in sent) == 1


def test_fallback_path_when_no_heartbeats_exist(isolated, monkeypatch):
    """When workspace/heartbeats/ is empty, the monitor reverts to the
    legacy workspace-wide proxy probe."""
    tmp_path, sent = isolated
    from app.healing.monitors import listener_heartbeat

    monkeypatch.setenv("HEALING_LISTENER_STALE_MIN", "5")

    # Force a fake repo root so the probe paths point at tmp.
    fake_root = tmp_path / "fake-root"
    fake_root.mkdir()

    monkeypatch.setattr(
        listener_heartbeat, "_LIVENESS_PROBES",
        ["activity.log"],
    )

    # Build a stale activity probe.
    probe = fake_root / "activity.log"
    probe.write_text("")
    old = time.time() - 600
    os.utime(probe, (old, old))

    # Patch _check_workspace_proxy to use our fake root.
    real = listener_heartbeat._check_workspace_proxy
    def fake_proxy(now, threshold):
        age = now - probe.stat().st_mtime
        return age, "activity.log"
    monkeypatch.setattr(listener_heartbeat, "_check_workspace_proxy", fake_proxy)

    listener_heartbeat.run()
    assert any("workspace activity" in s.lower() for s in sent)


def test_no_heartbeats_no_probe_no_alert(isolated, monkeypatch):
    """If both paths come up empty (no probes, no heartbeats), alert
    fires the 'newest activity is ∞ old' message."""
    tmp_path, sent = isolated
    from app.healing.monitors import listener_heartbeat

    # Make every probe report missing.
    monkeypatch.setattr(listener_heartbeat, "_LIVENESS_PROBES", [])

    listener_heartbeat.run()
    assert any("∞" in s or "none" in s.lower() for s in sent)
