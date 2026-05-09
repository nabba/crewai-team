"""Tests for ``app.healing.watchdog`` — the daemon-thread reaper.

Tests focus on the deterministic ``_check_and_respawn`` core; the
daemon loop itself is just a sleep wrapper around it.
"""
from __future__ import annotations

import time

import pytest


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """Reset crash history + give-up state between tests."""
    from app.healing import watchdog
    monkeypatch.setattr(watchdog, "_crash_history",
                         __import__("collections").defaultdict(
                             lambda: __import__("collections").deque(maxlen=10)))
    monkeypatch.setattr(watchdog, "_given_up", {})
    yield


def _stub_registry(monkeypatch, mapping: dict[str, tuple[str, str]]) -> None:
    """Replace the registered daemons with a minimal mapping for the test."""
    from app.healing import watchdog
    monkeypatch.setattr(watchdog, "_REGISTERED_DAEMONS", dict(mapping))


# ── Liveness check ────────────────────────────────────────────────────────


def test_alive_daemon_not_respawned(monkeypatch):
    """A daemon currently alive must NOT be re-spawned."""
    from app.healing import watchdog

    _stub_registry(monkeypatch, {"healing-monitors": ("does.not.matter", "x")})
    monkeypatch.setattr(watchdog, "_is_alive", lambda name: True)
    started = []
    monkeypatch.setattr(watchdog, "_attempt_start",
                        lambda name: started.append(name) or True)

    summary = watchdog._check_and_respawn()
    assert summary["alive"] == ["healing-monitors"]
    assert summary["respawned"] == []
    assert started == []


def test_dead_daemon_respawned(monkeypatch):
    """A daemon NOT alive must be re-spawned."""
    from app.healing import watchdog

    _stub_registry(monkeypatch, {"daemon-A": ("mod.a", "start")})
    monkeypatch.setattr(watchdog, "_is_alive", lambda name: False)
    started = []
    monkeypatch.setattr(
        watchdog, "_attempt_start",
        lambda name: started.append(name) or True,
    )
    monkeypatch.setattr(watchdog, "_audit", lambda *a, **k: None)

    summary = watchdog._check_and_respawn()
    assert summary["respawned"] == ["daemon-A"]
    assert started == ["daemon-A"]


# ── Backoff / give-up ─────────────────────────────────────────────────────


def test_giveup_after_max_crashes(monkeypatch):
    """After 3 crashes within an hour, the watchdog gives up (no further
    re-spawn) and emits a Signal alert.
    """
    from app.healing import watchdog

    _stub_registry(monkeypatch, {"daemon-A": ("mod.a", "start")})
    monkeypatch.setattr(watchdog, "_is_alive", lambda name: False)
    monkeypatch.setattr(watchdog, "_attempt_start", lambda name: True)
    monkeypatch.setattr(watchdog, "_audit", lambda *a, **k: None)

    alerts = []
    monkeypatch.setattr(
        watchdog, "_send_giveup_alert",
        lambda name, n: alerts.append((name, n)),
    )

    # First 3 passes: respawn each time, populating crash history.
    for _ in range(watchdog._MAX_CRASHES_PER_HOUR):
        s = watchdog._check_and_respawn()
        assert s["respawned"] == ["daemon-A"]

    # 4th pass: history is at cap → give up, alert fires.
    s = watchdog._check_and_respawn()
    assert "daemon-A" in s["given_up"]
    assert alerts and alerts[0][0] == "daemon-A"

    # 5th pass: still in give-up — no respawn, no new alert.
    s = watchdog._check_and_respawn()
    assert "daemon-A" in s["still_in_giveup"]
    assert s["respawned"] == []
    assert len(alerts) == 1


def test_giveup_resets_after_24h_quiet(monkeypatch):
    """After 24 h since give-up, the daemon is allowed to be re-spawned."""
    from app.healing import watchdog

    _stub_registry(monkeypatch, {"daemon-A": ("mod.a", "start")})
    monkeypatch.setattr(watchdog, "_is_alive", lambda name: False)
    monkeypatch.setattr(watchdog, "_attempt_start", lambda name: True)
    monkeypatch.setattr(watchdog, "_audit", lambda *a, **k: None)
    monkeypatch.setattr(watchdog, "_send_giveup_alert", lambda *a, **k: None)

    # Force into give-up state.
    watchdog._given_up["daemon-A"] = (
        time.time() - (watchdog._GIVEUP_RESET_HOURS + 1) * 3600
    )

    s = watchdog._check_and_respawn()
    # 24h+ passed since giveup → reset and respawn.
    assert "daemon-A" in s["respawned"]
    assert "daemon-A" not in watchdog._given_up


def test_old_crashes_drop_out_of_window(monkeypatch):
    """Crashes older than 1 h shouldn't count toward the cap."""
    from app.healing import watchdog
    from collections import deque

    _stub_registry(monkeypatch, {"daemon-A": ("mod.a", "start")})
    monkeypatch.setattr(watchdog, "_is_alive", lambda name: False)
    monkeypatch.setattr(watchdog, "_attempt_start", lambda name: True)
    monkeypatch.setattr(watchdog, "_audit", lambda *a, **k: None)

    # Pre-seed crash history with 3 OLD crashes (> 1 h ago).
    old_history = deque(maxlen=10)
    old_ts = time.time() - 3 * 3600
    for _ in range(3):
        old_history.append(old_ts)
    watchdog._crash_history["daemon-A"] = old_history

    # The pass should drop the old crashes from the window and respawn.
    s = watchdog._check_and_respawn()
    assert "daemon-A" in s["respawned"]


# ── Multi-daemon ──────────────────────────────────────────────────────────


def test_one_dead_doesnt_block_others(monkeypatch):
    from app.healing import watchdog

    _stub_registry(monkeypatch, {
        "daemon-A": ("mod.a", "start"),
        "daemon-B": ("mod.b", "start"),
    })
    monkeypatch.setattr(
        watchdog, "_is_alive",
        lambda name: name == "daemon-A",  # A is alive, B is dead
    )
    monkeypatch.setattr(watchdog, "_attempt_start", lambda name: True)
    monkeypatch.setattr(watchdog, "_audit", lambda *a, **k: None)

    s = watchdog._check_and_respawn()
    assert "daemon-A" in s["alive"]
    assert "daemon-B" in s["respawned"]


def test_failed_respawn_still_counts_as_crash(monkeypatch):
    """If ``_attempt_start`` returns False, the crash counts toward the
    backoff cap — otherwise a daemon that refuses to start would be
    re-tried at the watchdog's full cadence forever.
    """
    from app.healing import watchdog

    _stub_registry(monkeypatch, {"broken": ("mod.x", "start")})
    monkeypatch.setattr(watchdog, "_is_alive", lambda name: False)
    monkeypatch.setattr(watchdog, "_attempt_start", lambda name: False)
    monkeypatch.setattr(watchdog, "_audit", lambda *a, **k: None)
    monkeypatch.setattr(watchdog, "_send_giveup_alert", lambda *a, **k: None)

    # 3 failed respawns then giveup.
    for _ in range(watchdog._MAX_CRASHES_PER_HOUR):
        watchdog._check_and_respawn()
    s = watchdog._check_and_respawn()
    assert "broken" in s["given_up"]


# ── Heartbeat + start() idempotency ───────────────────────────────────────


def test_heartbeat_touches_file(tmp_path, monkeypatch):
    from app.healing import watchdog

    fp = tmp_path / "watchdog_heartbeat"
    monkeypatch.setattr(watchdog, "_HEARTBEAT_PATH", fp)
    watchdog._touch_heartbeat()
    assert fp.exists()


def test_start_idempotent_when_alive(monkeypatch):
    from app.healing import watchdog

    monkeypatch.setattr(watchdog, "_is_alive", lambda name: True)
    monkeypatch.setattr(watchdog, "_enabled", lambda: True)

    spawned = []

    def fake_thread_start(self):
        spawned.append(self.name)

    # If start() spawned a thread we'd see it here — but with _is_alive
    # returning True it must NOT spawn.
    import threading as _th
    real_start = _th.Thread.start
    monkeypatch.setattr(_th.Thread, "start", fake_thread_start)
    try:
        watchdog.start()
    finally:
        monkeypatch.setattr(_th.Thread, "start", real_start)
    assert spawned == []


def test_start_disabled_short_circuits(monkeypatch):
    from app.healing import watchdog
    monkeypatch.setattr(watchdog, "_enabled", lambda: False)

    spawned = []
    import threading as _th
    real_start = _th.Thread.start
    monkeypatch.setattr(_th.Thread, "start", lambda self: spawned.append(self.name))
    try:
        watchdog.start()
    finally:
        monkeypatch.setattr(_th.Thread, "start", real_start)
    assert spawned == []


# ── Integration with start_fn idempotency ─────────────────────────────────


def test_existing_start_functions_are_thread_liveness_aware():
    """Regression guard: the daemons we watch must use thread-liveness
    detection in their start(), not just a stale ``_started`` flag.
    """
    from app.healing.monitors import _is_running as monitors_is_running
    from app.healing.auditor_bridge import _is_running as bridge_is_running
    # Just confirming the helpers exist and return bools — the watchdog
    # needs them.
    assert isinstance(monitors_is_running(), bool)
    assert isinstance(bridge_is_running(), bool)
