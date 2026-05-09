"""Phase H targeted tests — pin behavior for the 4 silent-failure fixes."""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pytest


# ── H4: mem0_manager startup-path bounded retry ─────────────────────────


def test_mem0_get_config_appends_connect_timeout():
    """The pg_url passed to mem0 must include connect_timeout so
    psycopg2.connect() can't hang at OS level."""
    from app.memory import mem0_manager
    import inspect
    src = inspect.getsource(mem0_manager._get_config)
    assert "connect_timeout=" in src
    assert "MEM0_PG_CONNECT_TIMEOUT_S" in src


def test_mem0_get_client_retries_then_alerts(monkeypatch):
    """Three consecutive Memory.from_config failures → degraded boot
    + Signal alert. Retry was the missing safety net."""
    from app.memory import mem0_manager

    # Reset module-level singleton state.
    monkeypatch.setattr(mem0_manager, "_client", None, raising=False)
    monkeypatch.setattr(mem0_manager, "_init_failed", False, raising=False)

    # Avoid the real mem0 import + speed up backoff.
    import sys
    import types
    fake_mem0 = types.ModuleType("mem0")
    class _FailingMemory:
        @classmethod
        def from_config(cls, cfg):
            raise RuntimeError("postgres unreachable")
    fake_mem0.Memory = _FailingMemory
    monkeypatch.setitem(sys.modules, "mem0", fake_mem0)
    monkeypatch.setattr(mem0_manager, "_get_config", lambda: {"x": 1})
    monkeypatch.setattr("time.sleep", lambda *_: None)

    # Stub get_settings to return a Settings-like object with the field
    # the config-build block reads. Without this, a previously-loaded
    # v2 test shim's stripped-down Settings raises AttributeError on
    # ``mem0_enabled`` and the code path bypasses the retry loop.
    class _S:
        mem0_enabled = True
    monkeypatch.setattr("app.config.get_settings", lambda: _S())

    sent: list[str] = []
    monkeypatch.setattr(
        "app.healing.handlers._common.send_signal_alert",
        lambda body, **kw: sent.append(body) or True,
    )
    out = mem0_manager.get_client()
    assert out is None
    assert mem0_manager._init_failed is True
    assert any("init failed 3" in s for s in sent)


# ── H3: idle_scheduler half-open retry ──────────────────────────────────


def test_half_open_probe_allowed_at_quarter_mark():
    """At ¼ of the cooldown window, a probe should be allowed."""
    from app import idle_scheduler
    # Reset state.
    idle_scheduler._job_half_open_used.pop("test_job", None)

    cooldown_s = idle_scheduler.JOB_COOLDOWN_AFTER_FAILURES_S
    skip_until = time.time() + cooldown_s * 0.74  # 26% elapsed
    now = time.time()
    assert idle_scheduler._half_open_probe_allowed("test_job", skip_until, now) is True
    # Second call at same fraction must NOT re-probe.
    assert idle_scheduler._half_open_probe_allowed("test_job", skip_until, now) is False


def test_half_open_probe_three_points():
    """Three probes available at 0.25 / 0.5 / 0.75."""
    from app import idle_scheduler
    idle_scheduler._job_half_open_used.pop("multi", None)
    cooldown_s = idle_scheduler.JOB_COOLDOWN_AFTER_FAILURES_S

    # At 26% elapsed: probe 1 allowed.
    su = time.time() + cooldown_s * 0.74
    assert idle_scheduler._half_open_probe_allowed("multi", su, time.time())
    # At 51% elapsed: probe 2.
    su = time.time() + cooldown_s * 0.49
    assert idle_scheduler._half_open_probe_allowed("multi", su, time.time())
    # At 76% elapsed: probe 3.
    su = time.time() + cooldown_s * 0.24
    assert idle_scheduler._half_open_probe_allowed("multi", su, time.time())
    # No more probes after all three consumed.
    assert idle_scheduler._half_open_probe_allowed("multi", su, time.time()) is False


def test_half_open_probe_skipped_at_start():
    """Right after cooldown starts (~0% elapsed), no probe."""
    from app import idle_scheduler
    idle_scheduler._job_half_open_used.pop("fresh", None)
    cooldown_s = idle_scheduler.JOB_COOLDOWN_AFTER_FAILURES_S
    skip_until = time.time() + cooldown_s * 0.99   # 1% elapsed
    assert idle_scheduler._half_open_probe_allowed("fresh", skip_until, time.time()) is False


def test_clear_cooldown_resets_state():
    """``_clear_cooldown(name)`` removes skip + counter + probe state."""
    from app import idle_scheduler
    idle_scheduler._job_skip_until["clearme"] = time.time() + 3600
    idle_scheduler._job_failure_counts["clearme"] = 5
    idle_scheduler._job_half_open_used["clearme"] = {0.25}
    idle_scheduler._clear_cooldown("clearme")
    assert "clearme" not in idle_scheduler._job_skip_until
    assert idle_scheduler._job_failure_counts.get("clearme", 0) == 0
    assert "clearme" not in idle_scheduler._job_half_open_used


# ── H2: Signal keepalive ────────────────────────────────────────────────


@pytest.fixture
def keepalive(tmp_path, monkeypatch):
    from app.healing.monitors import signal_keepalive
    from app.healing.handlers import _common as _h_common

    monkeypatch.setattr(_h_common, "_STATE_DIR", tmp_path / "self_heal")
    sent: list[str] = []
    monkeypatch.setattr(
        signal_keepalive, "send_signal_alert",
        lambda body, **kw: sent.append(body) or True,
    )
    monkeypatch.setattr(signal_keepalive, "audit_event", lambda *a, **k: None)
    yield tmp_path, sent


def test_keepalive_sends_first_run(keepalive, monkeypatch):
    tmp_path, sent = keepalive
    from app.healing.monitors import signal_keepalive
    sends: list[tuple[str, str]] = []
    def fake_send(rcpt, txt):
        sends.append((rcpt, txt))
        return 1715200000123  # Signal-cli ts
    monkeypatch.setattr(
        "app.signal_client.send_message_blocking", fake_send,
    )
    class _S:
        signal_owner_number = "+10000000001"
    monkeypatch.setattr("app.config.get_settings", lambda: _S())

    summary = signal_keepalive.run()
    assert summary["sent"] is True
    assert len(sends) == 1
    assert "[andrusai-keepalive]" in sends[0][1]


def test_keepalive_dedup_within_30_days(keepalive, monkeypatch):
    tmp_path, sent = keepalive
    from app.healing.monitors import signal_keepalive
    sends: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "app.signal_client.send_message_blocking",
        lambda r, t: (sends.append((r, t)) or 1) and 1,
    )
    class _S:
        signal_owner_number = "+10000000001"
    monkeypatch.setattr("app.config.get_settings", lambda: _S())

    signal_keepalive.run()
    initial = len(sends)

    # Reset cadence; same window — no second keepalive.
    state_path = tmp_path / "self_heal" / "signal_keepalive.json"
    state = json.loads(state_path.read_text())
    state["last_run_at"] = 0.0
    state_path.write_text(json.dumps(state))
    signal_keepalive.run()
    assert len(sends) == initial


def test_keepalive_alerts_after_3_consecutive_failures(keepalive, monkeypatch):
    tmp_path, sent = keepalive
    from app.healing.monitors import signal_keepalive
    monkeypatch.setattr(
        "app.signal_client.send_message_blocking", lambda r, t: None,
    )
    class _S:
        signal_owner_number = "+10000000001"
    monkeypatch.setattr("app.config.get_settings", lambda: _S())

    # First 2 failures: silent.
    signal_keepalive.run()
    state_path = tmp_path / "self_heal" / "signal_keepalive.json"
    for _ in range(2):
        state = json.loads(state_path.read_text())
        state["last_run_at"] = 0.0
        state_path.write_text(json.dumps(state))
        signal_keepalive.run()
    # 3rd failure should trigger Signal alert.
    assert any("Signal keepalive failed" in s for s in sent)


def test_keepalive_disabled(monkeypatch, keepalive):
    monkeypatch.setenv("SIGNAL_KEEPALIVE_ENABLED", "0")
    from app.healing.monitors import signal_keepalive
    summary = signal_keepalive.run()
    assert summary["ran"] is False


# ── H1: restore-drill freshness monitor ─────────────────────────────────


@pytest.fixture
def drill(tmp_path, monkeypatch):
    from app.healing.monitors import restore_drill
    from app.healing.handlers import _common as _h_common

    monkeypatch.setattr(_h_common, "_STATE_DIR", tmp_path / "self_heal")
    monkeypatch.setattr(restore_drill, "_MANIFEST_PATH",
                        tmp_path / "restore_drill_manifest.json")
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        restore_drill, "send_signal_alert",
        lambda body, tag=None, **kw: sent.append((body, tag or "")),
    )
    monkeypatch.setattr(restore_drill, "audit_event", lambda *a, **k: None)
    yield tmp_path, sent


def test_drill_alerts_when_manifest_missing(drill):
    tmp_path, sent = drill
    from app.healing.monitors import restore_drill
    summary = restore_drill.run()
    assert summary["ran"] is True
    assert summary["manifest_present"] is False
    assert summary["alert_fired"] is True
    assert any("never" in body.lower() or "no drill manifest" in body.lower()
               for body, _ in sent)


def test_drill_alerts_when_stale(drill):
    tmp_path, sent = drill
    from app.healing.monitors import restore_drill
    old_iso = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
    (tmp_path / "restore_drill_manifest.json").write_text(json.dumps({
        "runs": [{"ts": old_iso, "all_ok": True}],
        "last_drill_at": old_iso,
        "last_drill_ok": True,
    }))
    summary = restore_drill.run()
    assert summary["alert_fired"] is True
    assert any("days ago" in body.lower() or "old" in body.lower()
               for body, _ in sent)


def test_drill_alerts_when_last_failed(drill):
    tmp_path, sent = drill
    from app.healing.monitors import restore_drill
    recent_iso = datetime.now(timezone.utc).isoformat()
    (tmp_path / "restore_drill_manifest.json").write_text(json.dumps({
        "runs": [{"ts": recent_iso, "all_ok": False}],
        "last_drill_at": recent_iso,
        "last_drill_ok": False,
    }))
    summary = restore_drill.run()
    assert summary["alert_fired"] is True
    assert any("FAILED" in body for body, _ in sent)


def test_drill_quiet_when_recent_ok(drill):
    tmp_path, sent = drill
    from app.healing.monitors import restore_drill
    recent_iso = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    (tmp_path / "restore_drill_manifest.json").write_text(json.dumps({
        "runs": [{"ts": recent_iso, "all_ok": True}],
        "last_drill_at": recent_iso,
        "last_drill_ok": True,
    }))
    summary = restore_drill.run()
    assert summary["alert_fired"] is False
    assert sent == []


def test_drill_dedup_within_window(drill):
    tmp_path, sent = drill
    from app.healing.monitors import restore_drill
    summary = restore_drill.run()  # alerts (manifest missing)
    initial = len(sent)
    state_path = tmp_path / "self_heal" / "restore_drill_monitor.json"
    state = json.loads(state_path.read_text())
    state["last_run_at"] = 0.0
    state_path.write_text(json.dumps(state))
    restore_drill.run()
    assert len(sent) == initial  # 14-day dedup
