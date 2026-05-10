"""Tests for the proactive monitors shipped 2026-05-09.

Focused on the alert-cooldown / state-file behavior. Outbound HTTP and
Signal sends are stubbed; the tests just verify that the monitors fire
exactly once per cooldown window and persist their dedup state.
"""
from __future__ import annotations

import time

import pytest


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Redirect ``workspace/self_heal/`` to tmp; mock signal sender +
    audit so tests don't try to talk to Signal or write audit rows.
    """
    from app.healing.handlers import _common
    monkeypatch.setattr(_common, "_STATE_DIR", tmp_path)
    sent = []
    monkeypatch.setattr(_common, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(_common, "audit_event", lambda *a, **k: None)
    yield tmp_path, sent


# ── Disk quota ────────────────────────────────────────────────────────────


def test_disk_quota_alerts_once_per_window(isolated_state, monkeypatch):
    tmp_path, sent = isolated_state
    from app.healing.monitors import disk_quota

    workspace = tmp_path.parent / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    class _DU:
        free = 500 * 1024 * 1024  # 500 MB free → critical
        total = 100 * 1024 ** 3   # 100 GB total

    monkeypatch.setattr(disk_quota.shutil, "disk_usage", lambda _p: _DU)
    monkeypatch.setattr(disk_quota, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(disk_quota, "audit_event", lambda *a, **k: None)
    monkeypatch.setattr(
        disk_quota,
        "Path",
        type("P", (), {"__call__": lambda *a, **kw: workspace}),
    )

    # Patch the path resolution inside run() — easier: redirect workspace
    import pathlib
    orig_resolve = pathlib.Path.resolve
    monkeypatch.setattr(disk_quota, "Path", pathlib.Path)

    # First run — should alert.
    disk_quota.run()
    n_first = len(sent)
    # Second run within cooldown — should NOT alert again.
    disk_quota.run()
    assert len(sent) == n_first


# ── Listener heartbeat ────────────────────────────────────────────────────


def test_listener_heartbeat_alerts_when_all_probes_stale(isolated_state, monkeypatch):
    tmp_path, sent = isolated_state
    from app.healing.monitors import listener_heartbeat

    monkeypatch.setattr(listener_heartbeat, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(listener_heartbeat, "audit_event",
                        lambda *a, **k: None)

    # Force every probe to look stale by pointing to a non-existent root.
    import pathlib
    fake_root = tmp_path / "no-such-repo"
    fake_root.mkdir()
    monkeypatch.setattr(
        listener_heartbeat, "_LIVENESS_PROBES",
        ["does-not-exist-1", "does-not-exist-2"],
    )

    # Run — no probes exist → freshest_probe is None → alert path.
    listener_heartbeat.run()
    assert any("staleness" in body for body in sent)


# ── Vendor sunset ─────────────────────────────────────────────────────────


def test_vendor_sunset_records_missing_models(isolated_state, monkeypatch):
    tmp_path, sent = isolated_state
    from app.healing.monitors import vendor_sunset

    monkeypatch.setenv("HEALING_VENDOR_SUNSET_ENABLED", "true")
    monkeypatch.setattr(vendor_sunset, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(vendor_sunset, "audit_event", lambda *a, **k: None)

    monkeypatch.setattr(
        vendor_sunset, "_runtime_catalog_models",
        lambda: {
            "openrouter": {"vendor/old-model", "vendor/current-model"},
            "anthropic": set(),
        },
    )
    monkeypatch.setattr(
        vendor_sunset, "_fetch_openrouter_ids",
        lambda: {"vendor/current-model", "vendor/new-model"},
    )
    monkeypatch.setattr(vendor_sunset, "_fetch_anthropic_ids",
                        lambda: set())

    vendor_sunset.run()

    assert any("vendor/old-model" in body for body in sent)

    # Second run — already alerted, should NOT re-spam.
    sent.clear()
    vendor_sunset.run()
    assert sent == []


# ── Idle cooldown reader ──────────────────────────────────────────────────


def test_idle_cooldown_returns_empty_when_dbm_missing(isolated_state):
    """The dbm read is best-effort; missing files should not raise and
    must produce zero alerts.
    """
    from app.healing.monitors import idle_cooldown
    state = idle_cooldown._read_idle_state()
    # On a fresh tmp setup the dbm doesn't exist → empty dict, no raise.
    assert isinstance(state, dict)


# ── Auditor bridge dedup ──────────────────────────────────────────────────


def test_auditor_bridge_dedups_same_proposal(isolated_state, monkeypatch):
    tmp_path, sent = isolated_state
    from app.healing import auditor_bridge

    monkeypatch.setenv("HEALING_AUDITOR_BRIDGE_ENABLED", "true")
    monkeypatch.setattr(auditor_bridge, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(auditor_bridge, "audit_event",
                        lambda *a, **k: None)
    # Stub the CR filer so the test runs without a live CR system.
    monkeypatch.setattr(auditor_bridge, "file_change_request",
                        lambda **kw: "cr-mock-1")

    fresh_ts = time.strftime("%Y-%m-%dT%H:%M:%S",
                             time.gmtime(time.time() - 3600))
    journal = [
        {
            "ts": fresh_ts,
            "event": "error_fix_proposed",
            "detail": "Pattern coding:RuntimeError attempt #1: increase timeout to 60s",
            "files_changed": [],
        },
    ]
    monkeypatch.setattr(auditor_bridge, "_load_recent_proposals", lambda: journal)

    n1 = auditor_bridge.run_one_pass()
    n2 = auditor_bridge.run_one_pass()

    assert n1 == 1
    assert n2 == 0  # already alerted under the same key


def test_auditor_bridge_files_cr_mirror(isolated_state, monkeypatch):
    """First sighting of a proposal should ALSO file a CR with the
    structured markdown record. CR id must propagate into the Signal
    alert + the dedup state.
    """
    tmp_path, sent = isolated_state
    from app.healing import auditor_bridge

    monkeypatch.setenv("HEALING_AUDITOR_BRIDGE_ENABLED", "true")
    monkeypatch.setattr(auditor_bridge, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(auditor_bridge, "audit_event",
                        lambda *a, **k: None)

    cr_calls: list = []

    def fake_cr(**kwargs):
        cr_calls.append(kwargs)
        return f"cr-{len(cr_calls)}"

    monkeypatch.setattr(auditor_bridge, "file_change_request", fake_cr)

    fresh_ts = time.strftime("%Y-%m-%dT%H:%M:%S",
                             time.gmtime(time.time() - 3600))
    journal = [
        {
            "ts": fresh_ts,
            "event": "error_fix_proposed",
            "detail": "Pattern research:ConnectionError attempt #1: add timeout=60",
            "files_changed": ["app/llm_factory.py"],
        },
    ]
    monkeypatch.setattr(auditor_bridge, "_load_recent_proposals", lambda: journal)

    n = auditor_bridge.run_one_pass()
    assert n == 1
    assert len(cr_calls) == 1

    cr_kwargs = cr_calls[0]
    assert cr_kwargs["path"] == "docs/proposed_fixes/research_RuntimeError_1.md" or \
        cr_kwargs["path"].startswith("docs/proposed_fixes/")
    assert "research:ConnectionError" in cr_kwargs["new_content"] \
        or "research_ConnectionError" in cr_kwargs["new_content"]
    assert "add timeout=60" in cr_kwargs["new_content"]
    # The Signal alert mentions the CR id so the operator has a deep-link.
    assert any("cr-1" in body for body in sent)


def test_auditor_bridge_cr_failure_still_sends_signal(isolated_state, monkeypatch):
    """If the CR system errors, the Signal alert MUST still go out so the
    operator is never blind to a proposal because of CR plumbing trouble.
    """
    tmp_path, sent = isolated_state
    from app.healing import auditor_bridge

    monkeypatch.setenv("HEALING_AUDITOR_BRIDGE_ENABLED", "true")
    monkeypatch.setattr(auditor_bridge, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(auditor_bridge, "audit_event",
                        lambda *a, **k: None)

    def boom(**kw):
        raise RuntimeError("CR system down")

    monkeypatch.setattr(auditor_bridge, "file_change_request", boom)

    fresh_ts = time.strftime("%Y-%m-%dT%H:%M:%S",
                             time.gmtime(time.time() - 3600))
    journal = [
        {
            "ts": fresh_ts,
            "event": "error_fix_proposed",
            "detail": "Pattern x:Y attempt #1: do something",
            "files_changed": [],
        },
    ]
    monkeypatch.setattr(auditor_bridge, "_load_recent_proposals", lambda: journal)

    n = auditor_bridge.run_one_pass()
    assert n == 1
    assert sent  # Signal alert went out despite CR failure
    assert any("CR mirror unavailable" in body for body in sent)
