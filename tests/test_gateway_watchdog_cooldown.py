"""C3 (2026-05-18) — watchdog cooldown persists across crashes.

Pre-fix: ``last_restart_at`` was a local variable in ``main()``. A
watchdog crash + launchd respawn reset the cooldown to ``None``, so a
fresh watchdog could restart the gateway again immediately on the next
failure threshold — defeating the anti-thrashing intent of
``RESTART_COOLDOWN_SECONDS``.

The fix persists ``last_restart_at`` to ``STATE_PATH`` after every
restart and restores it on startup. Stale state (older than the
cooldown window) is dropped on load. Failure to read / write the state
file degrades gracefully back to pre-C3 behavior.

The watchdog is a host-side script (lives under ``scripts/``), uses
``requests``, and isn't normally on the test import path. We load it
via importlib from path with a ``requests`` stub, the same pattern as
``test_forwarder_outbox.py``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import pytest


def _load_watchdog(monkeypatch, tmp_path):
    """Load scripts/gateway_watchdog.py with a stubbed requests + a
    tmp_path-rooted STATE_PATH so tests are hermetic."""
    if "requests" not in sys.modules:
        fake = types.ModuleType("requests")
        class _Session:
            # Mirrors enough of requests.Session for both this test
            # and tests/test_forwarder_outbox.py — pytest test order
            # determines which test caches the stub first.
            def __init__(self):
                self.headers: dict = {}
            def get(self, *a, **kw):
                raise RuntimeError("network unavailable in test")
            def post(self, *a, **kw):
                raise RuntimeError("network unavailable in test")
        class _RequestException(Exception):
            pass
        exceptions = types.ModuleType("requests.exceptions")
        exceptions.RequestException = _RequestException
        fake.Session = _Session
        fake.exceptions = exceptions
        sys.modules["requests"] = fake
        sys.modules["requests.exceptions"] = exceptions

    state_file = tmp_path / "state.json"
    monkeypatch.setenv("STATE_PATH", str(state_file))
    # Short cooldown for deterministic tests.
    monkeypatch.setenv("RESTART_COOLDOWN_SECONDS", "300")

    path = Path(__file__).parent.parent / "scripts" / "gateway_watchdog.py"
    spec = importlib.util.spec_from_file_location("_test_watchdog", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_test_watchdog"] = mod
    spec.loader.exec_module(mod)
    return mod, state_file


def test_load_returns_none_when_no_state_file(monkeypatch, tmp_path):
    mod, _ = _load_watchdog(monkeypatch, tmp_path)
    assert mod._load_restart_state() is None


def test_save_then_load_round_trips(monkeypatch, tmp_path):
    mod, state_file = _load_watchdog(monkeypatch, tmp_path)
    now = time.time()
    mod._save_restart_state(now)
    assert state_file.exists()
    loaded = mod._load_restart_state()
    assert loaded == pytest.approx(now, abs=1.0)


def test_load_drops_stale_state_past_cooldown(monkeypatch, tmp_path):
    """A state file older than RESTART_COOLDOWN can't gate anything;
    returning the stale value would mis-report cooldown remaining."""
    mod, state_file = _load_watchdog(monkeypatch, tmp_path)
    stale = time.time() - (mod.RESTART_COOLDOWN + 60)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        json.dumps({"last_restart_at": stale}), encoding="utf-8",
    )
    assert mod._load_restart_state() is None


def test_load_keeps_fresh_state_inside_cooldown(monkeypatch, tmp_path):
    mod, state_file = _load_watchdog(monkeypatch, tmp_path)
    fresh = time.time() - (mod.RESTART_COOLDOWN / 2)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        json.dumps({"last_restart_at": fresh}), encoding="utf-8",
    )
    loaded = mod._load_restart_state()
    assert loaded is not None
    assert loaded == pytest.approx(fresh, abs=1.0)


def test_load_swallows_corrupt_state_file(monkeypatch, tmp_path):
    mod, state_file = _load_watchdog(monkeypatch, tmp_path)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text("not valid json {{{", encoding="utf-8")
    # Must not raise; returns None (degrades to pre-C3 behavior).
    assert mod._load_restart_state() is None


def test_load_swallows_wrong_shape(monkeypatch, tmp_path):
    """Defensive: a JSON file with the wrong shape (e.g. someone
    mangled the file) shouldn't crash the watchdog."""
    mod, state_file = _load_watchdog(monkeypatch, tmp_path)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        json.dumps({"last_restart_at": "not a number"}), encoding="utf-8",
    )
    assert mod._load_restart_state() is None


def test_save_uses_atomic_replace(monkeypatch, tmp_path):
    """The save path uses os.replace via a .tmp sidecar — no
    half-written state file ever appears at STATE_PATH."""
    mod, state_file = _load_watchdog(monkeypatch, tmp_path)
    mod._save_restart_state(time.time())
    assert state_file.exists()
    # No .tmp sidecar left around.
    assert not (state_file.parent / (state_file.name + ".tmp")).exists()
