"""Tests for the vendor_sunset CR-filing path (Wave 0/1 #A4).

The pre-existing alert path is already exercised by other suites.
This file targets the new behavior: when a sunset finding is detected,
file a change-request to update ``workspace/healing/sunset_models.json``
so the LLM router stops selecting the deprecated model on its own.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.healing.monitors import vendor_sunset
    from app.healing.handlers import _common as _h_common

    monkeypatch.setattr(_h_common, "_STATE_DIR", tmp_path / "self_heal")
    monkeypatch.setattr(vendor_sunset, "audit_event", lambda *a, **k: None)

    sent: list[str] = []
    monkeypatch.setattr(vendor_sunset, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)

    yield tmp_path, sent


def test_file_sunset_cr_calls_change_request(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.healing.monitors import vendor_sunset

    captured = {}

    def fake_file_change_request(**kwargs):
        captured.update(kwargs)
        return "cr-abc123"

    from app.healing.handlers import _common as _h_common
    monkeypatch.setattr(_h_common, "file_change_request",
                        fake_file_change_request)

    cr_id = vendor_sunset._file_sunset_cr({
        "provider": "openai",
        "model": "gpt-3.5-turbo-0301",
        "first_missed_at": 1700000000,
    })
    assert cr_id == "cr-abc123"
    assert captured["path"].endswith("sunset_models.json")
    assert "gpt-3.5-turbo-0301" in captured["new_content"]
    assert "openai" in captured["new_content"]
    assert "vendor_sunset" in captured["requestor"]


def test_file_sunset_cr_skips_when_already_blocked(isolated, monkeypatch):
    """If the model is already in the blocklist, don't refile."""
    tmp_path, sent = isolated
    from app.healing.monitors import vendor_sunset
    import app.healing.monitors.vendor_sunset as vs_mod
    from pathlib import Path as _Path

    # Pre-seed the blocklist.
    blocklist_dir = tmp_path / "workspace" / "healing"
    blocklist_dir.mkdir(parents=True)
    blocklist = blocklist_dir / "sunset_models.json"
    blocklist.write_text(json.dumps({
        "sunset": [
            {"provider": "openai", "model": "gpt-3.5-turbo-0301"},
        ],
    }))

    # The function reads from a hardcoded `/app/workspace/...` path; we
    # need to redirect Path("/app/workspace/healing/sunset_models.json")
    # to our temp file. Easiest: monkeypatch Path in the module.
    real_path = _Path
    def patched_path(s):
        if isinstance(s, str) and s == "/app/workspace/healing/sunset_models.json":
            return blocklist
        if isinstance(s, str) and s == "/app/workspace/healing/vendor_sunset_replacements.json":
            return tmp_path / "no-such-file"
        return real_path(s)
    monkeypatch.setattr(vs_mod, "Path", patched_path)

    captured = {}
    def fake_file_change_request(**kwargs):
        captured.update(kwargs)
        return "cr-shouldnt-fire"

    from app.healing.handlers import _common as _h_common
    monkeypatch.setattr(_h_common, "file_change_request",
                        fake_file_change_request)

    cr_id = vendor_sunset._file_sunset_cr({
        "provider": "openai",
        "model": "gpt-3.5-turbo-0301",
    })
    assert cr_id is None
    assert captured == {}  # no CR filed
