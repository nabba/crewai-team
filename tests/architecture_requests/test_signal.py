"""Tests for app.architecture_requests.signal."""

from __future__ import annotations

from app.architecture_requests import lifecycle, signal, store
from app.architecture_requests.models import FileSpec, IntegrationPoint
from app.architecture_requests.signal import (
    build_ask_body,
    find_request_by_signal_ts,
)
from .conftest import make_request


def test_build_ask_body_includes_intent_and_package_path() -> None:
    req = make_request(intent="Add weekly philosophical inquiry pass")
    body = build_ask_body(req)
    assert "ARCHITECTURE REQUEST" in body
    assert req.package_path in body
    assert "Add weekly philosophical inquiry pass" in body
    assert "👍" in body and "👎" in body
    assert req.id in body


def test_build_ask_body_truncates_long_motivation() -> None:
    long_text = "X" * 2000
    req = make_request()
    object.__setattr__(req, "motivation", long_text)  # bypass dataclass mutability
    req.motivation = long_text
    body = build_ask_body(req)
    assert "[motivation truncated]" in body


def test_build_ask_body_caps_file_layout() -> None:
    files = [
        FileSpec(path=f"app/inquiry/file_{i}.py", purpose=f"file {i}")
        for i in range(20)
    ]
    req = make_request(file_layout=files)
    body = build_ask_body(req)
    # First six rendered, rest collapsed.
    assert "file_0.py" in body
    assert "file_5.py" in body
    assert "file_6.py" not in body
    assert "+14 more" in body


def test_build_ask_body_handles_no_integration_points() -> None:
    req = make_request(integration_points=[])
    body = build_ask_body(req)
    assert "Integration points: (none)" in body


def test_build_ask_body_handles_no_env_switches() -> None:
    req = make_request(env_switches={})
    body = build_ask_body(req)
    assert "Env switches: (none)" in body


def test_find_request_by_signal_ts_round_trip() -> None:
    # Create through lifecycle so the request lands in the store.
    req = lifecycle.create_request(
        requestor="self_improver",
        intent="Add inquiry",
        motivation="Reasons.",
        package_path="app/inquiry/",
        file_layout=[FileSpec(path="app/inquiry/__init__.py", purpose="x")],
        integration_points=[
            IntegrationPoint(
                kind="idle_job_registration",
                target_module="app/idle_scheduler.py",
            ),
        ],
        env_switches={"INQUIRY_X_ENABLED": "true"},
        test_plan="ok",
    )
    lifecycle.attach_signal_ts(req.id, 1747900000)
    assert find_request_by_signal_ts(1747900000) == req.id
    assert find_request_by_signal_ts(0) is None
    assert find_request_by_signal_ts(99999) is None


def test_attach_signal_ts_persists_on_existing_request() -> None:
    req = lifecycle.create_request(
        requestor="self_improver",
        intent="Add inquiry",
        motivation="Reasons.",
        package_path="app/inquiry/",
        file_layout=[FileSpec(path="app/inquiry/__init__.py", purpose="x")],
        integration_points=[],
        env_switches={},
        test_plan="ok",
    )
    lifecycle.attach_signal_ts(req.id, 999)
    reloaded = store.get(req.id)
    assert reloaded is not None
    assert reloaded.signal_message_ts == 999


def test_send_ask_skips_when_no_recipient(monkeypatch) -> None:
    """No SIGNAL_OWNER_NUMBER set → send_ask returns None silently."""
    req = lifecycle.create_request(
        requestor="self_improver",
        intent="Add inquiry",
        motivation="Reasons.",
        package_path="app/inquiry/",
        file_layout=[FileSpec(path="app/inquiry/__init__.py", purpose="x")],
        integration_points=[],
        env_switches={},
        test_plan="ok",
    )

    class _FakeSettings:
        signal_owner_number = ""

    monkeypatch.setattr("app.config.get_settings", lambda: _FakeSettings())
    result = signal.send_ask(req.id)
    assert result is None


def test_send_ask_returns_none_for_unknown_request() -> None:
    assert signal.send_ask("nonexistent-id") is None
