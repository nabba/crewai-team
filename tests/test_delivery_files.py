"""
test_delivery_files — files API + Discord routing + send helpers.

Smoke coverage. Does NOT touch real Signal-cli, SMTP, or Discord — every
external dependency is mocked.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _redirect_workspace(tmp_path, monkeypatch):
    """Point app.paths.WORKSPACE_ROOT at a tmp dir so tests don't touch real
    workspace artifacts. Patch the references already imported by modules."""
    from app import paths
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", tmp_path, raising=True)
    # Also patch already-imported references in modules under test.
    import app.api.files_api as files_api
    import app.delivery.email_send as email_send  # noqa: F401 — only to ensure module loaded
    monkeypatch.setattr(files_api, "WORKSPACE_ROOT", tmp_path, raising=True)
    # Reset the per-root path map so the fresh tmp dir is the new base.
    monkeypatch.setattr(files_api, "_ROOTS", {
        "output": tmp_path / "output",
        "skills": tmp_path / "skills",
        "notes":  tmp_path / "notes",
    }, raising=True)
    yield tmp_path


# ── Files API listing ─────────────────────────────────────────────────────

def test_list_files_groups_by_root(_redirect_workspace):
    from app.api.files_api import list_files_endpoint
    root = _redirect_workspace
    (root / "output").mkdir()
    (root / "skills").mkdir()
    (root / "output" / "report.pdf").write_bytes(b"%PDF-1.4 test")
    (root / "output" / "deck.pptx").write_bytes(b"PK fake pptx")
    (root / "skills" / "foo.md").write_text("# foo")
    # File with non-listed extension — should be filtered out.
    (root / "output" / "ignore.tmp").write_text("nope")

    import asyncio
    out = asyncio.run(list_files_endpoint())
    assert {e["name"] for e in out["roots"]["output"]} == {"report.pdf", "deck.pptx"}
    assert {e["name"] for e in out["roots"]["skills"]} == {"foo.md"}


def test_list_files_handles_missing_root(_redirect_workspace):
    """When a root dir doesn't exist, the response still returns it as []."""
    import asyncio
    from app.api.files_api import list_files_endpoint
    out = asyncio.run(list_files_endpoint())
    assert out["roots"]["output"] == []
    assert out["roots"]["notes"] == []


# ── Path traversal guard ──────────────────────────────────────────────────

def test_safe_path_rejects_escape(_redirect_workspace):
    from app.api.files_api import _safe_path_for_download
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        _safe_path_for_download("../etc/passwd")
    assert exc.value.status_code == 400


def test_safe_path_rejects_missing_file(_redirect_workspace):
    from app.api.files_api import _safe_path_for_download
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        _safe_path_for_download("output/does-not-exist.pdf")
    assert exc.value.status_code == 404


def test_safe_path_resolves_valid_file(_redirect_workspace):
    from app.api.files_api import _safe_path_for_download
    out = _redirect_workspace / "output"
    out.mkdir()
    target = out / "ok.pdf"
    target.write_bytes(b"PDF bytes")
    p = _safe_path_for_download("output/ok.pdf")
    assert p == target.resolve()


# ── Send-file endpoint dispatch ───────────────────────────────────────────

@pytest.fixture
def _mock_request():
    """A FastAPI Request with a stub gateway-secret check that always passes."""
    req = MagicMock()
    return req


@pytest.fixture(autouse=True)
def _bypass_gateway_secret(monkeypatch):
    monkeypatch.setattr(
        "app.api.config_api.verify_gateway_secret", lambda request: True,
    )


def test_send_endpoint_routes_signal(monkeypatch, _redirect_workspace, _mock_request):
    out = _redirect_workspace / "output"
    out.mkdir()
    (out / "report.pdf").write_bytes(b"PDF")

    captured: dict = {}
    def fake_signal(paths, body=""):
        captured["paths"] = list(paths)
        captured["body"] = body
        return True, "Signal delivered: 1 file"

    monkeypatch.setattr("app.delivery.send_via_signal", fake_signal)
    _mock_request.json = MagicMock()

    async def run():
        _mock_request.json = lambda: {"channel": "signal", "path": "output/report.pdf", "body": "Here you go"}
        # Make request.json awaitable
        import inspect
        if not inspect.iscoroutinefunction(_mock_request.json):
            real = _mock_request.json
            async def _awaitable():
                return real()
            _mock_request.json = _awaitable
        from app.api.files_api import send_endpoint
        return await send_endpoint(_mock_request)

    import asyncio
    res = asyncio.run(run())
    assert res["status"] == "ok"
    assert "Signal delivered" in res["detail"]
    assert captured["body"] == "Here you go"


def test_send_endpoint_routes_email_requires_to(monkeypatch, _redirect_workspace, _mock_request):
    out = _redirect_workspace / "output"
    out.mkdir()
    (out / "report.pdf").write_bytes(b"PDF")

    async def run(payload):
        async def _json():
            return payload
        _mock_request.json = _json
        from app.api.files_api import send_endpoint
        return await send_endpoint(_mock_request)

    from fastapi import HTTPException
    import asyncio
    with pytest.raises(HTTPException) as exc:
        asyncio.run(run({"channel": "email", "path": "output/report.pdf"}))
    assert exc.value.status_code == 400
    assert "needs `to`" in exc.value.detail


def test_send_endpoint_routes_email_calls_helper(monkeypatch, _redirect_workspace, _mock_request):
    out = _redirect_workspace / "output"
    out.mkdir()
    (out / "report.pdf").write_bytes(b"PDF")

    captured: dict = {}
    def fake_email(to, subject, body, attachment_paths):
        captured["to"] = to
        captured["subject"] = subject
        captured["paths"] = list(attachment_paths)
        return True, f"sent to {to}"
    monkeypatch.setattr("app.delivery.send_via_email", fake_email)

    async def run(payload):
        async def _json():
            return payload
        _mock_request.json = _json
        from app.api.files_api import send_endpoint
        return await send_endpoint(_mock_request)

    import asyncio
    res = asyncio.run(run({
        "channel": "email", "path": "output/report.pdf",
        "to": "alice@example.com", "subject": "FYI", "body": "Attached.",
    }))
    assert res["status"] == "ok"
    assert captured["to"] == "alice@example.com"
    assert captured["subject"] == "FYI"


def test_send_endpoint_unknown_channel(_redirect_workspace, _mock_request):
    out = _redirect_workspace / "output"
    out.mkdir()
    (out / "report.pdf").write_bytes(b"PDF")

    async def run():
        async def _json():
            return {"channel": "carrier_pigeon", "path": "output/report.pdf"}
        _mock_request.json = _json
        from app.api.files_api import send_endpoint
        return await send_endpoint(_mock_request)

    from fastapi import HTTPException
    import asyncio
    with pytest.raises(HTTPException) as exc:
        asyncio.run(run())
    assert exc.value.status_code == 400


# ── Email send helper ─────────────────────────────────────────────────────

def test_email_send_disabled_when_email_off(monkeypatch):
    from app.delivery.email_send import send_via_email
    fake_settings = MagicMock()
    fake_settings.email_enabled = False
    monkeypatch.setattr("app.delivery.email_send.get_settings",
                        lambda: fake_settings, raising=False)
    # Direct import path is "app.config.get_settings" but the module uses
    # `from app.config import get_settings` then calls it inside the
    # function — patch that reference too.
    monkeypatch.setattr("app.config.get_settings",
                        lambda: fake_settings, raising=False)
    ok, detail = send_via_email("a@b.com", "s", "b")
    assert ok is False
    assert "EMAIL_ENABLED" in detail


def test_email_send_rejects_invalid_recipient(monkeypatch):
    from app.delivery.email_send import send_via_email
    fake = MagicMock()
    fake.email_enabled = True
    fake.email_smtp_host = "smtp.example.com"
    fake.email_smtp_port = 587
    fake.email_address = "me@example.com"
    fake.email_password = MagicMock()
    fake.email_password.get_secret_value = lambda: "pw"
    monkeypatch.setattr("app.config.get_settings", lambda: fake, raising=False)
    ok, detail = send_via_email("not-an-email", "s", "b")
    assert ok is False
    assert "invalid recipient" in detail


def test_email_send_attaches_files(monkeypatch, tmp_path):
    """Capture the SMTP calls so we can assert the attachment landed."""
    from app.delivery import email_send

    fake = MagicMock()
    fake.email_enabled = True
    fake.email_smtp_host = "smtp.example.com"
    fake.email_smtp_port = 587
    fake.email_address = "me@example.com"
    fake.email_password = MagicMock()
    fake.email_password.get_secret_value = lambda: "pw"
    monkeypatch.setattr("app.config.get_settings", lambda: fake, raising=False)

    sent_msg = {}

    class FakeSMTP:
        def __init__(self, host, port):
            sent_msg["host"] = host
            sent_msg["port"] = port

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def ehlo(self): pass
        def starttls(self): pass
        def login(self, user, pw): sent_msg["login"] = (user, pw)
        def send_message(self, msg): sent_msg["msg"] = msg

    monkeypatch.setattr(email_send.smtplib, "SMTP", FakeSMTP)

    p = tmp_path / "report.pdf"
    p.write_bytes(b"%PDF fake")
    ok, detail = email_send.send_via_email(
        "alice@example.com", "Test", "Body",
        attachment_paths=[p],
    )
    assert ok is True
    assert sent_msg["host"] == "smtp.example.com"
    # The MIMEMultipart should have at least 2 parts: body + attachment.
    msg = sent_msg["msg"]
    parts = list(msg.walk())
    # walk() yields the multipart container plus children — at least 3 entries
    # when there's body alternative + attachment.
    assert any("application/pdf" in (p.get_content_type() or "") for p in parts) or \
           any(p.get_filename() == "report.pdf" for p in parts)


# ── Discord routing ───────────────────────────────────────────────────────

def test_send_via_discord_no_client():
    """Without a running bot, send_via_discord returns a clean failure."""
    from app.discord_client import send_via_discord
    ok, detail = send_via_discord("12345", "hello")
    assert ok is False
    assert "not running" in detail.lower()


def test_send_via_discord_validates_paths(tmp_path, monkeypatch):
    from app.discord_client.sender import _validate_paths
    p = tmp_path / "ok.pdf"
    p.write_bytes(b"x" * 100)
    assert _validate_paths([p]) == ""
    too_big = tmp_path / "big.pdf"
    too_big.write_bytes(b"x" * (9 * 1024 * 1024))
    err = _validate_paths([too_big])
    assert "8 MB" in err
    err2 = _validate_paths([tmp_path / "missing.pdf"])
    assert "not found" in err2


def test_discord_chunk_helper():
    from app.discord_client.sender import _chunk
    assert _chunk("short", 2000) == ["short"]
    long = ("Sentence one. " * 200).strip()
    chunks = _chunk(long, 100)
    assert all(len(c) <= 100 for c in chunks)
    # Reassembled chunks should preserve the original (modulo split-point trim).
    assert "".join(c.replace(" ", "").replace(".", "") for c in chunks) \
        .startswith("Sentenceone")


def test_discord_bot_disabled_no_token(monkeypatch):
    """start_bot is a clean no-op when DISCORD_BOT_TOKEN is empty."""
    import asyncio
    from app.discord_client import bot, is_running
    fake = MagicMock()
    fake.discord_enabled = True
    fake.discord_owner_id = "123"
    monkeypatch.setattr("app.discord_client.bot.get_settings",
                        lambda: fake, raising=False)
    monkeypatch.setattr("app.discord_client.bot.get_discord_bot_token",
                        lambda: "", raising=False)
    asyncio.run(bot.start_bot())
    assert is_running() is False


def test_discord_bot_disabled_when_flag_off(monkeypatch):
    """Even with a token, start_bot is a no-op when DISCORD_ENABLED is false."""
    import asyncio
    from app.discord_client import bot, is_running
    fake = MagicMock()
    fake.discord_enabled = False
    fake.discord_owner_id = "123"
    monkeypatch.setattr("app.discord_client.bot.get_settings",
                        lambda: fake, raising=False)
    monkeypatch.setattr("app.discord_client.bot.get_discord_bot_token",
                        lambda: "tok", raising=False)
    asyncio.run(bot.start_bot())
    assert is_running() is False
