"""Tests for app.inbox.scheduler — the notify-on-failure surfacing."""
from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from app.inbox import scheduler


@pytest.fixture
def inbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("INBOX_INGESTION_ENABLED", "true")
    monkeypatch.setenv("INBOX_DIR", str(tmp_path / "inbox"))
    p = tmp_path / "inbox"
    p.mkdir()
    return p


def test_get_idle_jobs_returns_inbox_tick() -> None:
    jobs = scheduler.get_idle_jobs()
    assert len(jobs) == 1
    assert jobs[0][0] == "inbox-tick"


def test_run_inbox_tick_no_files_no_notify(inbox: Path) -> None:
    """Empty inbox → no notify call."""
    captured: list[tuple] = []

    def fake_notify(title, body, **kwargs):
        captured.append((title, body))
        return {}

    with patch("app.notify.notify", side_effect=fake_notify):
        scheduler.run_inbox_tick()

    assert captured == []


def test_run_inbox_tick_notifies_on_failure(
    inbox: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dropped PDF (recognised, no handler) triggers a Signal ping."""
    f = inbox / "vacation.pdf"
    f.write_bytes(b"%PDF-1.4\n" + b"\x00" * 100)
    past = time.time() - 60
    os.utime(f, (past, past))

    captured: list[tuple] = []

    def fake_notify(title, body, **kwargs):
        captured.append((title, body))
        return {}

    with patch("app.notify.notify", side_effect=fake_notify):
        scheduler.run_inbox_tick()

    assert len(captured) == 1
    title, body = captured[0]
    assert "need attention" in title or "Inbox" in title
    assert "vacation.pdf" in body


def test_run_inbox_tick_notifies_on_unknown_extension(
    inbox: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrecognised extension triggers a ping."""
    f = inbox / "weird.xyz"
    f.write_bytes(b"\x00" * 32)
    past = time.time() - 60
    os.utime(f, (past, past))

    captured: list[tuple] = []

    def fake_notify(title, body, **kwargs):
        captured.append((title, body))
        return {}

    with patch("app.notify.notify", side_effect=fake_notify):
        scheduler.run_inbox_tick()

    assert len(captured) == 1
    title, body = captured[0]
    assert "weird.xyz" in body
    assert "unrecognised" in body


def test_run_inbox_tick_text_handler_does_not_notify(
    inbox: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful text drop does NOT trigger a notify (push spam)."""
    monkeypatch.setenv("INBOX_NOTES_DIR", str(tmp_path / "notes"))
    f = inbox / "note.md"
    f.write_text("hello", encoding="utf-8")
    past = time.time() - 60
    os.utime(f, (past, past))

    captured: list[tuple] = []

    def fake_notify(title, body, **kwargs):
        captured.append((title, body))
        return {}

    with patch("app.notify.notify", side_effect=fake_notify):
        scheduler.run_inbox_tick()

    # The text drop should have been processed, but should NOT have
    # triggered a notify ping.
    assert captured == []


def test_run_inbox_tick_apple_health_does_notify(
    inbox: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful Apple Health import IS pinged (notable success)."""
    monkeypatch.setenv("HEALTH_INGESTION_ENABLED", "true")
    monkeypatch.setenv("HEALTH_BASE_DIR", str(tmp_path / "health"))

    import zipfile
    src = inbox / "apple_health_export.zip"
    xml_dir = tmp_path / "xml" / "apple_health_export"
    xml_dir.mkdir(parents=True)
    (xml_dir / "export.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<HealthData>'
        '<Record type="HKQuantityTypeIdentifierStepCount" '
        '        sourceName="iPhone" sourceVersion="u1" '
        '        startDate="2026-05-10 08:00:00 +0300" '
        '        endDate="2026-05-10 08:15:00 +0300" '
        '        value="500"/>'
        '</HealthData>',
        encoding="utf-8",
    )
    with zipfile.ZipFile(src, "w") as zf:
        zf.write(xml_dir / "export.xml", "apple_health_export/export.xml")

    past = time.time() - 60
    os.utime(src, (past, past))

    captured: list[tuple] = []

    def fake_notify(title, body, **kwargs):
        captured.append((title, body))
        return {}

    with patch("app.notify.notify", side_effect=fake_notify):
        scheduler.run_inbox_tick()

    assert len(captured) == 1
    title, body = captured[0]
    assert "apple_health_export.zip" in body
    assert "imported" in body
