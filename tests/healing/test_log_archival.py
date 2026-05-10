"""Tests for ``app.healing.monitors.log_archival`` (Wave 0/1 #A5)."""
from __future__ import annotations

import gzip
import os
import time

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.healing.monitors import log_archival
    from app.life_companion import _common

    monkeypatch.setattr(_common, "_STATE_DIR", tmp_path / "lc")
    monkeypatch.setattr(log_archival, "background_enabled", lambda: True)
    monkeypatch.setattr(log_archival, "audit_event", lambda *a, **k: None)

    monkeypatch.setattr(log_archival, "_ERRORS_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(log_archival, "_ERRORS_ARCHIVE_DIR", tmp_path / "logs/archive")
    monkeypatch.setattr(log_archival, "_AUDIT_ARCHIVE_DIR", tmp_path / "audit_archive")

    (tmp_path / "logs").mkdir()
    yield tmp_path


def test_archive_rotated_errors(isolated):
    from app.healing.monitors import log_archival

    rotated = isolated / "logs" / "errors.jsonl.1"
    rotated.write_text('{"err":"x"}\n')
    summary = log_archival._archive_errors_jsonl()
    assert summary["rotated_files"] == 1
    assert summary["bytes_archived"] > 0
    assert not rotated.exists()  # consumed
    archive_files = list((isolated / "logs/archive").glob("*.jsonl.gz"))
    assert len(archive_files) == 1
    # Round-trip check: data is recoverable.
    with gzip.open(archive_files[0], "rb") as f:
        assert b'{"err":"x"}' in f.read()


def test_live_errors_log_not_archived(isolated):
    from app.healing.monitors import log_archival

    live = isolated / "logs" / "errors.jsonl"
    live.write_text("live data\n")
    log_archival._archive_errors_jsonl()
    # Live file stays put.
    assert live.exists()


def test_audit_journal_archive_is_now_a_noop(isolated):
    """The audit journal moved to rolled-segment storage in C2; the
    archival shim returns the legacy summary shape so the telemetry
    tuple in :func:`run` keeps its keys, but performs no work."""
    from app.healing.monitors import log_archival

    summary = log_archival._archive_audit_journal()
    assert summary == {"rotated": False, "bytes_archived": 0}


def test_purge_old_archives(isolated):
    from app.healing.monitors import log_archival

    archive_dir = isolated / "logs/archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    fresh = archive_dir / "2026-05.jsonl.gz"
    stale = archive_dir / "2024-01.jsonl.gz"
    fresh.write_text("fresh")
    stale.write_text("stale")
    old_mtime = time.time() - 200 * 86400  # 200 days
    os.utime(stale, (old_mtime, old_mtime))

    summary = log_archival._purge_old_archives(retention_days=90)
    assert summary["deleted_files"] == 1
    assert fresh.exists()
    assert not stale.exists()


def test_run_respects_cadence(isolated):
    from app.healing.monitors import log_archival

    log_archival.run()
    # Second run within cadence — should be a no-op and not raise.
    log_archival.run()
