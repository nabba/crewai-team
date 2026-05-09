"""Tests for evolution-runs archival in log_archival (Phase D #2)."""
from __future__ import annotations

import os
import tarfile
import time

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.healing.monitors import log_archival
    from app.life_companion import _common

    monkeypatch.setattr(_common, "_STATE_DIR", tmp_path / "lc")
    monkeypatch.setattr(log_archival, "background_enabled", lambda: True)
    monkeypatch.setattr(log_archival, "audit_event", lambda *a, **k: None)

    monkeypatch.setattr(log_archival, "_EVOLUTION_RUNS_DIR",
                        tmp_path / "shinka_results")
    monkeypatch.setattr(log_archival, "_EVOLUTION_ARCHIVE_DIR",
                        tmp_path / "shinka_results" / "archive")
    yield tmp_path


def test_no_evolution_dir_no_op(isolated):
    from app.healing.monitors import log_archival
    summary = log_archival._archive_evolution_runs()
    assert summary["archived_runs"] == 0


def test_recent_run_skipped(isolated):
    from app.healing.monitors import log_archival
    runs = isolated / "shinka_results"
    runs.mkdir(parents=True)
    fresh = runs / "run_20260509_120000"
    fresh.mkdir()
    (fresh / "data.txt").write_text("recent")
    summary = log_archival._archive_evolution_runs()
    assert summary["archived_runs"] == 0
    assert fresh.exists()


def test_old_run_archived_and_removed(isolated):
    from app.healing.monitors import log_archival
    runs = isolated / "shinka_results"
    runs.mkdir(parents=True)
    old = runs / "run_20260101_120000"
    old.mkdir()
    (old / "log.txt").write_text("ancient" * 100)
    # Backdate mtime past the 90 day threshold.
    old_ts = time.time() - 100 * 86400
    os.utime(old, (old_ts, old_ts))
    os.utime(old / "log.txt", (old_ts, old_ts))

    summary = log_archival._archive_evolution_runs()
    assert summary["archived_runs"] == 1
    assert summary["bytes_archived"] > 0
    assert not old.exists()
    archive = isolated / "shinka_results" / "archive" / "run_20260101_120000.tar.gz"
    assert archive.exists()
    # Verify the tarball is intact and contains the original contents.
    with tarfile.open(archive, "r:gz") as tar:
        names = {m.name for m in tar.getmembers()}
    assert any(n.endswith("log.txt") for n in names)


def test_skips_archive_subdir(isolated):
    """The archive subdir itself should never be tarballed up."""
    from app.healing.monitors import log_archival
    runs = isolated / "shinka_results"
    runs.mkdir(parents=True)
    arch = runs / "archive"
    arch.mkdir()
    (arch / "run_old.tar.gz").write_text("x")
    old_ts = time.time() - 200 * 86400
    os.utime(arch, (old_ts, old_ts))

    summary = log_archival._archive_evolution_runs()
    assert summary["archived_runs"] == 0
    assert arch.exists()
