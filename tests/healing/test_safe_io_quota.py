"""Tests for the disk-quota guard added to ``app.safe_io``."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Force the threshold to a known value + stub the audit logger."""
    monkeypatch.setenv("DISK_FREE_THRESHOLD_MB", "100")

    # Bypass the audit log so tests don't touch Postgres.
    from app import safe_io

    audit_calls: list[dict] = []

    class _FakeAudit:
        def log(self, **kw):
            audit_calls.append(kw)

    def _fake_get_audit():
        return _FakeAudit()

    monkeypatch.setattr(
        "app.control_plane.audit.get_audit",
        _fake_get_audit,
        raising=False,
    )
    yield tmp_path, audit_calls


# ── Threshold / env-var behaviour ─────────────────────────────────────────


def test_disabled_when_threshold_zero(isolated, monkeypatch):
    """``DISK_FREE_THRESHOLD_MB=0`` skips the check entirely."""
    tmp_path, _ = isolated
    monkeypatch.setenv("DISK_FREE_THRESHOLD_MB", "0")

    from app import safe_io

    # Mock disk_usage to return *zero* free space — we should still write.
    class _Usage:
        total = 1
        used = 1
        free = 0

    monkeypatch.setattr(safe_io.shutil, "disk_usage", lambda _p: _Usage)

    target = tmp_path / "out.txt"
    safe_io.safe_write(target, "hi")
    assert target.read_text() == "hi"


def test_invalid_threshold_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("DISK_FREE_THRESHOLD_MB", "garbage")
    from app import safe_io
    assert safe_io._free_threshold_mb() == 200


def test_negative_threshold_clamped_to_zero(monkeypatch):
    monkeypatch.setenv("DISK_FREE_THRESHOLD_MB", "-5")
    from app import safe_io
    assert safe_io._free_threshold_mb() == 0


# ── Refusal path ──────────────────────────────────────────────────────────


def test_refuses_write_below_threshold(isolated, monkeypatch):
    tmp_path, audit_calls = isolated
    from app import safe_io, safe_io as _si

    class _Usage:
        total = 1024 * 1024 * 200
        used = 1024 * 1024 * 199
        free = 1024 * 1024 * 1  # 1 MB free, threshold 100 MB

    monkeypatch.setattr(safe_io.shutil, "disk_usage", lambda _p: _Usage)

    with pytest.raises(_si.DiskQuotaError):
        safe_io.safe_write(tmp_path / "out.txt", "no")

    # Audit row was emitted with the right shape.
    assert len(audit_calls) == 1
    detail = audit_calls[0]["detail"]
    assert detail["threshold_mb"] == 100
    assert detail["free_mb"] < 100


def test_refuses_append_below_threshold(isolated, monkeypatch):
    tmp_path, _ = isolated
    from app import safe_io, safe_io as _si

    class _Usage:
        total = 1024 * 1024 * 200
        used = 1024 * 1024 * 199
        free = 1024 * 1024 * 1

    monkeypatch.setattr(safe_io.shutil, "disk_usage", lambda _p: _Usage)

    with pytest.raises(_si.DiskQuotaError):
        safe_io.safe_append(tmp_path / "log.jsonl", '{"x": 1}')


def test_disk_quota_error_is_oserror(isolated, monkeypatch):
    """Existing handlers that catch ``OSError`` should still route correctly."""
    tmp_path, _ = isolated
    from app import safe_io, safe_io as _si

    class _Usage:
        total = 1
        used = 1
        free = 0

    monkeypatch.setattr(safe_io.shutil, "disk_usage", lambda _p: _Usage)

    try:
        safe_io.safe_write(tmp_path / "out.txt", "x")
    except OSError as exc:
        assert isinstance(exc, _si.DiskQuotaError)
    else:
        pytest.fail("expected DiskQuotaError")


# ── Fail-open paths ───────────────────────────────────────────────────────


def test_fails_open_when_disk_usage_raises(isolated, monkeypatch):
    """If shutil.disk_usage itself errors, the guard fails OPEN — we don't
    let a buggy probe halt the system.
    """
    tmp_path, _ = isolated
    from app import safe_io

    def _broken(_p):
        raise PermissionError("simulated")

    monkeypatch.setattr(safe_io.shutil, "disk_usage", _broken)

    target = tmp_path / "out.txt"
    safe_io.safe_write(target, "ok")
    assert target.read_text() == "ok"


def test_walks_up_to_existing_ancestor(isolated, monkeypatch):
    """Path doesn't exist yet — the guard probes the nearest existing parent."""
    tmp_path, _ = isolated
    from app import safe_io

    captured: list[str] = []

    class _Usage:
        total = 1024 * 1024 * 1024
        used = 1024
        free = 1024 * 1024 * 1024  # plenty

    def _captured(p):
        captured.append(str(p))
        return _Usage

    monkeypatch.setattr(safe_io.shutil, "disk_usage", _captured)

    deep = tmp_path / "a" / "b" / "c" / "out.txt"
    safe_io.safe_write(deep, "ok")
    # The probe path was tmp_path itself (the only existing ancestor).
    assert captured[0] == str(tmp_path)


def test_happy_path_above_threshold(isolated, monkeypatch):
    tmp_path, _ = isolated
    from app import safe_io

    class _Usage:
        total = 1024 * 1024 * 1024
        used = 1
        free = 1024 * 1024 * 500  # 500 MB free, threshold 100 MB

    monkeypatch.setattr(safe_io.shutil, "disk_usage", lambda _p: _Usage)

    target = tmp_path / "out.txt"
    safe_io.safe_write(target, "yes")
    assert target.read_text() == "yes"
