"""Tests for ``app.healing.boot_reset`` (Wave 0/1 #A7)."""
from __future__ import annotations

import dbm
import time

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.healing import boot_reset

    # Reset the module-level guard so each test re-runs the sweep.
    monkeypatch.setattr(boot_reset, "_already_ran", False)
    monkeypatch.setattr(
        boot_reset, "_DBM_PATH_CANDIDATES", (tmp_path / "idle_job_state",),
    )
    yield tmp_path


def test_no_dbm_returns_empty(isolated):
    from app.healing import boot_reset
    summary = boot_reset.reset_stale_cooldowns()
    # Empty dbm is created by dbm.open(c-mode); examined=0 because there are no keys.
    assert summary["examined"] == 0
    assert summary["reset"] == []


def test_sweeps_expired_skip_keys(isolated):
    from app.healing import boot_reset
    base = str(isolated / "idle_job_state")
    with dbm.open(base, "c") as db:
        db["skip:expired_job"] = str(time.time() - 10).encode("utf-8")
        db["skip:still_alive"] = str(time.time() + 3600).encode("utf-8")
        db["other_key"] = b"other"

    summary = boot_reset.reset_stale_cooldowns()
    assert "expired_job" in summary["reset"]
    assert any(s["job"] == "still_alive" for s in summary["spared_alive"])
    # other_key is ignored (doesn't start with skip:).
    assert summary["examined"] == 2

    # Verify the dbm now lacks the expired key.
    with dbm.open(base, "c") as db:
        assert b"skip:expired_job" not in db.keys()
        assert b"skip:still_alive" in db.keys()
        assert b"other_key" in db.keys()


def test_idempotent_within_process(isolated):
    from app.healing import boot_reset
    base = str(isolated / "idle_job_state")
    with dbm.open(base, "c") as db:
        db["skip:expired_job"] = str(time.time() - 10).encode("utf-8")

    s1 = boot_reset.reset_stale_cooldowns()
    s2 = boot_reset.reset_stale_cooldowns()
    assert s1["ran"] is True
    assert s2["ran"] is False  # second call skipped


def test_ignores_malformed_skip_value(isolated):
    from app.healing import boot_reset
    base = str(isolated / "idle_job_state")
    with dbm.open(base, "c") as db:
        db["skip:bad"] = b"not-a-float"
        db["skip:expired"] = str(time.time() - 10).encode("utf-8")

    summary = boot_reset.reset_stale_cooldowns()
    assert "expired" in summary["reset"]
    assert "bad" not in summary["reset"]  # skipped silently
