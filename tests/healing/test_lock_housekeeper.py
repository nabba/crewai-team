"""Tests for ``app.healing.monitors.lock_housekeeper``."""
from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.life_companion import _common
    from app.healing.monitors import lock_housekeeper

    monkeypatch.setattr(_common, "_STATE_DIR", tmp_path / "lc")
    monkeypatch.setattr(lock_housekeeper, "background_enabled", lambda: True)

    sent: list[str] = []
    monkeypatch.setattr(lock_housekeeper, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(lock_housekeeper, "audit_event", lambda *a, **k: None)

    # Watch only the tmp dir.
    watch_dir = tmp_path / "locks"
    watch_dir.mkdir()
    monkeypatch.setattr(lock_housekeeper, "_WATCHED_DIRS", (watch_dir,))

    yield tmp_path, watch_dir, sent


# ── Age-based protection ──────────────────────────────────────────────────


def test_skip_recent_lock(isolated):
    tmp_path, watch_dir, sent = isolated
    from app.healing.monitors import lock_housekeeper

    p = watch_dir / "fresh.lock"
    p.write_text("")
    # mtime is "now" — under the 1-hour threshold.
    lock_housekeeper.run()
    assert p.exists()  # not deleted


def test_delete_old_uncontested_lock(isolated):
    tmp_path, watch_dir, sent = isolated
    from app.healing.monitors import lock_housekeeper

    p = watch_dir / "stale.lock"
    p.write_text("")
    # Backdate the mtime past the 1-hour threshold.
    old = time.time() - lock_housekeeper._MIN_AGE_S - 60
    os.utime(p, (old, old))

    lock_housekeeper.run()
    assert not p.exists()


def test_skip_held_lock(isolated):
    """A lock currently held by SOMEONE (the test itself) must be spared
    even when old. We hold an fcntl.LOCK_EX on it for the duration of
    the run().
    """
    tmp_path, watch_dir, sent = isolated
    from app.healing.monitors import lock_housekeeper

    p = watch_dir / "held.lock"
    p.write_text("")
    old = time.time() - lock_housekeeper._MIN_AGE_S - 60
    os.utime(p, (old, old))

    fh = open(p, "rb+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_housekeeper.run()
        assert p.exists()  # held → not deleted
    finally:
        try:
            fcntl.flock(fh, fcntl.LOCK_UN)
        except Exception:
            pass
        fh.close()


def test_held_then_released_is_deleted_next_pass(isolated):
    tmp_path, watch_dir, sent = isolated
    from app.healing.monitors import lock_housekeeper

    p = watch_dir / "held.lock"
    p.write_text("")
    old = time.time() - lock_housekeeper._MIN_AGE_S - 60
    os.utime(p, (old, old))

    # Pass 1 — held → spared.
    fh = open(p, "rb+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_housekeeper.run()
        assert p.exists()
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()

    # Pass 2 — released → deleted.
    lock_housekeeper.run()
    assert not p.exists()


# ── Pile-up alert ─────────────────────────────────────────────────────────


def test_pile_up_alert_fires_above_threshold(isolated, monkeypatch):
    tmp_path, watch_dir, sent = isolated
    from app.healing.monitors import lock_housekeeper

    # Create 60 fresh files (above the 50 threshold). They're young, so
    # nothing gets deleted — but the count alone triggers the alert.
    for i in range(60):
        (watch_dir / f"leak_{i}.lock").write_text("")

    lock_housekeeper.run()
    assert any("piled up" in s for s in sent)


def test_pile_up_alert_deduped_within_cooldown(isolated):
    tmp_path, watch_dir, sent = isolated
    from app.healing.monitors import lock_housekeeper

    for i in range(60):
        (watch_dir / f"leak_{i}.lock").write_text("")

    lock_housekeeper.run()
    n_after_first = len(sent)
    lock_housekeeper.run()
    n_after_second = len(sent)
    # Second run shouldn't have re-alerted (24h cooldown).
    assert n_after_second == n_after_first


# ── Defensive paths ───────────────────────────────────────────────────────


def test_missing_dir_is_no_op(isolated, monkeypatch):
    tmp_path, watch_dir, sent = isolated
    from app.healing.monitors import lock_housekeeper

    # Repoint to a non-existent dir.
    monkeypatch.setattr(
        lock_housekeeper, "_WATCHED_DIRS", (tmp_path / "does-not-exist",),
    )
    lock_housekeeper.run()  # should not raise
    assert sent == []


def test_non_lock_files_ignored(isolated):
    tmp_path, watch_dir, sent = isolated
    from app.healing.monitors import lock_housekeeper

    (watch_dir / "not-a-lock.txt").write_text("")
    p_old_txt = watch_dir / "also-not.json"
    p_old_txt.write_text("")
    old = time.time() - lock_housekeeper._MIN_AGE_S - 60
    os.utime(p_old_txt, (old, old))

    lock_housekeeper.run()
    # Both still exist — extension must end in .lock.
    assert (watch_dir / "not-a-lock.txt").exists()
    assert p_old_txt.exists()
