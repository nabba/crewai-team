"""Tests for ``app.healing.listener_heartbeats`` (Wave 0/1 #A3)."""
from __future__ import annotations

import time

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.healing import listener_heartbeats
    monkeypatch.setattr(listener_heartbeats, "_HEARTBEAT_DIR", tmp_path / "hb")
    yield tmp_path / "hb"


def test_touch_creates_file(isolated):
    from app.healing import listener_heartbeats
    listener_heartbeats.touch("firebase-mode-poll")
    assert (isolated / "firebase-mode-poll.heartbeat").exists()


def test_touch_updates_mtime(isolated):
    from app.healing import listener_heartbeats
    listener_heartbeats.touch("firebase-kb-poll")
    p = isolated / "firebase-kb-poll.heartbeat"
    first_mtime = p.stat().st_mtime
    time.sleep(0.05)
    listener_heartbeats.touch("firebase-kb-poll")
    assert p.stat().st_mtime > first_mtime


def test_touch_sanitizes_path_separators(isolated):
    from app.healing import listener_heartbeats
    listener_heartbeats.touch("a/../../etc/passwd")
    # No traversal — file lives inside the heartbeat dir.
    assert (isolated / "a_.._.._etc_passwd.heartbeat").exists()


def test_touch_swallows_errors(isolated, monkeypatch):
    from app.healing import listener_heartbeats
    import pathlib

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(pathlib.Path, "touch", boom)
    # Must not raise.
    listener_heartbeats.touch("firebase-mode-poll")


def test_list_heartbeats_returns_records(isolated):
    from app.healing import listener_heartbeats
    listener_heartbeats.touch("firebase-mode-poll")
    listener_heartbeats.touch("firebase-kb-poll")
    records = listener_heartbeats.list_heartbeats()
    names = {r["name"] for r in records}
    assert names == {"firebase-mode-poll", "firebase-kb-poll"}
    for r in records:
        assert "mtime" in r
        assert "age_s" in r
        assert r["age_s"] >= 0


def test_list_heartbeats_empty_when_dir_missing(isolated):
    from app.healing import listener_heartbeats
    # Don't create dir.
    assert listener_heartbeats.list_heartbeats() == []


def test_known_listeners_match_thread_names():
    """KNOWN_LISTENERS must match the threading.Thread name= conventions."""
    from app.healing.listener_heartbeats import KNOWN_LISTENERS
    expected = {
        "firebase-mode-poll", "firebase-kb-poll", "firebase-phil-poll",
        "firebase-fiction-poll", "firebase-episteme-poll",
        "firebase-experiential-poll", "firebase-aesthetics-poll",
        "firebase-tensions-poll", "firebase-chat-poll",
    }
    assert set(KNOWN_LISTENERS) == expected
