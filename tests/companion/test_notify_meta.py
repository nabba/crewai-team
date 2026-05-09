"""Tests for ``app.companion.notify_meta`` (Phase B #3, 2026-05-09)."""
from __future__ import annotations

import json
import time

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.companion import notify_meta
    monkeypatch.setattr(notify_meta, "_META_PATH", tmp_path / "notify_meta.jsonl")
    yield tmp_path / "notify_meta.jsonl"


def test_record_creates_file(isolated):
    from app.companion import notify_meta
    notify_meta.record(1715200000123, {"skill_id": "abc"})
    assert isolated.exists()
    line = isolated.read_text().strip()
    assert json.loads(line)["metadata"]["skill_id"] == "abc"


def test_lookup_within_window(isolated):
    from app.companion import notify_meta
    notify_meta.record(1715200000123, {"recipe_id": "r1"})
    found = notify_meta.lookup(1715200000123)
    assert found == {"recipe_id": "r1"}


def test_lookup_within_skew_window(isolated):
    from app.companion import notify_meta
    notify_meta.record(1715200000000, {"recipe_id": "r2"})
    # 3 s skew (3000 ms) — within the 5 s window.
    assert notify_meta.lookup(1715200003000) == {"recipe_id": "r2"}


def test_lookup_outside_window(isolated):
    from app.companion import notify_meta
    notify_meta.record(1715200000000, {"recipe_id": "r3"})
    # 10 s skew — outside window.
    assert notify_meta.lookup(1715200010001) is None


def test_lookup_missing_file(isolated):
    from app.companion import notify_meta
    assert notify_meta.lookup(123456789) is None


def test_record_swallows_errors(isolated, monkeypatch):
    from app.companion import notify_meta

    def boom(*a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr("pathlib.Path.mkdir", boom)
    notify_meta.record(1, {"x": "y"})  # must not raise


def test_prune_removes_old(isolated):
    from app.companion import notify_meta
    now_ms = int(time.time() * 1000)
    old_ms = now_ms - 30 * 86400 * 1000  # 30 days ago
    notify_meta.record(old_ms, {"a": 1})
    notify_meta.record(now_ms, {"b": 2})
    dropped = notify_meta.prune(retention_days=14)
    assert dropped == 1
    text = isolated.read_text()
    assert '"b": 2' in text
    assert '"a": 1' not in text


def test_prune_handles_garbage(isolated):
    from app.companion import notify_meta
    isolated.write_text("not json\n{\"ts\": 0, \"metadata\": {}}\n")
    dropped = notify_meta.prune(retention_days=14)
    # Both should drop: garbage + old.
    assert dropped == 2
