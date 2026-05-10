"""Tests for app.audit.journal — the audit-journal facade."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.audit import journal


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path):
    journal._reset_for_tests(tmp_path)
    yield
    journal._reset_for_tests(None)


def test_append_and_read_recent(tmp_path: Path) -> None:
    journal.append("boot", "system started")
    journal.append("task_completed", "x")
    journal.append("anomaly", "y", files_changed=["a.py", "b.py"])

    entries = journal.read_recent(10)
    assert [e["event"] for e in entries] == ["boot", "task_completed", "anomaly"]
    assert entries[2]["files_changed"] == ["a.py", "b.py"]
    assert "ts" in entries[0]


def test_read_recent_caps_to_n(tmp_path: Path) -> None:
    for i in range(50):
        journal.append("evt", f"d{i}")
    last5 = journal.read_recent(5)
    assert len(last5) == 5
    assert [e["detail"] for e in last5] == [f"d{i}" for i in range(45, 50)]


def test_read_since_filters_by_payload_ts(tmp_path: Path) -> None:
    journal.append("old", "x")
    cutoff = datetime.now(timezone.utc) + timedelta(milliseconds=10)
    # Brief sleep substitute: the next append's ts will be after cutoff.
    import time
    time.sleep(0.05)
    journal.append("new1", "y")
    journal.append("new2", "z")

    out = journal.read_since(cutoff)
    assert [e["event"] for e in out] == ["new1", "new2"]


def test_read_since_skips_unparseable_ts(tmp_path: Path) -> None:
    # Inject an entry with a malformed ts directly through the underlying store.
    from app.audit.rolled_log import RolledLogStore
    store = RolledLogStore(tmp_path, "audit_journal")
    store.append({"ts": "not-a-timestamp", "event": "broken", "detail": "x", "files_changed": []})
    journal.append("ok", "y")

    out = journal.read_since(datetime(2000, 1, 1, tzinfo=timezone.utc))
    assert [e["event"] for e in out] == ["ok"]


def test_migration_from_legacy_single_file(tmp_path: Path) -> None:
    legacy = tmp_path / "audit_journal.json"
    legacy.write_text(json.dumps([
        {"ts": "2026-05-01T00:00:00+00:00", "event": "old1", "detail": "a", "files_changed": []},
        {"ts": "2026-05-02T00:00:00+00:00", "event": "old2", "detail": "b", "files_changed": ["x.py"]},
    ]))

    seq = journal.append("post-migration", "c")
    assert seq == 3

    entries = journal.read_recent(10)
    assert [e["event"] for e in entries] == ["old1", "old2", "post-migration"]

    preserved = legacy.with_suffix(".json.preserved")
    assert preserved.exists()
    assert not legacy.exists()


def test_migration_idempotent_under_concurrent_first_call(tmp_path: Path) -> None:
    legacy = tmp_path / "audit_journal.json"
    legacy.write_text(json.dumps([{"ts": "2026-05-01T00:00:00+00:00", "event": "x"}]))

    threads = []
    seqs: list[int] = []
    barrier = threading.Barrier(8)

    def race():
        barrier.wait()
        seqs.append(journal.append("concurrent", "y"))

    for _ in range(8):
        threads.append(threading.Thread(target=race))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    entries = journal.read_recent(20)
    assert len(entries) == 1 + 8
    assert sorted(seqs) == list(range(2, 10))


def test_active_segment_path_is_heartbeat_probe(tmp_path: Path) -> None:
    journal.append("heartbeat", "x")
    p = journal.active_segment_path()
    assert p.exists()
    # mtime should be very recent.
    age_s = (datetime.now(timezone.utc).timestamp() - p.stat().st_mtime)
    assert 0 <= age_s < 5


def test_legacy_path_resolves_relative_to_workspace(tmp_path: Path) -> None:
    assert journal.legacy_path() == tmp_path / "audit_journal.json"
    assert journal.active_segment_path() == tmp_path / "audit_journal" / "current.jsonl"


def test_stats_after_migration(tmp_path: Path) -> None:
    legacy = tmp_path / "audit_journal.json"
    legacy.write_text(json.dumps([{"ts": "2026-05-01T00:00:00+00:00", "event": f"e{i}"} for i in range(5)]))

    journal.append("post", "x")
    s = journal.stats()
    assert s["log_name"] == "audit_journal"
    assert s["n_segments_closed"] >= 1  # seg-0000 from migration
    assert s["next_seq"] == 7  # 5 migrated + 1 new = next is 7


def test_append_does_not_lose_entries_past_200(tmp_path: Path) -> None:
    """Regression test for the FIFO-200 truncation bug. The legacy
    auditor.py truncated audit_journal.json to last 200 entries on
    every save; silent_regression_detector reads 14 days and saw at
    most a few hours of history. With rolled storage no entry is dropped.
    """
    for i in range(450):
        journal.append("evt", f"d{i}")

    s = journal.stats()
    total = s["n_entries_closed"] + s["n_entries_current"]
    assert total == 450

    all_entries = journal.read_recent(1000)
    assert len(all_entries) == 450
    assert [e["detail"] for e in all_entries[:3]] == ["d0", "d1", "d2"]
    assert [e["detail"] for e in all_entries[-3:]] == ["d447", "d448", "d449"]
