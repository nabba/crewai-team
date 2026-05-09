"""Tests for app.audit.rolled_log + app.audit.migration."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from app.audit.rolled_log import (
    GENESIS,
    LEGACY_PREFIX,
    RolledLogReader,
    RolledLogStore,
    RolledLogVerifier,
)
from app.audit.migration import migrate_json_list, migrate_jsonl


# ── primitive: basic append + read ─────────────────────────────────────────


def test_append_and_iter_chronological(tmp_path: Path) -> None:
    store = RolledLogStore(tmp_path, "demo")
    seqs = [store.append({"i": i}) for i in range(5)]
    assert seqs == [1, 2, 3, 4, 5]

    reader = RolledLogReader(tmp_path, "demo")
    payloads = [e["payload"]["i"] for e in reader.iter_entries()]
    assert payloads == [0, 1, 2, 3, 4]


def test_genesis_first_entry_has_genesis_prev_hash(tmp_path: Path) -> None:
    store = RolledLogStore(tmp_path, "demo")
    store.append({"hello": "world"})

    reader = RolledLogReader(tmp_path, "demo")
    first = next(iter(reader.iter_entries()))
    assert first["prev_hash"] == GENESIS


def test_chain_links_within_segment(tmp_path: Path) -> None:
    store = RolledLogStore(tmp_path, "demo")
    for i in range(20):
        store.append({"i": i})

    result = RolledLogVerifier(tmp_path, "demo").verify()
    assert result.ok, result.errors
    assert result.n_entries == 20


# ── rotation + cross-segment chain ─────────────────────────────────────────


def _force_rotation_store(tmp_path: Path) -> RolledLogStore:
    """Tiny segment cap so a few entries triggers rotation."""
    return RolledLogStore(tmp_path, "demo", max_segment_bytes=512)


def test_rotation_creates_segment_with_root(tmp_path: Path) -> None:
    store = _force_rotation_store(tmp_path)
    for i in range(40):
        store.append({"i": i, "filler": "x" * 32})

    seg_dir = tmp_path / "demo"
    closed_segments = sorted(seg_dir.glob("seg-*.jsonl"))
    assert len(closed_segments) >= 1, "rotation should have produced ≥1 closed segment"

    # current.jsonl should now lead with a segment_root marker
    cur_first = (seg_dir / "current.jsonl").read_text().splitlines()[0]
    obj = json.loads(cur_first)
    assert obj.get("type") == "segment_root"
    assert "prev_root_hash" in obj


def test_chain_continuity_across_rotation(tmp_path: Path) -> None:
    store = _force_rotation_store(tmp_path)
    for i in range(60):
        store.append({"i": i, "filler": "x" * 32})

    result = RolledLogVerifier(tmp_path, "demo").verify()
    assert result.ok, result.errors
    assert result.n_entries == 60
    assert result.n_segments >= 2


def test_iter_entries_spans_segments(tmp_path: Path) -> None:
    store = _force_rotation_store(tmp_path)
    for i in range(50):
        store.append({"i": i, "filler": "x" * 32})

    reader = RolledLogReader(tmp_path, "demo")
    seqs = [e["seq"] for e in reader.iter_entries()]
    assert seqs == list(range(1, 51))


def test_recent_walks_segments_backward(tmp_path: Path) -> None:
    store = _force_rotation_store(tmp_path)
    for i in range(40):
        store.append({"i": i, "filler": "x" * 32})

    reader = RolledLogReader(tmp_path, "demo")
    last5 = reader.recent(5)
    assert [e["payload"]["i"] for e in last5] == [35, 36, 37, 38, 39]
    # newest-last
    assert last5[-1]["seq"] > last5[0]["seq"]


# ── tamper detection ──────────────────────────────────────────────────────


def test_tamper_in_current_detected(tmp_path: Path) -> None:
    store = RolledLogStore(tmp_path, "demo")
    for i in range(5):
        store.append({"i": i})

    cur = tmp_path / "demo" / "current.jsonl"
    text = cur.read_text()
    # Flip the value of the second entry's payload.
    tampered = text.replace('"i":1', '"i":99', 1)
    assert tampered != text, "test setup: nothing replaced"
    cur.write_text(tampered)

    result = RolledLogVerifier(tmp_path, "demo").verify()
    assert not result.ok
    assert any("CHAIN_BREAK" in e for e in result.errors), result.errors


def test_tamper_in_closed_segment_detected(tmp_path: Path) -> None:
    store = _force_rotation_store(tmp_path)
    for i in range(40):
        store.append({"i": i, "filler": "x" * 32})

    seg = next(iter(sorted((tmp_path / "demo").glob("seg-*.jsonl"))))
    text = seg.read_text()
    tampered = text.replace('"i":2', '"i":222', 1)
    if tampered == text:
        pytest.skip("tampering target not present in this segment")
    seg.write_text(tampered)

    result = RolledLogVerifier(tmp_path, "demo").verify()
    assert not result.ok


# ── migration: JSON list-of-dicts (audit_journal.json case) ───────────────


def test_migrate_json_list(tmp_path: Path) -> None:
    legacy = tmp_path / "audit_journal.json"
    legacy.write_text(json.dumps([
        {"event": "boot", "ts": "2026-05-01T00:00:00Z"},
        {"event": "task_completed", "detail": "x"},
        {"event": "anomaly", "ts": "2026-05-02T00:00:00Z"},
    ]))

    base = tmp_path / "logs"
    summary = migrate_json_list(legacy, base, "audit_journal")
    assert summary["status"] == "migrated"
    assert summary["n_entries"] == 3

    # Source preserved, not deleted.
    assert (legacy.with_suffix(".json.preserved")).exists()
    assert not legacy.exists()

    # Verifier accepts seg-0000 via the legacy boundary.
    result = RolledLogVerifier(base, "audit_journal").verify()
    assert result.ok, result.errors
    assert 0 in result.legacy_boundaries

    # current.jsonl exists with a segment_root.
    cur_first = (base / "audit_journal" / "current.jsonl").read_text().splitlines()[0]
    obj = json.loads(cur_first)
    assert obj["type"] == "segment_root"
    assert obj["prev_segment_id"] == 0


def test_migration_idempotent(tmp_path: Path) -> None:
    legacy = tmp_path / "audit_journal.json"
    legacy.write_text(json.dumps([{"event": "x"}]))
    base = tmp_path / "logs"

    first = migrate_json_list(legacy, base, "audit_journal")
    assert first["status"] == "migrated"

    # Re-running points at the .preserved file (legacy is gone). Recreate the
    # legacy filename with garbage to prove the idempotent guard kicks in
    # before any read attempt.
    legacy.write_text(json.dumps([{"event": "garbage"}]))
    second = migrate_json_list(legacy, base, "audit_journal")
    assert second["status"] == "already_migrated"


def test_post_migration_append_chains_correctly(tmp_path: Path) -> None:
    legacy = tmp_path / "audit_journal.json"
    legacy.write_text(json.dumps([
        {"event": f"e{i}", "ts": "2026-05-01T00:00:00Z"} for i in range(10)
    ]))
    base = tmp_path / "logs"
    migrate_json_list(legacy, base, "audit_journal")

    # New writes flow into current.jsonl via RolledLogStore.
    store = RolledLogStore(base, "audit_journal")
    new_seq = store.append({"event": "post-migration"})
    assert new_seq == 11

    result = RolledLogVerifier(base, "audit_journal").verify()
    assert result.ok, result.errors
    assert result.n_entries == 11
    assert 0 in result.legacy_boundaries


# ── migration: JSONL + rotated companions (errors.jsonl case) ─────────────


def test_migrate_jsonl_with_rotated_companions(tmp_path: Path) -> None:
    rotated_old = tmp_path / "errors.jsonl.2"
    rotated_old.write_text('{"i": 1}\n{"i": 2}\n')
    rotated_recent = tmp_path / "errors.jsonl.1"
    rotated_recent.write_text('{"i": 3}\n')
    active = tmp_path / "errors.jsonl"
    active.write_text('{"i": 4}\n{"i": 5}\n')

    base = tmp_path / "logs"
    summary = migrate_jsonl(
        active,
        base,
        "errors",
        rotated_companions=[rotated_old, rotated_recent],
    )
    assert summary["status"] == "migrated"
    assert summary["n_entries"] == 5

    reader = RolledLogReader(base, "errors")
    payloads = [e["payload"]["i"] for e in reader.iter_entries()]
    assert payloads == [1, 2, 3, 4, 5]

    result = RolledLogVerifier(base, "errors").verify()
    assert result.ok, result.errors


def test_migrate_jsonl_skips_malformed_lines(tmp_path: Path) -> None:
    src = tmp_path / "errors.jsonl"
    src.write_text('{"ok": 1}\nthis is not json\n{"ok": 2}\n')

    base = tmp_path / "logs"
    summary = migrate_jsonl(src, base, "errors")
    assert summary["n_entries"] == 2

    result = RolledLogVerifier(base, "errors").verify()
    assert result.ok


# ── crash recovery ────────────────────────────────────────────────────────


def test_recovery_when_current_missing_after_rotation(tmp_path: Path) -> None:
    store = _force_rotation_store(tmp_path)
    for i in range(40):
        store.append({"i": i, "filler": "x" * 32})

    cur = tmp_path / "demo" / "current.jsonl"
    cur.unlink()

    new_seq = store.append({"i": "after-recovery"})
    assert new_seq > 40

    # Newly created current.jsonl should lead with a segment_root linking to
    # the last closed segment.
    cur_lines = cur.read_text().splitlines()
    head = json.loads(cur_lines[0])
    assert head.get("type") == "segment_root"

    result = RolledLogVerifier(tmp_path, "demo").verify()
    assert result.ok, result.errors


def test_reader_skips_torn_last_line(tmp_path: Path) -> None:
    store = RolledLogStore(tmp_path, "demo")
    for i in range(3):
        store.append({"i": i})

    cur = tmp_path / "demo" / "current.jsonl"
    cur.write_text(cur.read_text() + '{"seq": 999, "ts":')  # torn append

    reader = RolledLogReader(tmp_path, "demo")
    payloads = [e["payload"]["i"] for e in reader.iter_entries()]
    assert payloads == [0, 1, 2]


# ── concurrency ───────────────────────────────────────────────────────────


def test_concurrent_appends_yield_unique_monotonic_seqs(tmp_path: Path) -> None:
    store = RolledLogStore(tmp_path, "demo")
    n_threads = 8
    per_thread = 30

    def worker(tid: int) -> None:
        for k in range(per_thread):
            store.append({"tid": tid, "k": k})

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    reader = RolledLogReader(tmp_path, "demo")
    seqs = [e["seq"] for e in reader.iter_entries()]
    assert seqs == sorted(seqs)
    assert len(seqs) == n_threads * per_thread
    assert len(set(seqs)) == len(seqs), "sequence numbers must be unique"

    result = RolledLogVerifier(tmp_path, "demo").verify()
    assert result.ok, result.errors


# ── stats ──────────────────────────────────────────────────────────────────


def test_stats_reflect_segment_state(tmp_path: Path) -> None:
    store = _force_rotation_store(tmp_path)
    for i in range(40):
        store.append({"i": i, "filler": "x" * 32})

    s = store.stats()
    assert s["log_name"] == "demo"
    assert s["n_segments_closed"] >= 1
    assert s["n_entries_closed"] + s["n_entries_current"] == 40
    assert s["next_seq"] == 41
