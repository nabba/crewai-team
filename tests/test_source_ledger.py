"""PROGRAM §56 — source ledger tests.

Coverage:
  * append/read roundtrip
  * hash chain construction + verification
  * tamper detection (modify a row, chain breaks)
  * malformed rows skipped during read
  * count_rows correctness
  * read_since (incremental window)
  * list_kbs discovery (skips quarantined)
  * hook_collection_add bulk path
  * drift detection (in_sync, kb_short, ledger_short)
  * bit_rot_scan extension picks up source ledgers
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sqlite3
import time
from pathlib import Path

import pytest


@pytest.fixture
def ledger_module(tmp_path, monkeypatch):
    """Reload source_ledger with a tmp WORKSPACE_ROOT."""
    import app.paths as paths
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", tmp_path)
    import app.memory.source_ledger as sl
    importlib.reload(sl)
    return sl


def _make_workspace_kb(ws: Path, kb_name: str, with_chroma: bool = True) -> Path:
    kb_dir = ws / kb_name
    kb_dir.mkdir(parents=True, exist_ok=True)
    if with_chroma:
        # Create a placeholder chroma.sqlite3 so list_kbs picks it up.
        conn = sqlite3.connect(str(kb_dir / "chroma.sqlite3"))
        conn.execute("CREATE TABLE IF NOT EXISTS placeholder (id INTEGER)")
        conn.commit()
        conn.close()
    return kb_dir


# ────────────────────────────────────────────────────────────────────
#   Append / read
# ────────────────────────────────────────────────────────────────────


def test_append_creates_ledger_file_on_first_write(ledger_module, tmp_path):
    kb = "memory"
    _make_workspace_kb(tmp_path, kb)
    row = ledger_module.append_row(kb, "beliefs", "doc-1", "hello", {"k": "v"})
    assert row is not None
    assert row.collection == "beliefs"
    assert row.doc_id == "doc-1"
    assert row.text == "hello"
    assert row.prev_hash == ledger_module.GENESIS_HASH
    assert len(row.hash) == 64
    # File exists.
    assert (tmp_path / kb / ".source_ledger.jsonl").exists()


def test_append_chains_each_row_to_the_previous(ledger_module, tmp_path):
    kb = "memory"
    _make_workspace_kb(tmp_path, kb)
    r1 = ledger_module.append_row(kb, "c", "d1", "one", {})
    r2 = ledger_module.append_row(kb, "c", "d2", "two", {})
    r3 = ledger_module.append_row(kb, "c", "d3", "three", {})
    assert r1.prev_hash == ledger_module.GENESIS_HASH
    assert r2.prev_hash == r1.hash
    assert r3.prev_hash == r2.hash


def test_read_all_returns_rows_in_append_order(ledger_module, tmp_path):
    kb = "memory"
    _make_workspace_kb(tmp_path, kb)
    for i in range(5):
        ledger_module.append_row(kb, "c", f"d{i}", f"text-{i}", {})
    rows = list(ledger_module.read_all(kb))
    assert [r.doc_id for r in rows] == ["d0", "d1", "d2", "d3", "d4"]


def test_count_rows_matches_appended(ledger_module, tmp_path):
    kb = "memory"
    _make_workspace_kb(tmp_path, kb)
    for i in range(7):
        ledger_module.append_row(kb, "c", f"d{i}", f"text-{i}", {})
    assert ledger_module.count_rows(kb) == 7


def test_read_all_handles_missing_file(ledger_module, tmp_path):
    rows = list(ledger_module.read_all("no_such_kb"))
    assert rows == []


def test_read_all_skips_malformed_lines(ledger_module, tmp_path):
    kb = "memory"
    _make_workspace_kb(tmp_path, kb)
    ledger_module.append_row(kb, "c", "ok-1", "valid", {})
    path = ledger_module.ledger_path(kb)
    # Append junk + then a valid row.
    with path.open("a") as f:
        f.write("this is not json\n")
    ledger_module.append_row(kb, "c", "ok-2", "valid", {})
    rows = list(ledger_module.read_all(kb))
    doc_ids = [r.doc_id for r in rows]
    # The junk line is skipped; both valid rows survive.
    assert doc_ids == ["ok-1", "ok-2"]


def test_read_since_returns_only_newer_rows(ledger_module, tmp_path):
    kb = "memory"
    _make_workspace_kb(tmp_path, kb)
    ledger_module.append_row(kb, "c", "d1", "first", {}, ts=100.0)
    ledger_module.append_row(kb, "c", "d2", "second", {}, ts=200.0)
    ledger_module.append_row(kb, "c", "d3", "third", {}, ts=300.0)
    newer = list(ledger_module.read_since(kb, 150.0))
    assert [r.doc_id for r in newer] == ["d2", "d3"]


# ────────────────────────────────────────────────────────────────────
#   Hash chain
# ────────────────────────────────────────────────────────────────────


def test_verify_chain_passes_on_clean_ledger(ledger_module, tmp_path):
    kb = "memory"
    _make_workspace_kb(tmp_path, kb)
    for i in range(10):
        ledger_module.append_row(kb, "c", f"d{i}", f"text-{i}", {})
    result = ledger_module.verify_chain(kb)
    assert result.ok, result.to_dict()
    assert result.rows_seen == 10


def test_verify_chain_catches_modified_row(ledger_module, tmp_path):
    """If any payload field is changed, the recomputed hash won't
    match the stored one — chain breaks at that row."""
    kb = "memory"
    _make_workspace_kb(tmp_path, kb)
    for i in range(5):
        ledger_module.append_row(kb, "c", f"d{i}", f"text-{i}", {})
    # Tamper: rewrite row 2's text but keep the original hash.
    path = ledger_module.ledger_path(kb)
    lines = path.read_text().strip().split("\n")
    row2 = json.loads(lines[2])
    row2["text"] = "TAMPERED"
    lines[2] = json.dumps(row2, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    result = ledger_module.verify_chain(kb)
    assert not result.ok
    assert result.first_bad_row == 2
    assert "hash_mismatch" in result.first_bad_reason


def test_verify_chain_catches_deleted_row(ledger_module, tmp_path):
    """Removing a row mid-chain breaks the prev_hash linkage at the
    next row."""
    kb = "memory"
    _make_workspace_kb(tmp_path, kb)
    for i in range(5):
        ledger_module.append_row(kb, "c", f"d{i}", f"text-{i}", {})
    path = ledger_module.ledger_path(kb)
    lines = path.read_text().strip().split("\n")
    # Remove row 2.
    del lines[2]
    path.write_text("\n".join(lines) + "\n")
    result = ledger_module.verify_chain(kb)
    assert not result.ok
    # Row 2 now claims prev_hash from row 1, but the on-disk file
    # makes "row 2" actually be the former row 3 — its prev_hash
    # references the former row 2 which is gone.
    assert "prev_hash_mismatch" in result.first_bad_reason


def test_verify_chain_genesis_link(ledger_module, tmp_path):
    """The very first row must have prev_hash == GENESIS_HASH."""
    kb = "memory"
    _make_workspace_kb(tmp_path, kb)
    r = ledger_module.append_row(kb, "c", "d1", "first", {})
    assert r.prev_hash == ledger_module.GENESIS_HASH


def test_verify_chain_first_row_corrupted(ledger_module, tmp_path):
    """If the first row's prev_hash isn't genesis, chain breaks
    immediately."""
    kb = "memory"
    _make_workspace_kb(tmp_path, kb)
    ledger_module.append_row(kb, "c", "d1", "first", {})
    path = ledger_module.ledger_path(kb)
    lines = path.read_text().strip().split("\n")
    row0 = json.loads(lines[0])
    row0["prev_hash"] = "f" * 64  # not genesis
    lines[0] = json.dumps(row0, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    result = ledger_module.verify_chain(kb)
    assert not result.ok
    assert result.first_bad_row == 0


# ────────────────────────────────────────────────────────────────────
#   Bulk hook
# ────────────────────────────────────────────────────────────────────


def test_hook_collection_add_appends_each_row(ledger_module, tmp_path):
    kb = "episteme"
    _make_workspace_kb(tmp_path, kb)
    ids = ["a", "b", "c"]
    docs = ["doc-a", "doc-b", "doc-c"]
    metas = [{"i": 1}, {"i": 2}, {"i": 3}]
    n = ledger_module.hook_collection_add(kb, "episteme", ids, docs, metas)
    assert n == 3
    rows = list(ledger_module.read_all(kb))
    assert [r.doc_id for r in rows] == ids
    assert [r.text for r in rows] == docs
    # Chain still verifies.
    assert ledger_module.verify_chain(kb).ok


def test_hook_collection_add_skips_empty_text(ledger_module, tmp_path):
    kb = "episteme"
    _make_workspace_kb(tmp_path, kb)
    n = ledger_module.hook_collection_add(
        kb, "c", ["a", "b"], ["", "valid"], [{}, {}],
    )
    assert n == 1
    rows = list(ledger_module.read_all(kb))
    assert [r.doc_id for r in rows] == ["b"]


# ────────────────────────────────────────────────────────────────────
#   KB discovery
# ────────────────────────────────────────────────────────────────────


def test_list_kbs_finds_live_skips_quarantined(ledger_module, tmp_path):
    _make_workspace_kb(tmp_path, "memory")
    _make_workspace_kb(tmp_path, "episteme")
    _make_workspace_kb(tmp_path, "memory.corrupt_20260425_221402")
    _make_workspace_kb(tmp_path, "philosophy.bak_2026")
    kbs = ledger_module.list_kbs()
    assert sorted(kbs) == ["episteme", "memory"]


def test_list_kbs_empty_workspace(ledger_module, tmp_path):
    assert ledger_module.list_kbs() == []


# ────────────────────────────────────────────────────────────────────
#   Drift detection (mocked chromadb client)
# ────────────────────────────────────────────────────────────────────


class _FakeCollection:
    def __init__(self, name: str, count: int):
        self.name = name
        self._count = count

    def count(self) -> int:
        return self._count


class _FakeClient:
    def __init__(self, collections: list[_FakeCollection]):
        self._cols = collections

    def list_collections(self):
        return self._cols


def test_check_drift_in_sync(ledger_module, tmp_path, monkeypatch):
    kb = "memory"
    _make_workspace_kb(tmp_path, kb)
    for i in range(10):
        ledger_module.append_row(kb, "beliefs", f"d{i}", f"text-{i}", {})

    # Mock chromadb_manager.get_kb_client.
    class FakeMgr:
        def get_kb_client(self, name):
            return _FakeClient([_FakeCollection("beliefs", 10)])
    import sys
    fake = FakeMgr()
    monkeypatch.setitem(sys.modules, "app.memory.chromadb_manager", fake)
    drift = ledger_module.check_drift(kb)
    assert drift.direction == "in_sync"
    assert not drift.needs_replay


def test_check_drift_kb_short_triggers_replay(ledger_module, tmp_path, monkeypatch):
    kb = "memory"
    _make_workspace_kb(tmp_path, kb)
    for i in range(20):
        ledger_module.append_row(kb, "beliefs", f"d{i}", f"text-{i}", {})

    class FakeMgr:
        def get_kb_client(self, name):
            return _FakeClient([_FakeCollection("beliefs", 5)])
    import sys
    monkeypatch.setitem(sys.modules, "app.memory.chromadb_manager", FakeMgr())
    drift = ledger_module.check_drift(kb)
    assert drift.direction == "kb_short"
    assert drift.needs_replay  # 75% loss, well above 5% threshold


def test_check_drift_ledger_short_no_replay(ledger_module, tmp_path, monkeypatch):
    """KB has more rows than the ledger — bootstrap will fix it, not replay."""
    kb = "memory"
    _make_workspace_kb(tmp_path, kb)
    for i in range(5):
        ledger_module.append_row(kb, "beliefs", f"d{i}", f"text-{i}", {})

    class FakeMgr:
        def get_kb_client(self, name):
            return _FakeClient([_FakeCollection("beliefs", 20)])
    import sys
    monkeypatch.setitem(sys.modules, "app.memory.chromadb_manager", FakeMgr())
    drift = ledger_module.check_drift(kb)
    assert drift.direction == "ledger_short"
    assert not drift.needs_replay


def test_check_drift_below_threshold_no_replay(ledger_module, tmp_path, monkeypatch):
    """Tiny drift (<5%) should not trigger replay — it's noise."""
    kb = "memory"
    _make_workspace_kb(tmp_path, kb)
    for i in range(100):
        ledger_module.append_row(kb, "beliefs", f"d{i}", f"text-{i}", {})

    class FakeMgr:
        def get_kb_client(self, name):
            return _FakeClient([_FakeCollection("beliefs", 98)])  # 2% short
    import sys
    monkeypatch.setitem(sys.modules, "app.memory.chromadb_manager", FakeMgr())
    drift = ledger_module.check_drift(kb)
    assert drift.direction == "kb_short"
    assert not drift.needs_replay  # 2% below 5% threshold


# ────────────────────────────────────────────────────────────────────
#   Last-hash optimisation
# ────────────────────────────────────────────────────────────────────


def test_last_hash_uses_tail_seek_for_large_files(ledger_module, tmp_path):
    """Append many rows and verify _last_hash matches the actual final
    row hash without needing a full read."""
    kb = "memory"
    _make_workspace_kb(tmp_path, kb)
    final_row = None
    for i in range(500):
        final_row = ledger_module.append_row(kb, "c", f"d{i}", f"text-{i}", {})
    assert ledger_module._last_hash(ledger_module.ledger_path(kb)) == final_row.hash


# ────────────────────────────────────────────────────────────────────
#   Bit-rot scan integration (pinning test for §56 extension)
# ────────────────────────────────────────────────────────────────────


def test_bit_rot_scan_includes_source_ledgers(ledger_module, tmp_path, monkeypatch):
    """The §56 extension to bit_rot_scan must pick up every KB's
    .source_ledger.jsonl. Pinned so future refactors can't accidentally
    drop this coverage."""
    pytest.importorskip("pydantic_settings")
    import app.paths as paths
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", tmp_path)
    _make_workspace_kb(tmp_path, "memory")
    ledger_module.append_row("memory", "c", "d", "text", {})
    _make_workspace_kb(tmp_path, "philosophy")
    ledger_module.append_row("philosophy", "c", "d", "text", {})
    # Quarantined ledger should be excluded.
    _make_workspace_kb(tmp_path, "memory.corrupt_20260425")
    (tmp_path / "memory.corrupt_20260425" / ".source_ledger.jsonl").write_text("{}\n")

    import app.healing.monitors.bit_rot_scan as brs
    importlib.reload(brs)
    paths_list = brs._critical_paths()
    names = [str(p) for p in paths_list]
    assert any("memory/.source_ledger.jsonl" in n for n in names)
    assert any("philosophy/.source_ledger.jsonl" in n for n in names)
    assert not any("corrupt_" in n for n in names)
