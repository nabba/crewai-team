"""PROGRAM §56 iter-2 — tombstones, fold-replay, compaction, manifest.

Covers:
  * append_delete / append_update emit op-tagged rows with valid hashes
  * Hash chain stays valid across mixed add/update/delete sequences
  * Fold-replay: deletes are honored, updates merge, adds survive
  * Compaction: skips small ledgers, runs on dirty ones, archives history
  * Compaction is idempotent (re-running on compacted ledger no-ops)
  * Warm-spare manifest includes new KB ledger + backup paths
"""
from __future__ import annotations

import importlib
import json
import sqlite3
import time
from pathlib import Path

import pytest


@pytest.fixture
def ledger_module(tmp_path, monkeypatch):
    import app.paths as paths
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", tmp_path)
    import app.memory.source_ledger as sl
    importlib.reload(sl)
    return sl


def _make_kb(ws: Path, name: str) -> Path:
    kb = ws / name
    kb.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(kb / "chroma.sqlite3"))
    conn.execute("CREATE TABLE IF NOT EXISTS placeholder (id INTEGER)")
    conn.commit()
    conn.close()
    return kb


# ────────────────────────────────────────────────────────────────────
#   Tombstone op rows
# ────────────────────────────────────────────────────────────────────


def test_append_delete_emits_delete_op_row(ledger_module, tmp_path):
    kb = "memory"
    _make_kb(tmp_path, kb)
    ledger_module.append_row(kb, "c", "d1", "hello", {})
    row = ledger_module.append_delete(kb, "c", "d1")
    assert row is not None
    assert row.op == ledger_module.OP_DELETE
    assert row.text == ""
    # Chain stays valid.
    assert ledger_module.verify_chain(kb).ok


def test_append_update_emits_update_op_row(ledger_module, tmp_path):
    kb = "memory"
    _make_kb(tmp_path, kb)
    ledger_module.append_row(kb, "c", "d1", "original", {"v": 1})
    row = ledger_module.append_update(
        kb, "c", "d1", new_text="modified", new_metadata={"v": 2},
    )
    assert row is not None
    assert row.op == ledger_module.OP_UPDATE
    assert row.text == "modified"
    assert row.metadata == {"v": 2}
    assert ledger_module.verify_chain(kb).ok


def test_hash_chain_valid_across_mixed_ops(ledger_module, tmp_path):
    """Add → update → delete → add chain must all hash-link correctly."""
    kb = "memory"
    _make_kb(tmp_path, kb)
    ledger_module.append_row(kb, "c", "d1", "v0", {})
    ledger_module.append_update(kb, "c", "d1", new_text="v1")
    ledger_module.append_delete(kb, "c", "d1")
    ledger_module.append_row(kb, "c", "d2", "fresh", {})
    result = ledger_module.verify_chain(kb)
    assert result.ok, result.to_dict()
    assert result.rows_seen == 4


def test_pre_tombstone_rows_remain_hash_valid(ledger_module, tmp_path):
    """A ledger written under the original schema (no op field) must
    still verify after the iter-2 schema extension. Tests the
    backward-compat hash logic in payload_for_hash."""
    kb = "memory"
    _make_kb(tmp_path, kb)
    # Write a row using the original-schema canonical encoding.
    path = ledger_module.ledger_path(kb)
    payload = {
        "ts": 100.0,
        "collection": "c",
        "doc_id": "old",
        "text": "pre-tombstone row",
        "metadata": {"k": "v"},
        "prev_hash": ledger_module.GENESIS_HASH,
    }
    h = ledger_module._compute_hash(ledger_module.GENESIS_HASH, payload)
    # Write WITHOUT the op field (mimics pre-iter-2 ledger).
    row = {**payload, "hash": h}
    with path.open("w") as f:
        f.write(ledger_module._canonical_json(row) + "\n")
    # Append a new add (post-iter-2 — has op field).
    ledger_module.append_row(kb, "c", "new", "post-tombstone row", {})
    result = ledger_module.verify_chain(kb)
    assert result.ok, result.to_dict()


def test_invalid_op_returns_none(ledger_module, tmp_path):
    kb = "memory"
    _make_kb(tmp_path, kb)
    row = ledger_module._append_op(kb, "bogus", "c", "d1", "text", {})
    assert row is None


# ────────────────────────────────────────────────────────────────────
#   Bulk hooks
# ────────────────────────────────────────────────────────────────────


def test_hook_collection_delete_appends_tombstones(ledger_module, tmp_path):
    kb = "episteme"
    _make_kb(tmp_path, kb)
    ledger_module.hook_collection_add(kb, "c", ["a", "b", "c"], ["x", "y", "z"], None)
    n = ledger_module.hook_collection_delete(kb, "c", ["a", "c"])
    assert n == 2
    rows = list(ledger_module.read_all(kb))
    assert [r.op for r in rows[-2:]] == [ledger_module.OP_DELETE] * 2
    assert ledger_module.verify_chain(kb).ok


def test_hook_collection_update_appends_updates(ledger_module, tmp_path):
    kb = "episteme"
    _make_kb(tmp_path, kb)
    ledger_module.hook_collection_add(kb, "c", ["a"], ["original"], [{"v": 1}])
    n = ledger_module.hook_collection_update(
        kb, "c", ["a"], documents=["modified"], metadatas=[{"v": 2}],
    )
    assert n == 1
    rows = list(ledger_module.read_all(kb))
    assert rows[-1].op == ledger_module.OP_UPDATE
    assert rows[-1].text == "modified"
    assert rows[-1].metadata == {"v": 2}


# ────────────────────────────────────────────────────────────────────
#   Replay folding
# ────────────────────────────────────────────────────────────────────


class _FakeCol:
    def __init__(self, name):
        self.name = name
        self.upserts = []
    def upsert(self, *, ids, documents, metadatas, embeddings):
        self.upserts.append({"ids": ids, "documents": documents, "metadatas": metadatas})
    def count(self):
        return sum(len(u["ids"]) for u in self.upserts)


class _FakeClient:
    def __init__(self):
        self.cols = {}
    def get_or_create_collection(self, name):
        if name not in self.cols:
            self.cols[name] = _FakeCol(name)
        return self.cols[name]


def _install_fake_chromadb(monkeypatch, fake_client):
    import sys
    class FakeMgr:
        def get_kb_client(self, name):
            return fake_client
        def embed(self, text):
            return [0.0] * 384
    monkeypatch.setitem(sys.modules, "app.memory.chromadb_manager", FakeMgr())


def test_replay_skips_deleted_docs(ledger_module, tmp_path, monkeypatch):
    kb = "memory"
    _make_kb(tmp_path, kb)
    ledger_module.append_row(kb, "c", "d1", "alive", {})
    ledger_module.append_row(kb, "c", "d2", "to-be-deleted", {})
    ledger_module.append_delete(kb, "c", "d2")

    fake = _FakeClient()
    _install_fake_chromadb(monkeypatch, fake)
    result = ledger_module.replay_kb(kb)
    assert result.ok
    # Only d1 should have been upserted.
    upserts = fake.cols["c"].upserts
    all_ids = []
    for u in upserts:
        all_ids.extend(u["ids"])
    assert all_ids == ["d1"]


def test_replay_folds_updates_over_adds(ledger_module, tmp_path, monkeypatch):
    kb = "memory"
    _make_kb(tmp_path, kb)
    ledger_module.append_row(kb, "c", "d1", "v0", {"version": 0})
    ledger_module.append_update(kb, "c", "d1", new_text="v1", new_metadata={"version": 1})
    ledger_module.append_update(kb, "c", "d1", new_metadata={"version": 2})

    fake = _FakeClient()
    _install_fake_chromadb(monkeypatch, fake)
    result = ledger_module.replay_kb(kb)
    assert result.ok
    # The folded row should have v1 text (last non-empty text) and
    # version=2 metadata (last update).
    upserts = fake.cols["c"].upserts
    assert len(upserts) == 1
    assert upserts[0]["ids"] == ["d1"]
    assert upserts[0]["documents"] == ["v1"]
    assert upserts[0]["metadatas"][0]["version"] == 2


def test_replay_resurrect_pattern_works(ledger_module, tmp_path, monkeypatch):
    """Delete-then-readd should land in the rebuild — operator can
    deliberately overwrite a previously-deleted id."""
    kb = "memory"
    _make_kb(tmp_path, kb)
    ledger_module.append_row(kb, "c", "d1", "first", {})
    ledger_module.append_delete(kb, "c", "d1")
    ledger_module.append_row(kb, "c", "d1", "second", {})

    fake = _FakeClient()
    _install_fake_chromadb(monkeypatch, fake)
    result = ledger_module.replay_kb(kb)
    assert result.ok
    upserts = fake.cols["c"].upserts
    assert upserts[0]["documents"] == ["second"]


def test_replay_orphan_update_is_skipped(ledger_module, tmp_path, monkeypatch):
    """Update with no prior add should be silently skipped — we can't
    invent the original text. The doc isn't in the rebuild output."""
    kb = "memory"
    _make_kb(tmp_path, kb)
    ledger_module.append_update(kb, "c", "orphan", new_text="ghost")

    fake = _FakeClient()
    _install_fake_chromadb(monkeypatch, fake)
    result = ledger_module.replay_kb(kb)
    assert result.ok
    # No collection should have been created (no rows).
    assert "c" not in fake.cols or not fake.cols["c"].upserts


# ────────────────────────────────────────────────────────────────────
#   Compaction
# ────────────────────────────────────────────────────────────────────


def test_compaction_skipped_when_too_small(ledger_module, tmp_path):
    kb = "memory"
    _make_kb(tmp_path, kb)
    for i in range(10):
        ledger_module.append_row(kb, "c", f"d{i}", f"text-{i}", {})
    result = ledger_module.compact_ledger(kb)
    assert result.skipped_reason == "below_min_rows"
    assert result.ok  # skipping isn't a failure


def test_compaction_skipped_when_reduction_small(ledger_module, tmp_path):
    """Ledger with 150 unique adds + 5 deletes → only 3.3% reduction.
    Below the 20% threshold — skip."""
    kb = "memory"
    _make_kb(tmp_path, kb)
    for i in range(150):
        ledger_module.append_row(kb, "c", f"d{i}", f"text-{i}", {})
    for i in range(5):
        ledger_module.append_delete(kb, "c", f"d{i}")
    result = ledger_module.compact_ledger(kb)
    assert "below_min_reduction" in result.skipped_reason
    assert result.ok


def test_compaction_folds_when_dirty(ledger_module, tmp_path):
    """150 adds + 50 updates of the same docs + 30 deletes → 33%
    reduction. Compaction proceeds."""
    kb = "memory"
    _make_kb(tmp_path, kb)
    for i in range(150):
        ledger_module.append_row(kb, "c", f"d{i}", f"v0-{i}", {})
    for i in range(50):
        ledger_module.append_update(kb, "c", f"d{i}", new_text=f"v1-{i}")
    for i in range(30):
        ledger_module.append_delete(kb, "c", f"d{i + 100}")

    rows_before = ledger_module.count_rows(kb)
    assert rows_before == 230

    result = ledger_module.compact_ledger(kb)
    assert result.ok, result.to_dict()
    assert result.skipped_reason == ""
    # 150 - 30 deleted = 120 surviving docs.
    assert result.rows_after == 120
    assert result.history_path  # snapshot was created

    # Live ledger now has 120 rows; history has the original 230.
    assert ledger_module.count_rows(kb) == 120
    history_path = Path(result.history_path)
    assert history_path.exists()
    with history_path.open() as f:
        history_rows = sum(1 for _ in f)
    assert history_rows == 230


def test_compaction_idempotent(ledger_module, tmp_path):
    """Re-running compaction on a clean ledger produces no further
    changes (below_min_reduction)."""
    kb = "memory"
    _make_kb(tmp_path, kb)
    for i in range(150):
        ledger_module.append_row(kb, "c", f"d{i}", f"v0-{i}", {})
    for i in range(50):
        ledger_module.append_delete(kb, "c", f"d{i}")
    r1 = ledger_module.compact_ledger(kb)
    # 150 adds + 50 deletes = 200 rows → 100 surviving → 100 dropped
    # (both the 50 add rows for deleted docs AND the 50 delete rows
    # collapse away).
    assert r1.ok and r1.rows_dropped == 100
    r2 = ledger_module.compact_ledger(kb)
    assert r2.ok
    assert "below_min_reduction" in r2.skipped_reason or r2.rows_dropped == 0


def test_compaction_preserves_hash_chain(ledger_module, tmp_path):
    """The compacted ledger must verify under verify_chain — new
    genesis link + rebuilt prev_hash chain."""
    kb = "memory"
    _make_kb(tmp_path, kb)
    for i in range(150):
        ledger_module.append_row(kb, "c", f"d{i}", f"text-{i}", {})
    for i in range(30):
        ledger_module.append_delete(kb, "c", f"d{i}")
    result = ledger_module.compact_ledger(kb)
    assert result.ok
    verify = ledger_module.verify_chain(kb)
    assert verify.ok, verify.to_dict()


def test_compaction_force_overrides_gates(ledger_module, tmp_path):
    """force=True compacts even on tiny ledgers."""
    kb = "memory"
    _make_kb(tmp_path, kb)
    for i in range(10):
        ledger_module.append_row(kb, "c", f"d{i}", f"text-{i}", {})
    for i in range(2):
        ledger_module.append_delete(kb, "c", f"d{i}")
    result = ledger_module.compact_ledger(kb, force=True)
    assert result.ok
    assert result.rows_after == 8


def test_list_history_after_compaction(ledger_module, tmp_path):
    kb = "memory"
    _make_kb(tmp_path, kb)
    for i in range(150):
        ledger_module.append_row(kb, "c", f"d{i}", f"text-{i}", {})
    for i in range(30):
        ledger_module.append_delete(kb, "c", f"d{i}")
    assert ledger_module.list_history(kb) == []
    ledger_module.compact_ledger(kb)
    assert len(ledger_module.list_history(kb)) == 1


# ────────────────────────────────────────────────────────────────────
#   Warm-spare manifest covers iter-2 surfaces
# ────────────────────────────────────────────────────────────────────


def test_manifest_covers_kb_ledger_and_postgres_backups():
    """Pin: the warm-spare manifest's allowlist must include the
    iter-2 KB recovery surfaces AND the Postgres/Neo4j dump dirs.
    Without these, replication misses the data we just spent a day
    making recoverable."""
    src = Path("/Users/andrus/BotArmy/crewai-team/app/warm_spare/manifest.py")
    if not src.exists():
        src = Path("app/warm_spare/manifest.py")
    if not src.exists():
        pytest.skip("manifest.py not on canonical paths")
    text = src.read_text()
    # The four KBs whose parent dirs aren't in the original list need
    # their ledger surfaces explicitly named.
    for kb in ("memory", "experiential", "philosophy", "knowledge"):
        assert f'"{kb}/.source_ledger.jsonl"' in text, (
            f"missing {kb}/.source_ledger.jsonl in warm-spare manifest"
        )
        assert f'"{kb}/.sqlite_snapshots"' in text, (
            f"missing {kb}/.sqlite_snapshots in warm-spare manifest"
        )
    # Postgres + Neo4j dumps from the host LaunchAgent.
    assert '"backups/postgres"' in text
    assert '"backups/neo4j"' in text
