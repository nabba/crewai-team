"""B4 (2026-05-18) — incremental verify_chain with per-KB checkpoint.

The daily daemon previously called ``verify_chain`` which walks from
genesis. On a 21 MB / 100k-row ledger that's a non-trivial daily CPU
cost. ``verify_chain_incremental`` persists
``(rows_verified, hash_at_that_row, first_row_hash)`` and resumes from
the tail. Tests pin:

  * first call walks from genesis and writes a checkpoint
  * second call resumes from the checkpoint (only appended rows seen)
  * compaction (new genesis) invalidates the checkpoint via the
    first_row_hash sentinel — next pass falls back to full walk
  * tampering in the appended tail is detected on the next pass
  * tampering in the already-checkpointed prefix is silently trusted
    by incremental, but the authoritative ``verify_chain`` still
    catches it (drills + dashboard + audit_chain_check use that one)
  * a deleted / shrunk live ledger invalidates the checkpoint
  * corrupt checkpoint file falls back to genesis walk
"""
from __future__ import annotations

import importlib
import json
import sqlite3
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


def _seed(sl, kb_name: str, n: int, start_id: int = 0):
    for i in range(n):
        sl.append_row(
            kb_name, "c", f"doc-{start_id + i}", f"text-{start_id + i}", {"i": start_id + i},
        )


def test_incremental_first_call_writes_checkpoint(ledger_module, tmp_path):
    kb = "memory"
    _make_kb(tmp_path, kb)
    _seed(ledger_module, kb, 5)

    cp_path = ledger_module._verify_checkpoint_path(kb)
    assert not cp_path.exists()

    res = ledger_module.verify_chain_incremental(kb)
    assert res.ok
    assert res.rows_seen == 5
    assert cp_path.exists()

    payload = json.loads(cp_path.read_text(encoding="utf-8"))
    assert payload["rows_verified"] == 5
    assert payload["hash_at"]
    assert payload["first_row_hash"]


def test_incremental_resumes_from_checkpoint(ledger_module, tmp_path, monkeypatch):
    kb = "memory"
    _make_kb(tmp_path, kb)
    _seed(ledger_module, kb, 5)
    ledger_module.verify_chain_incremental(kb)

    # Append more rows.
    _seed(ledger_module, kb, 3, start_id=5)

    # Spy on read_all by counting rows the iterator yields BEYOND the
    # skip-window. The implementation uses ``i < start_idx`` to skip,
    # so we instead instrument _compute_hash, which is only called on
    # rows that are actually being verified.
    calls = {"n": 0}
    original = ledger_module._compute_hash

    def spy(prev, payload):
        calls["n"] += 1
        return original(prev, payload)
    monkeypatch.setattr(ledger_module, "_compute_hash", spy)

    res = ledger_module.verify_chain_incremental(kb)
    assert res.ok
    assert res.rows_seen == 8
    # First call recomputed all 5; second pass must only recompute the
    # 3 new tail rows. The +1 in the original verifies the spy reads.
    assert calls["n"] == 3


def test_compaction_invalidates_checkpoint_via_first_row_hash(
    ledger_module, tmp_path,
):
    """Compaction creates a new genesis. The next incremental pass
    must detect via first_row_hash and walk from genesis."""
    kb = "memory"
    _make_kb(tmp_path, kb)
    # Seed with adds + deletes so compaction has work to do.
    _seed(ledger_module, kb, 120)
    for i in range(60):
        ledger_module.append_delete(kb, "c", f"doc-{i}")

    ledger_module.verify_chain_incremental(kb)
    before = json.loads(
        ledger_module._verify_checkpoint_path(kb).read_text(encoding="utf-8")
    )

    result = ledger_module.compact_ledger(kb, force=True)
    assert result.ok
    # After compaction the live ledger has a fresh chain.

    res = ledger_module.verify_chain_incremental(kb)
    assert res.ok

    after = json.loads(
        ledger_module._verify_checkpoint_path(kb).read_text(encoding="utf-8")
    )
    # first_row_hash sentinel changed; rows_verified reset and re-grew
    # against the post-compaction genesis.
    assert after["first_row_hash"] != before["first_row_hash"]
    assert after["rows_verified"] == res.rows_seen


def test_incremental_detects_tail_tampering(ledger_module, tmp_path):
    kb = "memory"
    _make_kb(tmp_path, kb)
    _seed(ledger_module, kb, 5)
    ledger_module.verify_chain_incremental(kb)

    # Append more rows, then tamper with one of them.
    _seed(ledger_module, kb, 3, start_id=5)
    p = ledger_module.ledger_path(kb)
    lines = p.read_text(encoding="utf-8").splitlines()
    target = json.loads(lines[6])  # row index 6 (first tail row)
    target["text"] = "TAMPERED"
    lines[6] = json.dumps(target, sort_keys=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    res = ledger_module.verify_chain_incremental(kb)
    assert not res.ok
    assert res.first_bad_row == 6


def test_full_verify_still_catches_prefix_tampering(ledger_module, tmp_path):
    """Incremental trusts the already-verified prefix. The
    authoritative full ``verify_chain`` (used by drills, dashboard,
    audit_chain_check) must still catch tampering there."""
    kb = "memory"
    _make_kb(tmp_path, kb)
    _seed(ledger_module, kb, 10)
    ledger_module.verify_chain_incremental(kb)

    # Tamper with row 2 (well inside the checkpointed prefix).
    p = ledger_module.ledger_path(kb)
    lines = p.read_text(encoding="utf-8").splitlines()
    target = json.loads(lines[2])
    target["text"] = "TAMPERED"
    lines[2] = json.dumps(target, sort_keys=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Incremental skips the prefix and reports OK (this is the
    # documented trade-off — incremental is the fast daily probe).
    fast = ledger_module.verify_chain_incremental(kb)
    assert fast.ok

    # Full verify catches it.
    full = ledger_module.verify_chain(kb)
    assert not full.ok
    assert full.first_bad_row == 2


def test_truncated_ledger_invalidates_checkpoint(ledger_module, tmp_path):
    kb = "memory"
    _make_kb(tmp_path, kb)
    _seed(ledger_module, kb, 10)
    ledger_module.verify_chain_incremental(kb)

    # Drop the last 5 rows on disk — checkpoint now points past EOF.
    p = ledger_module.ledger_path(kb)
    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join(lines[:5]) + "\n", encoding="utf-8")

    res = ledger_module.verify_chain_incremental(kb)
    # Should fall back to full walk against the 5 surviving rows.
    assert res.ok
    assert res.rows_seen == 5
    # Checkpoint was either reset or re-written with the new count.
    cp = json.loads(
        ledger_module._verify_checkpoint_path(kb).read_text(encoding="utf-8")
    )
    assert cp["rows_verified"] == 5


def test_corrupt_checkpoint_falls_back_to_genesis(ledger_module, tmp_path):
    kb = "memory"
    _make_kb(tmp_path, kb)
    _seed(ledger_module, kb, 5)

    cp_path = ledger_module._verify_checkpoint_path(kb)
    cp_path.parent.mkdir(parents=True, exist_ok=True)
    cp_path.write_text("not json", encoding="utf-8")

    res = ledger_module.verify_chain_incremental(kb)
    assert res.ok
    assert res.rows_seen == 5
    # Checkpoint rewritten cleanly.
    payload = json.loads(cp_path.read_text(encoding="utf-8"))
    assert payload["rows_verified"] == 5
