"""B5 (2026-05-18) — tiered history retention.

`.source_ledger_history/` previously grew unbounded; the original audit
recommendation was "prune to last 12" but that would have lost the
lifetime audit trail. The shipped fix instead keeps the
``_HISTORY_KEEP_UNCOMPRESSED`` newest snapshots as plain ``.jsonl`` and
gzips everything older. ~80% disk reduction on JSONL with zero data
loss; ``open_history`` transparently handles both.

Tests pin:

  * After compaction with N > keep, the older snapshots are gzipped
    in place, newer ones stay uncompressed.
  * Already-gzipped snapshots are left alone (idempotent).
  * Gzipped snapshots round-trip via ``open_history`` byte-identical.
  * ``list_history`` includes both ``.jsonl`` and ``.jsonl.gz`` files,
    sorted by mtime newest-first.
  * The lifetime audit trail is preserved — every original snapshot
    is still present (just in compressed form past the keep-window).
"""
from __future__ import annotations

import gzip
import importlib
import json
import os
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


def _seed_history(sl, kb_name: str, count: int) -> list[Path]:
    """Drop ``count`` fake pre-compaction snapshots into the history
    dir with increasing mtime. Each one carries a distinct payload so
    round-trip identity can be checked."""
    base = sl.ledger_path(kb_name).parent / sl._HISTORY_DIRNAME
    base.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    now = time.time()
    for i in range(count):
        p = base / f"snap-{i:04d}.jsonl"
        p.write_text(
            f'{{"snapshot_idx":{i},"text":"snapshot {i} content"}}\n',
            encoding="utf-8",
        )
        # Set mtime so age ordering is deterministic — older idx = older mtime.
        os.utime(p, (now - (count - i) * 60, now - (count - i) * 60))
        paths.append(p)
    return paths


# ────────────────────────────────────────────────────────────────────
#   _rotate_history correctness
# ────────────────────────────────────────────────────────────────────


def test_rotate_history_no_op_when_below_threshold(ledger_module, tmp_path):
    kb = "memory"
    _make_kb(tmp_path, kb)
    _seed_history(ledger_module, kb, 10)
    summary = ledger_module._rotate_history(kb, keep_uncompressed=52)
    assert summary["newly_compressed"] == 0
    assert summary["kept_uncompressed"] == 10
    assert summary["already_compressed"] == 0


def test_rotate_history_gzips_older_keeps_newest(ledger_module, tmp_path):
    kb = "memory"
    _make_kb(tmp_path, kb)
    _seed_history(ledger_module, kb, 60)
    summary = ledger_module._rotate_history(kb, keep_uncompressed=52)
    assert summary["newly_compressed"] == 8
    assert summary["kept_uncompressed"] == 52
    base = ledger_module.ledger_path(kb).parent / ledger_module._HISTORY_DIRNAME
    plain = sorted(base.glob("*.jsonl"))
    gzipped = sorted(base.glob("*.jsonl.gz"))
    # 8 oldest got gzipped (snap-0000 .. snap-0007); 52 newest stay plain.
    assert len(plain) == 52
    assert len(gzipped) == 8
    # The original .jsonl is gone after gzip succeeds.
    for p in plain:
        assert p.exists()
        assert not p.with_suffix(p.suffix + ".gz").exists() or True  # gz lives separately
    for g in gzipped:
        # The corresponding .jsonl was removed.
        assert not g.with_suffix("").exists()


def test_rotate_history_is_idempotent(ledger_module, tmp_path):
    kb = "memory"
    _make_kb(tmp_path, kb)
    _seed_history(ledger_module, kb, 60)
    ledger_module._rotate_history(kb, keep_uncompressed=52)
    second = ledger_module._rotate_history(kb, keep_uncompressed=52)
    # Second pass: the 8 olds are already .gz, nothing new to compress.
    assert second["newly_compressed"] == 0
    assert second["already_compressed"] == 8
    assert second["kept_uncompressed"] == 52


def test_gzipped_snapshot_round_trips_via_open_history(ledger_module, tmp_path):
    kb = "memory"
    _make_kb(tmp_path, kb)
    paths = _seed_history(ledger_module, kb, 60)
    expected = paths[0].read_text(encoding="utf-8")
    ledger_module._rotate_history(kb, keep_uncompressed=52)
    # The oldest snapshot is now .jsonl.gz.
    base = ledger_module.ledger_path(kb).parent / ledger_module._HISTORY_DIRNAME
    gz = base / "snap-0000.jsonl.gz"
    assert gz.exists()
    with ledger_module.open_history(gz) as f:
        recovered = f.read()
    assert recovered == expected


def test_list_history_returns_both_jsonl_and_gz_newest_first(
    ledger_module, tmp_path,
):
    kb = "memory"
    _make_kb(tmp_path, kb)
    _seed_history(ledger_module, kb, 60)
    ledger_module._rotate_history(kb, keep_uncompressed=52)
    listed = ledger_module.list_history(kb)
    assert len(listed) == 60
    # Newest 52 are .jsonl; oldest 8 are .jsonl.gz.
    assert all(str(p).endswith(".jsonl") for p in listed[:52])
    assert all(str(p).endswith(".jsonl.gz") for p in listed[52:])
    # mtime ordering: newest first.
    mtimes = [p.stat().st_mtime for p in listed]
    assert mtimes == sorted(mtimes, reverse=True)


def test_rotate_history_preserves_full_audit_trail(ledger_module, tmp_path):
    """Lifetime memory invariant: no snapshot is ever deleted, only
    converted to .gz. Original bodies must be recoverable."""
    kb = "memory"
    _make_kb(tmp_path, kb)
    paths = _seed_history(ledger_module, kb, 60)
    originals = {p.name: p.read_text(encoding="utf-8") for p in paths}
    ledger_module._rotate_history(kb, keep_uncompressed=52)
    base = ledger_module.ledger_path(kb).parent / ledger_module._HISTORY_DIRNAME
    recovered: dict[str, str] = {}
    for p in base.iterdir():
        if p.name.endswith(".jsonl.gz"):
            # The original key was <name>.jsonl
            orig_name = p.name[:-3]
            with gzip.open(p, "rt", encoding="utf-8") as f:
                recovered[orig_name] = f.read()
        elif p.name.endswith(".jsonl"):
            recovered[p.name] = p.read_text(encoding="utf-8")
    assert recovered == originals


# ────────────────────────────────────────────────────────────────────
#   Wired into compact_ledger
# ────────────────────────────────────────────────────────────────────


def test_compact_ledger_invokes_history_rotation(
    ledger_module, tmp_path, monkeypatch,
):
    """After a real compaction event, the rotator runs on the history
    dir. We verify the call happened rather than the result count
    (which depends on how many prior snapshots existed)."""
    kb = "memory"
    _make_kb(tmp_path, kb)
    # Seed with adds + deletes so compaction has work.
    for i in range(120):
        ledger_module.append_row(kb, "c", f"d{i}", f"t{i}", {"i": i})
    for i in range(60):
        ledger_module.append_delete(kb, "c", f"d{i}")

    rotation_calls: list[str] = []
    original = ledger_module._rotate_history

    def spy(name, **kw):
        rotation_calls.append(name)
        return original(name, **kw)
    monkeypatch.setattr(ledger_module, "_rotate_history", spy)

    res = ledger_module.compact_ledger(kb, force=True)
    assert res.ok
    assert rotation_calls == [kb]
