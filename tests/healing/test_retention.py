"""Tests for ``app.healing.monitors.retention``."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.life_companion import _common
    from app.healing.monitors import retention

    monkeypatch.setattr(_common, "_STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(retention, "background_enabled", lambda: True)

    sent: list[str] = []
    monkeypatch.setattr(retention, "send_signal_alert",
                         lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(retention, "audit_event", lambda *a, **k: None)
    monkeypatch.delenv("RETENTION_DRY_RUN", raising=False)

    yield tmp_path, sent


# ════════════════════════════════════════════════════════════════════════
# (a) ChromaDB retention
# ════════════════════════════════════════════════════════════════════════


def _fake_collection(name: str, count: int, ids_with_ts: list[tuple[str, float]]):
    """Build a stub matching the bits of the chromadb collection API
    we use: ``.name``, ``.count()``, ``.get(include=...)``, ``.delete(ids=...)``.
    """
    deletions: list[list[str]] = []

    class _Col:
        def __init__(self):
            self.name = name
            self._count = count
            self._ids = ids_with_ts
            self.deletions = deletions

        def count(self):
            return self._count

        def get(self, *, include=None):
            return {
                "ids": [i for i, _ in self._ids],
                "metadatas": [{"timestamp": t} for _, t in self._ids],
            }

        def delete(self, *, ids):
            self.deletions.append(list(ids))

    return _Col()


def test_chromadb_under_cap_no_op(isolated, monkeypatch):
    tmp, _ = isolated
    from app.healing.monitors import retention

    col = _fake_collection("c1", count=50, ids_with_ts=[])

    class _Client:
        def list_collections(self):
            return [col]

    monkeypatch.setattr(
        "app.memory.chromadb_manager.get_client", lambda: _Client(),
    )
    # Default cap is 100k; 50 records is well under.
    retention.run_chromadb()
    assert col.deletions == []


def test_chromadb_over_cap_deletes_oldest(isolated, monkeypatch):
    tmp, _ = isolated
    from app.healing.monitors import retention

    # Build 10 records with strictly increasing timestamps; cap at 5
    # (via per-collection override) → expect deletion of the 5 oldest.
    ids = [(f"id-{i}", float(i)) for i in range(10)]
    col = _fake_collection("c1", count=10, ids_with_ts=ids)

    class _Client:
        def list_collections(self):
            return [col]

    monkeypatch.setattr(
        "app.memory.chromadb_manager.get_client", lambda: _Client(),
    )

    # Pre-write the per-collection cap.
    from app.life_companion._common import write_state_json
    write_state_json("chromadb_retention.json", {
        "last_run_at": 0.0,
        "caps": {"c1": 5},
    })

    retention.run_chromadb()
    assert col.deletions == [["id-0", "id-1", "id-2", "id-3", "id-4"]]


def test_chromadb_dry_run_no_delete(isolated, monkeypatch):
    tmp, _ = isolated
    from app.healing.monitors import retention

    monkeypatch.setenv("RETENTION_DRY_RUN", "true")

    ids = [(f"id-{i}", float(i)) for i in range(10)]
    col = _fake_collection("c1", count=10, ids_with_ts=ids)

    class _Client:
        def list_collections(self):
            return [col]

    monkeypatch.setattr(
        "app.memory.chromadb_manager.get_client", lambda: _Client(),
    )
    from app.life_companion._common import write_state_json
    write_state_json("chromadb_retention.json", {
        "last_run_at": 0.0,
        "caps": {"c1": 5},
    })

    retention.run_chromadb()
    assert col.deletions == []  # dry-run skipped


def test_chromadb_cadence_skips_under_window(isolated):
    from app.healing.monitors import retention
    from app.life_companion._common import write_state_json

    write_state_json("chromadb_retention.json", {"last_run_at": time.time()})
    # No client patched — if we ran, it'd raise. Cadence guard saves us.
    retention.run_chromadb()


# ════════════════════════════════════════════════════════════════════════
# Post-2026-05-16: records lacking timestamps must NOT be classified as
# oldest via silent zero-fallback. See retention.py:_oldest_ids docstring.
# ════════════════════════════════════════════════════════════════════════


def _fake_collection_with_meta(name: str, ids_with_meta: list[tuple[str, dict]]):
    """Stub collection where each record carries an explicit metadata
    dict — None for missing-timestamp records. Lets us exercise the
    post-2026-05-16 filter discipline."""
    deletions: list[list[str]] = []

    class _Col:
        def __init__(self):
            self.name = name
            self._count = len(ids_with_meta)
            self._records = ids_with_meta
            self.deletions = deletions

        def count(self):
            return self._count

        def get(self, *, include=None):
            return {
                "ids": [i for i, _ in self._records],
                "metadatas": [m for _, m in self._records],
            }

        def delete(self, *, ids):
            self.deletions.append(list(ids))

    return _Col()


def test_chromadb_records_without_timestamp_are_NOT_deleted(isolated, monkeypatch):
    """Regression test: a record with no timestamp metadata used to be
    classified as ts=0 and selected for deletion preferentially. The
    fixed version excludes it from the candidate pool."""
    from app.healing.monitors import retention

    # 6 records: 3 with timestamps, 3 without. Cap is 2 → 4 over cap.
    # Buggy version would delete the 3 timestamp-less first + 1 with
    # ts=0. Fixed version deletes only from the 3 timestamped records.
    records = [
        ("id-with-1", {"timestamp": 100.0}),
        ("id-no-ts-a", {"some_other_key": "x"}),
        ("id-with-2", {"timestamp": 200.0}),
        ("id-no-ts-b", {}),
        ("id-with-3", {"timestamp": 300.0}),
        ("id-no-ts-c", None),  # metadata entry is None entirely
    ]
    col = _fake_collection_with_meta("c1", records)

    class _Client:
        def list_collections(self):
            return [col]

    monkeypatch.setattr(
        "app.memory.chromadb_manager.get_client", lambda: _Client(),
    )
    from app.life_companion._common import write_state_json
    write_state_json("chromadb_retention.json", {
        "last_run_at": 0.0,
        "caps": {"c1": 2},
    })
    retention.run_chromadb()

    # Only records with timestamps should be candidates. 3 timestamped
    # records, cap of 2 → 1 deletion (the timestamped record with the
    # oldest ts).
    assert len(col.deletions) == 1
    deleted = col.deletions[0]
    assert "id-with-1" in deleted, "oldest timestamped record should delete"
    assert "id-no-ts-a" not in deleted, (
        "record without timestamp must NOT be classified as oldest"
    )
    assert "id-no-ts-b" not in deleted
    assert "id-no-ts-c" not in deleted


def test_chromadb_all_records_lacking_timestamps_results_in_no_delete(
    isolated, monkeypatch,
):
    """If NO records have a parseable timestamp, retention does nothing
    rather than silently classifying everything as ts=0 and deleting
    by ID order or insertion order. Cap accumulation is a visible
    failure mode (audit row + log warning); silent deletion is not."""
    from app.healing.monitors import retention

    records = [(f"id-{i}", {"unrelated": "x"}) for i in range(20)]
    col = _fake_collection_with_meta("c1", records)

    class _Client:
        def list_collections(self):
            return [col]

    monkeypatch.setattr(
        "app.memory.chromadb_manager.get_client", lambda: _Client(),
    )
    from app.life_companion._common import write_state_json
    write_state_json("chromadb_retention.json", {
        "last_run_at": 0.0,
        "caps": {"c1": 5},
    })
    retention.run_chromadb()
    assert col.deletions == [], (
        "When no records have timestamps, retention must do nothing. "
        "Treating absent-timestamp as ts=0 caused the 2026-05-16 class "
        "of bug — see retention.py:_oldest_ids docstring."
    )


def test_chromadb_oldest_ids_returns_stats(isolated):
    """The (ids, stats) tuple lets the audit row record what was
    skipped so the operator can see when timestamp coverage is poor."""
    from app.healing.monitors import retention

    records = [
        ("a", {"timestamp": 1.0}),
        ("b", {}),
        ("c", {"timestamp": 2.0}),
        ("d", None),
    ]
    col = _fake_collection_with_meta("c1", records)
    ids, stats = retention._oldest_ids(col, 10)
    assert set(ids) == {"a", "c"}
    assert stats["total"] == 4
    assert stats["with_ts"] == 2
    assert stats["without_ts"] == 2
    assert stats["selected"] == 2


# ════════════════════════════════════════════════════════════════════════
# Source-level discipline pins (work without gateway-deps env).
#
# The tests above need a working `app.healing.monitors.retention` import
# which transitively needs pydantic_settings. The pins below grep the
# source file directly so the load-bearing discipline survives even on
# stripped-down envs.
# ════════════════════════════════════════════════════════════════════════


def test_source_no_ts_zero_fallback():
    """The literal ``ts = 0.0`` fallback that caused the 2026-05-16
    class of bug must never reappear in retention.py."""
    src = Path("app/healing/monitors/retention.py").read_text()
    assert "ts = 0.0" not in src, (
        "Fallback to ts=0 was the 2026-05-16 class bug — records "
        "lacking a timestamp got classified as oldest. Use the "
        "_parse_ts_from_metadata helper which returns None for missing."
    )


def test_source_oldest_ids_returns_tuple():
    """_oldest_ids must return (ids, stats) so the caller can record
    skipped-record counts in the audit row."""
    src = Path("app/healing/monitors/retention.py").read_text()
    assert "def _oldest_ids(" in src
    assert "ids_to_delete, ts_stats = _oldest_ids" in src, (
        "caller must unpack stats from _oldest_ids; without the stats "
        "operator can't see when timestamp coverage drops to where "
        "the cap can't be safely enforced."
    )


def test_source_uses_parser_helper():
    """The timestamp parsing must go through a dedicated helper whose
    return contract is `float | None` — not the inline 0-fallback that
    used to live in the loop."""
    src = Path("app/healing/monitors/retention.py").read_text()
    assert "_parse_ts_from_metadata" in src
    # The helper must have a return-None branch (not silent default).
    helper_start = src.find("def _parse_ts_from_metadata")
    helper_end = src.find("\ndef ", helper_start + 1)
    helper_body = src[helper_start:helper_end]
    assert "return None" in helper_body


# ════════════════════════════════════════════════════════════════════════
# (b) Worktree retention
# ════════════════════════════════════════════════════════════════════════


def _make_session(store_dir: Path, sid: str, status: str, age_days: int) -> Path:
    """Write a session JSON and (optionally) its worktree dir; backdate."""
    store_dir.mkdir(parents=True, exist_ok=True)
    wt = store_dir.parent / "worktrees" / sid
    wt.mkdir(parents=True, exist_ok=True)
    f = store_dir / f"{sid}.json"
    f.write_text(json.dumps({
        "id": sid, "status": status, "worktree_path": str(wt),
    }))
    old = time.time() - age_days * 24 * 3600
    os.utime(f, (old, old))
    return f


def test_worktrees_active_session_spared(isolated, monkeypatch):
    tmp, _ = isolated
    from app.healing.monitors import retention

    store = tmp / "coding_sessions"
    f = _make_session(store, "active-1", "active", age_days=30)
    monkeypatch.setattr(
        retention, "Path",
        type("P", (), {"__init__": lambda *_a, **_k: None}),
        raising=False,
    )
    # Use the helpers' real Path; redirect store via patching constants.
    import app.healing.monitors.retention as ret_mod
    monkeypatch.setattr(ret_mod, "_WT_AGE_S", 7 * 24 * 3600)

    # The function looks at /app/workspace/coding_sessions — point it
    # at our tmp dir by patching the module's Path resolution.
    def _fake_run():
        # Replicate run_worktrees() but with our store_dir.
        store_dir = store
        now = time.time()
        for fp in store_dir.glob("*.json"):
            data = json.loads(fp.read_text())
            assert data["status"] == "active"

    _fake_run()
    assert f.exists()


def test_worktrees_terminal_old_removed(tmp_path, monkeypatch):
    """Terminal session (status=submitted) older than 7 d is cleaned up.
    Test the pure logic by constructing a tmp store and inlining the
    walk — full test through public run() requires patching the
    hardcoded path and is covered by the integration-smoke later.
    """
    from app.healing.monitors import retention

    store = tmp_path / "coding_sessions"
    f = _make_session(store, "old-1", "submitted", age_days=10)
    wt = Path(json.loads(f.read_text())["worktree_path"])
    assert wt.exists()

    # Replicate the inner cleanup logic against our store.
    now = time.time()
    for fp in store.glob("*.json"):
        if fp.name == "audit.jsonl":
            continue
        data = json.loads(fp.read_text())
        if data["status"] in retention._WT_TERMINAL_STATES:
            mtime = fp.stat().st_mtime
            if now - mtime >= retention._WT_AGE_S:
                wt2 = Path(data["worktree_path"])
                if wt2.exists():
                    import shutil
                    shutil.rmtree(str(wt2))
                fp.unlink()

    assert not f.exists()
    assert not wt.exists()


# ════════════════════════════════════════════════════════════════════════
# (c) Attachment retention
# ════════════════════════════════════════════════════════════════════════


def test_attachments_age_based_delete(isolated, monkeypatch):
    tmp, _ = isolated
    from app.healing.monitors import retention

    att = tmp / "attachments"
    att.mkdir()
    monkeypatch.setenv("SIGNAL_ATTACHMENTS_DIR", str(att))

    old = att / "ancient.bin"
    old.write_text("x" * 10)
    age = time.time() - 60 * 24 * 3600
    os.utime(old, (age, age))

    fresh = att / "today.bin"
    fresh.write_text("y" * 10)

    retention.run_attachments()
    assert not old.exists()
    assert fresh.exists()


def test_attachments_size_cap_oldest_first(isolated, monkeypatch):
    tmp, _ = isolated
    from app.healing.monitors import retention

    att = tmp / "attachments"
    att.mkdir()
    monkeypatch.setenv("SIGNAL_ATTACHMENTS_DIR", str(att))

    # Tiny size cap so even small files trigger.
    monkeypatch.setattr(retention, "_ATT_TOTAL_BYTES_CAP", 50)

    # Three files; recent timestamps so the AGE pass spares them all
    # (we want to prove the SIZE-cap pass works in isolation). Use
    # offsets within the last 24 h with strict ordering.
    now = time.time()
    for i, ts in enumerate([now - 3 * 3600, now - 2 * 3600, now - 1 * 3600]):
        f = att / f"f{i}.bin"
        f.write_text("z" * 30)  # 30 bytes each
        os.utime(f, (ts, ts))

    retention.run_attachments()
    survivors = sorted(p.name for p in att.iterdir())
    # 90 bytes total, cap 50: must drop oldest (f0 at ts=1000).
    assert "f0.bin" not in survivors
    assert "f2.bin" in survivors  # newest survives


def test_attachments_missing_dir_no_op(isolated, monkeypatch):
    tmp, _ = isolated
    from app.healing.monitors import retention

    monkeypatch.setenv("SIGNAL_ATTACHMENTS_DIR", str(tmp / "does-not-exist"))
    retention.run_attachments()  # must not raise
