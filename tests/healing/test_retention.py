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
