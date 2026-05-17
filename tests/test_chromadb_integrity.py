"""PROGRAM §55 — ChromaDB integrity protection tests.

Built in response to the dual-writer SQLite corruption events of
2026-04-25 and 2026-05-17 that wiped the gateway's ``memory/`` KB.
Root cause was the orphaned chromadb container (removed in this PR).
This suite tests the defense-in-depth layer that catches the next
class of damage even after the dual-writer is gone.

Coverage:
  * chromadb_kbs discovery (with quarantine-dir exclusion)
  * enforce_wal_mode idempotency + journal_mode persistence
  * integrity_check on a known-good DB
  * integrity_check on a deliberately-damaged DB
  * daily_snapshot creates atomic backup + prunes old snapshots
  * quarantine_kb renames + creates fresh slate + emits ledger
  * boot_integrity_scan composes layer end-to-end with mocked KBs
  * monitor probe respects cadence guard

Each test isolates state in a tmp_path. No real chromadb or postgres
dependency — we use raw SQLite for the integrity surface (chromadb
just uses SQLite under the hood).
"""
from __future__ import annotations

import importlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ────────────────────────────────────────────────────────────────────
#   Helpers
# ────────────────────────────────────────────────────────────────────


def _make_chroma_like_sqlite(path: Path) -> None:
    """Create a SQLite file shaped like chromadb's metadata DB.

    We don't need the full chromadb schema — just enough that
    ``PRAGMA integrity_check`` has structure to walk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS collections "
            "(id TEXT PRIMARY KEY, name TEXT, dimension INTEGER)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(id INTEGER PRIMARY KEY, segment_id TEXT, embedding BLOB)"
        )
        for i in range(50):
            conn.execute(
                "INSERT INTO embeddings (segment_id, embedding) VALUES (?, ?)",
                (f"seg-{i // 10}", b"\x00" * 16),
            )
        conn.commit()
    finally:
        conn.close()


def _corrupt_sqlite(path: Path) -> None:
    """Deliberately damage the SQLite file at a known offset so
    ``PRAGMA integrity_check`` reports an error.

    Overwriting a btree page header is the most reliable way to
    produce a "database disk image is malformed" or similar error
    that integrity_check catches.
    """
    raw = path.read_bytes()
    # Page size for new SQLite DBs is 4096. Page 1 is the header;
    # page 2+ are btree. Trash page 2 to break a btree node.
    if len(raw) < 8192:
        # File too small — overwrite from byte 100 (past the header).
        bad = bytearray(raw)
        for i in range(100, min(200, len(bad))):
            bad[i] = 0xFF
        path.write_bytes(bytes(bad))
        return
    bad = bytearray(raw)
    # Trash bytes 4096..4196 — start of page 2 btree header.
    for i in range(4096, 4196):
        bad[i] = 0xFF
    path.write_bytes(bytes(bad))


@pytest.fixture
def fake_workspace(tmp_path, monkeypatch):
    """Builds a fake workspace dir with several KBs.

    Layout:
      tmp/workspace/memory/chroma.sqlite3            (healthy)
      tmp/workspace/episteme/chroma.sqlite3          (healthy)
      tmp/workspace/memory.corrupt_20260425/...      (quarantined — must be skipped)
    """
    ws = tmp_path / "workspace"
    ws.mkdir()
    for kb in ("memory", "episteme"):
        _make_chroma_like_sqlite(ws / kb / "chroma.sqlite3")
    # Quarantined dir — should be filtered out by chromadb_kbs.
    quar = ws / "memory.corrupt_20260425_221402"
    quar.mkdir()
    _make_chroma_like_sqlite(quar / "chroma.sqlite3")
    return ws


@pytest.fixture
def integrity_module(fake_workspace, monkeypatch):
    """Reload the integrity module with a patched WORKSPACE_ROOT."""
    # Patch app.paths.WORKSPACE_ROOT — the integrity module reads it.
    import app.paths as paths
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", fake_workspace)
    import app.memory.chromadb_integrity as ci
    importlib.reload(ci)
    return ci


# ────────────────────────────────────────────────────────────────────
#   chromadb_kbs discovery
# ────────────────────────────────────────────────────────────────────


def test_chromadb_kbs_finds_live_and_skips_quarantined(integrity_module, fake_workspace):
    kbs = integrity_module.chromadb_kbs(fake_workspace)
    names = sorted(p.parent.name for p in kbs)
    assert names == ["episteme", "memory"], names
    # Quarantine dirs must be excluded.
    assert not any("corrupt_" in p.parent.name for p in kbs)


def test_chromadb_kbs_skips_bak_and_backup_directories(integrity_module, fake_workspace):
    for name in ("memory.bak_20260101", "memory_backup", "memory.backup_test"):
        (fake_workspace / name).mkdir()
        _make_chroma_like_sqlite(fake_workspace / name / "chroma.sqlite3")
    kbs = integrity_module.chromadb_kbs(fake_workspace)
    names = sorted(p.parent.name for p in kbs)
    assert names == ["episteme", "memory"], names


# ────────────────────────────────────────────────────────────────────
#   WAL enforcement
# ────────────────────────────────────────────────────────────────────


def test_enforce_wal_mode_sets_wal_and_full_sync(integrity_module, fake_workspace):
    db = fake_workspace / "memory" / "chroma.sqlite3"
    result = integrity_module.enforce_wal_mode(db)
    assert result["ok"], result
    assert str(result["mode_after"]).lower() == "wal"
    assert int(result["sync_after"]) == 2
    # Persistence: a fresh connection should see WAL too.
    conn = sqlite3.connect(str(db))
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    assert mode.lower() == "wal"


def test_enforce_wal_mode_idempotent(integrity_module, fake_workspace):
    db = fake_workspace / "memory" / "chroma.sqlite3"
    first = integrity_module.enforce_wal_mode(db)
    second = integrity_module.enforce_wal_mode(db)
    assert first["ok"] and second["ok"]
    # Re-running on an already-WAL DB still returns WAL.
    assert str(second["mode_after"]).lower() == "wal"


# ────────────────────────────────────────────────────────────────────
#   Integrity check
# ────────────────────────────────────────────────────────────────────


def test_integrity_check_passes_on_healthy_db(integrity_module, fake_workspace):
    db = fake_workspace / "memory" / "chroma.sqlite3"
    assert integrity_module.integrity_check(db) == "ok"


def test_integrity_check_catches_corruption(integrity_module, fake_workspace):
    db = fake_workspace / "memory" / "chroma.sqlite3"
    _corrupt_sqlite(db)
    verdict = integrity_module.integrity_check(db)
    assert verdict != "ok", f"corruption not detected, verdict={verdict!r}"
    # Damage should manifest as either a structural integrity-check
    # message or an open-time failure — both count.
    assert any(
        kw in verdict.lower()
        for kw in ("malformed", "page", "btree", "row", "corrupt", "open_failed", "image")
    ), verdict


def test_integrity_check_reports_missing(integrity_module, tmp_path):
    assert integrity_module.integrity_check(tmp_path / "no_such.db") == "missing"


# ────────────────────────────────────────────────────────────────────
#   Snapshot
# ────────────────────────────────────────────────────────────────────


def test_daily_snapshot_creates_atomic_backup(integrity_module, fake_workspace):
    db = fake_workspace / "memory" / "chroma.sqlite3"
    result = integrity_module.daily_snapshot(db)
    assert result["ok"], result
    snap_path = Path(result["snapshot_path"])
    assert snap_path.exists()
    assert snap_path.parent.name == ".sqlite_snapshots"
    # Snapshot is a valid SQLite file we can open.
    conn = sqlite3.connect(str(snap_path))
    try:
        rows = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    finally:
        conn.close()
    assert rows == 50


def test_daily_snapshot_prunes_old_snapshots(integrity_module, fake_workspace):
    db = fake_workspace / "memory" / "chroma.sqlite3"
    snap_dir = db.parent / ".sqlite_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    # Drop a fake old snapshot.
    old = snap_dir / "20200101T000000Z.db"
    old.write_bytes(b"SQLite format 3\x00")
    # Backdate to 30 days old.
    old_ts = time.time() - 30 * 86400
    import os as _os
    _os.utime(old, (old_ts, old_ts))
    result = integrity_module.daily_snapshot(db, retention_days=7)
    assert result["ok"], result
    assert str(old) in result["removed"], result
    assert not old.exists()


def test_daily_snapshot_missing_source(integrity_module, tmp_path):
    result = integrity_module.daily_snapshot(tmp_path / "absent.db")
    assert not result["ok"]
    assert result["error"] == "source_missing"


# ────────────────────────────────────────────────────────────────────
#   Quarantine
# ────────────────────────────────────────────────────────────────────


def test_quarantine_kb_renames_and_creates_fresh(integrity_module, fake_workspace, monkeypatch):
    kb_dir = fake_workspace / "memory"
    assert kb_dir.is_dir()
    # Stub out ledger to avoid touching real workspace.
    calls = []
    def fake_emit(kind, summary, detail, actor):
        calls.append({"kind": kind, "summary": summary, "detail": detail, "actor": actor})
    monkeypatch.setattr(integrity_module, "_emit_ledger_event", fake_emit)

    result = integrity_module.quarantine_kb(kb_dir, reason="test_reason")
    assert result["ok"], result
    quar_path = Path(result["quarantine_path"])
    assert quar_path.exists()
    assert "corrupt_" in quar_path.name
    # Original dir was re-created empty.
    assert kb_dir.is_dir()
    assert not (kb_dir / "chroma.sqlite3").exists()
    # Ledger event fired.
    assert calls, "expected ledger event"
    assert calls[0]["kind"] == "chromadb_corruption"
    assert calls[0]["detail"]["reason"] == "test_reason"
    assert calls[0]["actor"] == "chromadb_integrity"


def test_quarantine_kb_handles_missing_dir(integrity_module, tmp_path):
    result = integrity_module.quarantine_kb(
        tmp_path / "absent", reason="test",
    )
    assert not result["ok"]
    assert result["error"] == "kb_dir_missing"


# ────────────────────────────────────────────────────────────────────
#   Boot orchestrator end-to-end
# ────────────────────────────────────────────────────────────────────


def test_boot_integrity_scan_clean_workspace(integrity_module, fake_workspace, monkeypatch):
    """Healthy workspace — every KB passes, no quarantines, no replays."""
    monkeypatch.setattr(integrity_module, "_send_corruption_alert", lambda *a, **k: None)
    summary = integrity_module.boot_integrity_scan()
    assert "skipped" not in summary
    assert set(summary["integrity_results"].keys()) == {"memory", "episteme"}
    assert all(v == "ok" for v in summary["integrity_results"].values())
    assert summary["quarantines"] == {}
    assert summary["replays"] == {}


def test_boot_integrity_scan_quarantines_corrupt(integrity_module, fake_workspace, monkeypatch):
    """Corrupt memory KB → quarantined; auto-replay attempted (best-effort).

    Replay needs postgres; in tests we mock _postgres_connect to None
    so it gracefully returns an error rather than trying to connect.
    """
    monkeypatch.setattr(integrity_module, "_send_corruption_alert", lambda *a, **k: None)
    monkeypatch.setattr(integrity_module, "_emit_ledger_event", lambda *a, **k: None)
    monkeypatch.setattr(integrity_module, "_postgres_connect", lambda: None)
    db = fake_workspace / "memory" / "chroma.sqlite3"
    _corrupt_sqlite(db)
    summary = integrity_module.boot_integrity_scan()
    assert "memory" in summary["quarantines"]
    assert summary["quarantines"]["memory"]["ok"]
    # Fresh empty memory dir was re-created.
    assert (fake_workspace / "memory").is_dir()
    # Auto-replay was attempted (best-effort).
    assert "memory" in summary["replays"]
    # Replay returns an error code — exact code depends on which
    # dependency is missing in the test env (chromadb / psycopg /
    # runtime_settings). We assert the overall shape (failure code
    # present, no rows added) rather than pinning a specific code.
    replay = summary["replays"]["memory"]
    assert replay.get("ok") is False or replay.get("total_added", 0) == 0
    assert replay.get("error"), f"expected an error code, got {replay!r}"


def test_boot_integrity_scan_disabled_no_op(integrity_module, monkeypatch):
    """When master switch is OFF, scan returns skipped."""
    monkeypatch.setattr(
        integrity_module,
        "_gate",
        lambda name, default=True: False if name == "chromadb_boot_integrity_check_enabled" else default,
    )
    summary = integrity_module.boot_integrity_scan()
    assert summary.get("skipped") == "boot_integrity_check_disabled"


# ────────────────────────────────────────────────────────────────────
#   Monitor probe
# ────────────────────────────────────────────────────────────────────


def test_monitor_respects_cadence(monkeypatch, tmp_path, fake_workspace):
    """First run fires; second run within 23h is a no-op."""
    pytest.importorskip("pydantic_settings")  # gateway-deps shim
    # Patch the workspace + the state path so we don't touch real files.
    import app.paths as paths
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", fake_workspace)
    # Redirect life_companion state writes to tmp_path.
    import app.life_companion._common as common
    monkeypatch.setattr(common, "_STATE_DIR", tmp_path / "lc_state")

    import app.memory.chromadb_integrity as ci
    importlib.reload(ci)
    import app.healing.monitors.chromadb_integrity as monitor
    importlib.reload(monitor)

    monkeypatch.setattr(monitor, "send_signal_alert", lambda *a, **k: None)
    monkeypatch.setattr(monitor, "background_enabled", lambda: True)
    monkeypatch.setattr(monitor, "audit_event", lambda *a, **k: None)
    monkeypatch.setattr(ci, "_send_corruption_alert", lambda *a, **k: None)

    # First run — should mutate state.
    monitor.run()
    state_path = tmp_path / "lc_state" / "chromadb_integrity.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text())
    first_last_run = state["last_run_at"]
    assert first_last_run > 0

    # Second run within 23h — no state mutation.
    monitor.run()
    state2 = json.loads(state_path.read_text())
    assert state2["last_run_at"] == first_last_run, "monitor ran twice in 23h window"


def test_monitor_does_not_snapshot_corrupt_files(monkeypatch, tmp_path, fake_workspace):
    """Snapshot branch must skip KBs that failed integrity_check.

    This is a load-bearing invariant: snapshotting a corrupt file
    would just be replicating damage. Pinned here so a future refactor
    can't accidentally swap the order.
    """
    pytest.importorskip("pydantic_settings")  # gateway-deps shim
    import app.paths as paths
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", fake_workspace)
    import app.life_companion._common as common
    monkeypatch.setattr(common, "_STATE_DIR", tmp_path / "lc_state")

    import app.memory.chromadb_integrity as ci
    importlib.reload(ci)
    import app.healing.monitors.chromadb_integrity as monitor
    importlib.reload(monitor)

    monkeypatch.setattr(monitor, "send_signal_alert", lambda *a, **k: None)
    monkeypatch.setattr(monitor, "background_enabled", lambda: True)
    monkeypatch.setattr(monitor, "audit_event", lambda *a, **k: None)
    monkeypatch.setattr(ci, "_send_corruption_alert", lambda *a, **k: None)
    monkeypatch.setattr(ci, "_emit_ledger_event", lambda *a, **k: None)
    monkeypatch.setattr(ci, "_postgres_connect", lambda: None)

    # Damage the memory KB.
    _corrupt_sqlite(fake_workspace / "memory" / "chroma.sqlite3")
    monitor.run()
    state = json.loads((tmp_path / "lc_state" / "chromadb_integrity.json").read_text())
    summary = state["last_summary"]
    # The memory row should NOT contain a snapshot stanza.
    memory_row = next(r for r in summary["per_kb"] if r["name"] == "memory")
    assert "snapshot" not in memory_row, (
        "snapshot was taken on a damaged KB — invariant violated"
    )
    # The episteme row (healthy) should have a snapshot stanza.
    epi_row = next(r for r in summary["per_kb"] if r["name"] == "episteme")
    assert epi_row.get("snapshot", {}).get("ok")


# ────────────────────────────────────────────────────────────────────
#   Runtime-settings gates default ON (pins the failure-open posture)
# ────────────────────────────────────────────────────────────────────


def test_runtime_settings_chromadb_keys_default_on():
    """Pins the failure-OPEN posture: all four switches default ON.

    If a future refactor accidentally flips a default to OFF, the
    silent loss of integrity protection would be invisible. This test
    is the failsafe.
    """
    pytest.importorskip("pydantic_settings")  # gateway-deps shim
    import app.runtime_settings as rs
    # _defaults reads via get_settings; that may have an empty
    # Settings shim in test env. Call _defaults directly so we test
    # what the seed values are, regardless of env.
    defaults = rs._defaults()
    assert defaults["chromadb_wal_enforcement_enabled"] is True
    assert defaults["chromadb_boot_integrity_check_enabled"] is True
    assert defaults["chromadb_integrity_monitor_enabled"] is True
    assert defaults["chromadb_daily_snapshot_enabled"] is True
    assert defaults["chromadb_auto_replay_enabled"] is True


# ────────────────────────────────────────────────────────────────────
#   Pinning test: docker-compose no longer ships chromadb container
# ────────────────────────────────────────────────────────────────────


def test_docker_compose_has_no_chromadb_service():
    """The orphaned chromadb container was the root cause of the
    2026-04-25 + 2026-05-17 dual-writer corruption. Removing it from
    docker-compose.yml is the surgical fix; this test pins that
    removal so a future refactor can't accidentally restore it.

    If you genuinely need an HTTP chromadb later, restore the service
    with a NAMED volume (never bind-mount workspace/memory/ again)
    and update this test to assert the bind-mount is absent.
    """
    compose_path = Path("/Users/andrus/BotArmy/crewai-team/docker-compose.yml")
    if not compose_path.exists():
        compose_path = Path("docker-compose.yml")
    if not compose_path.exists():
        pytest.skip("docker-compose.yml not on canonical paths")
    text = compose_path.read_text()
    # The service name must not appear as a top-level service block.
    # We look for the precise indentation that defines a top-level
    # service ("  chromadb:" with exactly 2 leading spaces).
    assert "\n  chromadb:\n" not in text, (
        "chromadb service has been restored to docker-compose.yml — "
        "this re-introduces the dual-writer SQLite corruption bug "
        "that PROGRAM §55 fixed. If you need it, use a named volume."
    )
    # The bind-mount that paired the two writers must also be gone.
    assert "./workspace/memory:/chroma/chroma" not in text, (
        "the bind mount that caused the dual-writer corruption is "
        "back in docker-compose.yml — see PROGRAM §55 for context."
    )
