"""Tests for ``app.healing.db_backup`` engine + monitor (Wave 0/1 #A1).

The Postgres + Neo4j paths require a running Docker SDK to exec
inside sibling containers — we don't test those here. The ChromaDB
tar path is pure-Python and IS exercised; manifest format + freshness
detection in the monitor are also exercised.
"""
from __future__ import annotations

import json
import tarfile
import time
from datetime import datetime, timezone

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.healing import db_backup
    monkeypatch.setattr(db_backup, "_BACKUP_ROOT", tmp_path / "backups")
    monkeypatch.setattr(db_backup, "_MANIFEST_PATH",
                        tmp_path / "backups" / "manifest.json")
    monkeypatch.setattr(db_backup, "_CHROMA_DATA_DIR", tmp_path / "memory")
    yield tmp_path


def test_chromadb_tar_round_trip(isolated):
    from app.healing import db_backup

    chroma = isolated / "memory"
    chroma.mkdir()
    (chroma / "data.bin").write_bytes(b"chroma payload")
    (chroma / "stale.lock").write_text("ignored")
    (chroma / "stale-journal").write_text("ignored")

    result = db_backup._backup_chromadb("20260509T120000Z")
    assert result["ok"] is True
    assert result["bytes"] > 0
    archive = isolated / "backups" / "chromadb" / "chromadb-20260509T120000Z.tar.gz"
    assert archive.exists()

    # Tar contains data.bin but excludes lock + journal files.
    with tarfile.open(archive, "r:gz") as tar:
        names = {m.name for m in tar.getmembers()}
    assert any(n.endswith("memory/data.bin") for n in names)
    assert not any(n.endswith(".lock") for n in names)
    assert not any(n.endswith("-journal") for n in names)


def test_chromadb_handles_missing_dir(isolated):
    from app.healing import db_backup
    # _CHROMA_DATA_DIR is repointed but nothing was created.
    result = db_backup._backup_chromadb("20260509T120000Z")
    assert result["ok"] is False
    assert "not found" in result["error"]


def test_postgres_handles_missing_docker(isolated, monkeypatch):
    from app.healing import db_backup

    # Force the docker import inside the function to fail.
    import sys
    monkeypatch.setitem(sys.modules, "docker", None)
    result = db_backup._backup_postgres("20260509T120000Z")
    assert result["ok"] is False
    assert "docker" in result["error"].lower()


def test_neo4j_handles_missing_docker(isolated, monkeypatch):
    from app.healing import db_backup

    import sys
    monkeypatch.setitem(sys.modules, "docker", None)
    result = db_backup._backup_neo4j("20260509T120000Z")
    assert result["ok"] is False
    assert "docker" in result["error"].lower()


def test_purge_old_backups(isolated):
    from app.healing import db_backup
    import os

    archive_dir = isolated / "backups" / "postgres"
    archive_dir.mkdir(parents=True)
    fresh = archive_dir / "fresh.sql.gz"
    stale = archive_dir / "stale.sql.gz"
    fresh.write_text("fresh")
    stale.write_text("stale")
    old = time.time() - 200 * 86400
    os.utime(stale, (old, old))

    summary = db_backup._purge_old_archives(retention_days=30)
    assert summary["deleted"] == 1
    assert fresh.exists()
    assert not stale.exists()


def test_run_backup_writes_manifest(isolated):
    """Even though pg + neo4j fail (no docker), chroma works and manifest is written."""
    from app.healing import db_backup

    chroma = isolated / "memory"
    chroma.mkdir()
    (chroma / "data.bin").write_bytes(b"x")

    entry = db_backup.run_backup()
    assert "started_at" in entry
    assert "completed_at" in entry
    # ChromaDB succeeded; postgres + neo4j failed with no-container errors.
    assert entry["chromadb"]["ok"] is True

    # Manifest file exists and contains this run.
    manifest = json.loads((isolated / "backups/manifest.json").read_text())
    assert len(manifest["runs"]) == 1
    assert manifest["runs"][0]["chromadb"]["ok"] is True


def test_manifest_history_capped_at_200(isolated):
    """Old manifest with >200 runs gets trimmed when a new one appends."""
    from app.healing import db_backup

    chroma = isolated / "memory"
    chroma.mkdir()
    (chroma / "data.bin").write_bytes(b"x")

    # Pre-seed the manifest with 250 runs.
    db_backup._BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = db_backup._MANIFEST_PATH
    runs = [
        {"completed_at": f"2026-05-{i:02d}T00:00:00+00:00",
         "all_ok": True, "postgres": {"ok": True}, "neo4j": {"ok": True},
         "chromadb": {"ok": True}}
        for i in range(250)
    ]
    manifest_path.write_text(json.dumps({"runs": runs}))

    db_backup.run_backup()
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["runs"]) == 200


# ── Monitor tests ───────────────────────────────────────────────────────


@pytest.fixture
def monitor_isolated(tmp_path, monkeypatch):
    from app.healing.monitors import db_backup as monitor_mod
    from app.healing.handlers import _common as _h_common

    # db_backup reads/writes state via app.healing.handlers._common
    monkeypatch.setattr(_h_common, "_STATE_DIR", tmp_path / "self_heal")
    monkeypatch.setenv("HEALING_DB_BACKUP_ENABLED", "1")

    sent: list[str] = []
    monkeypatch.setattr(monitor_mod, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(monitor_mod, "audit_event", lambda *a, **k: None)
    monkeypatch.setattr(monitor_mod, "_MANIFEST_PATH",
                        tmp_path / "backups" / "manifest.json")

    yield tmp_path, sent


def test_monitor_alerts_on_no_manifest(monitor_isolated, monkeypatch):
    tmp_path, sent = monitor_isolated
    from app.healing.monitors import db_backup as monitor_mod

    # Don't run an actual backup — stub it to no-op so we exercise only
    # the staleness alert path.
    monkeypatch.setattr(monitor_mod, "_run_interval_s", lambda: 999_999_999)

    monitor_mod.run()
    assert any("never" in s.lower() for s in sent)


def test_monitor_alerts_on_stale_manifest(monitor_isolated, monkeypatch):
    tmp_path, sent = monitor_isolated
    from app.healing.monitors import db_backup as monitor_mod

    manifest_path = tmp_path / "backups" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # 30 days ago — beyond default 14-day threshold.
    old_iso = (datetime.now(timezone.utc).timestamp() - 30 * 86400)
    from datetime import datetime as _dt
    old_dt = _dt.fromtimestamp(old_iso, tz=timezone.utc)
    manifest_path.write_text(json.dumps({
        "runs": [{
            "completed_at": old_dt.isoformat(),
            "all_ok": True,
            "postgres": {"ok": True},
            "neo4j": {"ok": True},
            "chromadb": {"ok": True},
        }],
    }))
    # Suppress the actual backup.
    monkeypatch.setattr(monitor_mod, "_run_interval_s", lambda: 999_999_999)

    monitor_mod.run()
    assert any("old" in s.lower() and "backup" in s.lower() for s in sent)


def test_monitor_disabled_is_noop(monitor_isolated, monkeypatch):
    tmp_path, sent = monitor_isolated
    from app.healing.monitors import db_backup as monitor_mod
    monkeypatch.setenv("HEALING_DB_BACKUP_ENABLED", "0")
    monitor_mod.run()
    assert sent == []
