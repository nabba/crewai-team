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
    # Inherit-from-shell could leak host-managed mode into tests that
    # need to exercise the legacy in-gateway pg/neo4j paths.
    monkeypatch.delenv("DB_BACKUP_HOST_MANAGED", raising=False)
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


# ── Host-managed split tests (2026-05-16) ───────────────────────────────


def test_host_managed_skips_postgres(isolated, monkeypatch):
    """Under DB_BACKUP_HOST_MANAGED=1, pg step short-circuits BEFORE
    touching the docker SDK — proves the gateway no longer hits the
    proxy's denied /exec endpoint."""
    from app.healing import db_backup

    monkeypatch.setenv("DB_BACKUP_HOST_MANAGED", "1")
    # If the code DID try to call docker, this would surface the import
    # failure as an error — but with host_managed it shouldn't even try.
    import sys
    monkeypatch.setitem(sys.modules, "docker", None)

    result = db_backup._backup_postgres("20260516T120000Z")
    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["skipped_reason"] == "host_managed"
    assert result["error"] is None
    assert result["path"] is None


def test_host_managed_skips_neo4j(isolated, monkeypatch):
    from app.healing import db_backup

    monkeypatch.setenv("DB_BACKUP_HOST_MANAGED", "1")
    import sys
    monkeypatch.setitem(sys.modules, "docker", None)

    result = db_backup._backup_neo4j("20260516T120000Z")
    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["skipped_reason"] == "host_managed"


def test_host_managed_full_run_writes_skipped_to_manifest(isolated, monkeypatch):
    """Under host-managed mode: gateway run claims all_ok=True because
    chromadb actually succeeded and pg/neo4j are deferred to the host."""
    from app.healing import db_backup

    monkeypatch.setenv("DB_BACKUP_HOST_MANAGED", "1")
    chroma = isolated / "memory"
    chroma.mkdir()
    (chroma / "data.bin").write_bytes(b"payload")

    entry = db_backup.run_backup()
    assert entry["host_managed"] is True
    assert entry["postgres"]["skipped"] is True
    assert entry["neo4j"]["skipped"] is True
    assert entry["chromadb"]["ok"] is True
    assert entry["chromadb"].get("skipped") is not True
    assert entry["all_ok"] is True


def test_host_managed_does_not_mask_chromadb_failure(isolated, monkeypatch):
    """If chromadb itself fails (e.g. dir missing), all_ok=False even
    under host-managed — the failure surface for chromadb is still
    visible."""
    from app.healing import db_backup

    monkeypatch.setenv("DB_BACKUP_HOST_MANAGED", "1")
    # No chroma dir created → _backup_chromadb returns ok=False.
    entry = db_backup.run_backup()
    assert entry["postgres"]["skipped"] is True
    assert entry["neo4j"]["skipped"] is True
    assert entry["chromadb"]["ok"] is False
    assert entry["all_ok"] is False


def test_monitor_alerts_when_postgres_stale_even_if_chromadb_fresh(monitor_isolated, monkeypatch):
    """The post-split fix: a happy gateway-only chromadb stream MUST
    NOT mask a dead host LaunchAgent (pg/neo4j never backed up)."""
    tmp_path, sent = monitor_isolated
    from app.healing.monitors import db_backup as monitor_mod
    from datetime import datetime, timezone

    manifest_path = tmp_path / "backups" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).timestamp()
    old_ts = datetime.fromtimestamp(now - 30 * 86400, tz=timezone.utc).isoformat()
    recent_ts = datetime.fromtimestamp(now - 1 * 86400, tz=timezone.utc).isoformat()
    manifest_path.write_text(json.dumps({
        "runs": [
            # Real pg backup, but 30 days ago (stale).
            {"completed_at": old_ts, "all_ok": True,
             "postgres": {"ok": True}, "neo4j": {"ok": True},
             "chromadb": {"ok": True}},
            # Gateway-only chromadb success — yesterday — but pg/n4j
            # placeholders are skipped, so they don't update freshness.
            {"completed_at": recent_ts, "all_ok": True,
             "host_managed": True,
             "postgres": {"ok": True, "skipped": True},
             "neo4j": {"ok": True, "skipped": True},
             "chromadb": {"ok": True}},
        ],
    }))
    monkeypatch.setattr(monitor_mod, "_run_interval_s", lambda: 999_999_999)

    monitor_mod.run()
    assert any("postgres=30d" in s and "neo4j=30d" in s for s in sent)
    # And the hint should point at the host LaunchAgent, not at the
    # gateway logs (since only pg+n4j are stale).
    assert any("LaunchAgent" in s or "deploy/scripts/backup.sh" in s
               for s in sent)


def test_monitor_alerts_for_chromadb_stale_separately(monitor_isolated, monkeypatch):
    """Stale chromadb → gateway-owned hint, not the host LaunchAgent."""
    tmp_path, sent = monitor_isolated
    from app.healing.monitors import db_backup as monitor_mod
    from datetime import datetime, timezone

    manifest_path = tmp_path / "backups" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).timestamp()
    old_ts = datetime.fromtimestamp(now - 30 * 86400, tz=timezone.utc).isoformat()
    recent_ts = datetime.fromtimestamp(now - 1 * 86400, tz=timezone.utc).isoformat()
    manifest_path.write_text(json.dumps({
        "runs": [
            # Old chromadb success.
            {"completed_at": old_ts, "all_ok": True,
             "postgres": {"ok": True, "skipped": True},
             "neo4j": {"ok": True, "skipped": True},
             "chromadb": {"ok": True}},
            # Recent pg + neo4j from host, chromadb skipped (gateway
            # is supposed to do it but apparently hasn't).
            {"completed_at": recent_ts, "all_ok": True,
             "source": "operator_script",
             "postgres": {"ok": True}, "neo4j": {"ok": True},
             "chromadb": {"ok": True, "skipped": True}},
        ],
    }))
    monkeypatch.setattr(monitor_mod, "_run_interval_s", lambda: 999_999_999)

    monitor_mod.run()
    assert any("chromadb=30d" in s for s in sent)
    assert any("gateway" in s.lower() for s in sent)


def test_monitor_skipped_only_manifest_counts_as_never(monitor_isolated, monkeypatch):
    """A manifest with only skipped entries for a component is treated
    as 'never backed up' — gateway can't pretend pg is fresh just
    because it wrote a skipped placeholder."""
    tmp_path, sent = monitor_isolated
    from app.healing.monitors import db_backup as monitor_mod
    from datetime import datetime, timezone

    manifest_path = tmp_path / "backups" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    recent_ts = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps({
        "runs": [{
            "completed_at": recent_ts, "all_ok": True,
            "host_managed": True,
            "postgres": {"ok": True, "skipped": True},
            "neo4j": {"ok": True, "skipped": True},
            "chromadb": {"ok": True},
        }],
    }))
    monkeypatch.setattr(monitor_mod, "_run_interval_s", lambda: 999_999_999)

    monitor_mod.run()
    assert any("postgres=never" in s and "neo4j=never" in s for s in sent)
