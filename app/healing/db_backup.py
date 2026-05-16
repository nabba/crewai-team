"""Database backup engine — Postgres + Neo4j + ChromaDB.

Wave 0/1 closure (#A1, 2026-05-09). The gateway tars the ChromaDB
volume directly (mounted at ``/app/workspace/memory``). It can also
call ``exec`` inside the Postgres and Neo4j containers via the
``docker-proxy`` sidecar — BUT only when the proxy is configured
with ``EXEC: 1``. The default compose config (``CONTAINERS: 1`` +
``POST: 1``) is NOT sufficient: the tecnativa proxy gates
``/exec/.../start`` with a separate ``EXEC`` ACL flag, denied by
default for security. Without ``EXEC: 1`` the pg/neo4j paths
return 403 from the proxy.

The 2026-05-16 split (Option C, docs/RESILIENCE_POSTURE.md):

  * Gateway keeps doing ChromaDB (no exec needed — volume is shared).
  * Postgres + Neo4j move to a host launchd LaunchAgent running
    ``deploy/scripts/backup.sh`` (which uses the host's docker
    socket directly and doesn't go through the proxy).

When ``DB_BACKUP_HOST_MANAGED=1`` (default in compose), the pg/neo4j
steps are skipped at gateway side and the entry records
``skipped: True, skipped_reason: "host_managed"`` for them. The
freshness monitor checks for a recent non-skipped success per
component — if the host LaunchAgent isn't actually running the
operator still gets a staleness alert.

Output layout::

    workspace/backups/
      manifest.json                       # see _update_manifest
      postgres/postgres-YYYYMMDDTHHMMSSZ.sql.gz     # host
      neo4j/neo4j-YYYYMMDDTHHMMSSZ.dump             # host
      chromadb/chromadb-YYYYMMDDTHHMMSSZ.tar.gz     # gateway

Retention: ``DB_BACKUP_RETENTION_DAYS`` (default 30). Manifest never
deletes its history; only the actual archive files are purged.

The engine never raises — every step is guarded. A failed step
records its error in the manifest and a partial run still yields
manifest entries for the steps that succeeded. The freshness monitor
walks the manifest and alerts.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_BACKUP_ROOT = Path("/app/workspace/backups")
_MANIFEST_PATH = _BACKUP_ROOT / "manifest.json"
_CHROMA_DATA_DIR = Path("/app/workspace/memory")  # contains chroma sub-dir
_DEFAULT_RETENTION_DAYS = 30


def _retention_days() -> int:
    raw = os.getenv("DB_BACKUP_RETENTION_DAYS", str(_DEFAULT_RETENTION_DAYS)).strip()
    try:
        return max(7, int(raw))
    except ValueError:
        return _DEFAULT_RETENTION_DAYS


def _host_managed() -> bool:
    """True when pg + neo4j backups are owned by the host launchd LaunchAgent.

    Gateway then only runs the chromadb step (which doesn't need
    docker-exec — the volume is bind-mounted into the gateway).
    """
    raw = os.getenv("DB_BACKUP_HOST_MANAGED", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _skipped_entry(reason: str) -> dict[str, Any]:
    """Manifest payload for a step the gateway deliberately didn't run."""
    return {
        "ok": True,
        "skipped": True,
        "skipped_reason": reason,
        "path": None,
        "bytes": 0,
        "duration_s": 0.0,
        "error": None,
    }


def _ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _container_by_name_or_label(client: Any, *candidates: str):
    """Find a sibling container by exact name or compose service label."""
    try:
        for name in candidates:
            try:
                return client.containers.get(name)
            except Exception:
                continue
        # Fall back to label match — works regardless of compose project name.
        all_containers = client.containers.list(all=True)
        for c in all_containers:
            labels = c.labels or {}
            svc = labels.get("com.docker.compose.service")
            if svc in candidates:
                return c
    except Exception:
        logger.debug("db_backup: container lookup failed", exc_info=True)
    return None


def _backup_postgres(now_ts: str) -> dict:
    """Run ``pg_dump`` inside the Postgres container; gzip the output."""
    if _host_managed():
        return _skipped_entry("host_managed")
    out: dict[str, Any] = {
        "ok": False, "path": None, "bytes": 0, "duration_s": 0, "error": None,
    }
    started = time.monotonic()
    try:
        import docker  # type: ignore[import-not-found]
        client = docker.from_env(timeout=300)
    except Exception as exc:
        out["error"] = f"docker SDK unavailable: {type(exc).__name__}: {exc}"
        return out

    container = _container_by_name_or_label(
        client, "postgres", "mem0-postgres", "crewai-team-postgres-1",
    )
    if container is None:
        out["error"] = "postgres container not found"
        return out

    db_name = os.getenv("MEM0_PG_DB", "mem0")
    db_user = os.getenv("MEM0_PG_USER", "mem0")
    pg_password = os.getenv("MEM0_PG_PASSWORD") or os.getenv("POSTGRES_PASSWORD") or ""

    cmd = [
        "pg_dump",
        "--username", db_user,
        "--dbname", db_name,
        "--clean", "--if-exists",      # restore-friendly
        "--no-owner", "--no-privileges",
    ]
    env = {"PGPASSWORD": pg_password} if pg_password else {}

    target_dir = _BACKUP_ROOT / "postgres"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"postgres-{now_ts}.sql.gz"

    try:
        # exec_run with stream=True returns chunks we can pipe straight
        # into gzip. Buffering the entire dump in memory would be unsafe.
        exec_id = client.api.exec_create(
            container.id, cmd, environment=env, stdout=True, stderr=False,
        )
        stream = client.api.exec_start(exec_id, stream=True, demux=False)
        with gzip.open(target, "wb") as gz:
            for chunk in stream:
                if chunk:
                    gz.write(chunk)
        info = client.api.exec_inspect(exec_id)
        rc = int(info.get("ExitCode") or 0)
        if rc != 0:
            target.unlink(missing_ok=True)
            out["error"] = f"pg_dump exit={rc}"
            return out
        size = target.stat().st_size
        out["ok"] = True
        out["path"] = str(target.relative_to(_BACKUP_ROOT.parent))
        out["bytes"] = size
    except Exception as exc:
        target.unlink(missing_ok=True)
        out["error"] = f"{type(exc).__name__}: {exc}"
        logger.debug("db_backup.postgres: dump failed", exc_info=True)
    finally:
        out["duration_s"] = round(time.monotonic() - started, 2)
    return out


def _backup_neo4j(now_ts: str) -> dict:
    """Run ``neo4j-admin database dump`` inside the Neo4j container.

    The dump command requires the database to be stopped — but the
    Community edition's ``neo4j-admin database dump --to-stdout`` works
    online. We stream the dump to a local file (no compression — the
    dump itself is already block-deduplicated).
    """
    if _host_managed():
        return _skipped_entry("host_managed")
    out: dict[str, Any] = {
        "ok": False, "path": None, "bytes": 0, "duration_s": 0, "error": None,
    }
    started = time.monotonic()
    try:
        import docker  # type: ignore[import-not-found]
        client = docker.from_env(timeout=600)
    except Exception as exc:
        out["error"] = f"docker SDK unavailable: {type(exc).__name__}: {exc}"
        return out

    container = _container_by_name_or_label(
        client, "neo4j", "mem0-neo4j", "crewai-team-neo4j-1",
    )
    if container is None:
        out["error"] = "neo4j container not found"
        return out

    target_dir = _BACKUP_ROOT / "neo4j"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"neo4j-{now_ts}.dump"

    # Use --to-stdout to stream without needing a writable volume in
    # the neo4j container. ``neo4j`` is the default DB on community.
    cmd = ["neo4j-admin", "database", "dump", "neo4j", "--to-stdout"]

    try:
        exec_id = client.api.exec_create(
            container.id, cmd, stdout=True, stderr=False, user="neo4j",
        )
        stream = client.api.exec_start(exec_id, stream=True, demux=False)
        with target.open("wb") as f:
            for chunk in stream:
                if chunk:
                    f.write(chunk)
        info = client.api.exec_inspect(exec_id)
        rc = int(info.get("ExitCode") or 0)
        if rc != 0:
            target.unlink(missing_ok=True)
            out["error"] = f"neo4j-admin exit={rc}"
            return out
        size = target.stat().st_size
        out["ok"] = True
        out["path"] = str(target.relative_to(_BACKUP_ROOT.parent))
        out["bytes"] = size
    except Exception as exc:
        target.unlink(missing_ok=True)
        out["error"] = f"{type(exc).__name__}: {exc}"
        logger.debug("db_backup.neo4j: dump failed", exc_info=True)
    finally:
        out["duration_s"] = round(time.monotonic() - started, 2)
    return out


def _backup_chromadb(now_ts: str) -> dict:
    """Tar the ChromaDB persistence directory.

    The gateway shares ``/app/workspace/memory`` with the chromadb
    container. We tar the whole memory subdir (contains chroma + a
    couple of small SQLite files); tarball is gzip-compressed.
    """
    out: dict[str, Any] = {
        "ok": False, "path": None, "bytes": 0, "duration_s": 0, "error": None,
    }
    started = time.monotonic()
    if not _CHROMA_DATA_DIR.exists():
        out["error"] = f"chroma dir not found: {_CHROMA_DATA_DIR}"
        return out

    target_dir = _BACKUP_ROOT / "chromadb"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"chromadb-{now_ts}.tar.gz"

    try:
        with tarfile.open(target, "w:gz") as tar:
            # Filter: skip transient lock + journal files. They're
            # harmless on restore (re-created), and including them
            # makes the tarball racy when chroma is actively writing.
            def _filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
                name = Path(tarinfo.name).name
                if name.endswith(".lock") or name.endswith("-journal"):
                    return None
                return tarinfo
            tar.add(
                _CHROMA_DATA_DIR, arcname="memory", filter=_filter,
            )
        size = target.stat().st_size
        out["ok"] = True
        out["path"] = str(target.relative_to(_BACKUP_ROOT.parent))
        out["bytes"] = size
    except Exception as exc:
        target.unlink(missing_ok=True)
        out["error"] = f"{type(exc).__name__}: {exc}"
        logger.debug("db_backup.chromadb: tar failed", exc_info=True)
    finally:
        out["duration_s"] = round(time.monotonic() - started, 2)
    return out


def _purge_old_archives(retention_days: int) -> dict:
    """Delete archive files older than retention_days. Manifest preserved."""
    summary = {"deleted": 0, "bytes_freed": 0, "errors": 0}
    cutoff = time.time() - retention_days * 24 * 3600
    for sub in ("postgres", "neo4j", "chromadb"):
        d = _BACKUP_ROOT / sub
        if not d.exists():
            continue
        try:
            for f in d.iterdir():
                if not f.is_file():
                    continue
                try:
                    if f.stat().st_mtime > cutoff:
                        continue
                    size = f.stat().st_size
                    f.unlink()
                    summary["deleted"] += 1
                    summary["bytes_freed"] += size
                except OSError:
                    summary["errors"] += 1
        except OSError:
            summary["errors"] += 1
    return summary


def _read_manifest() -> dict:
    if not _MANIFEST_PATH.exists():
        return {"runs": []}
    try:
        return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("db_backup: manifest unreadable; starting fresh", exc_info=True)
        return {"runs": []}


def _update_manifest(entry: dict) -> None:
    _BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest()
    runs = manifest.setdefault("runs", [])
    runs.append(entry)
    # Keep the last 200 runs to bound size; older ones drop off the
    # head. The actual archive files have their own retention policy.
    if len(runs) > 200:
        manifest["runs"] = runs[-200:]
    manifest["last_updated_at"] = entry["completed_at"]
    tmp = _MANIFEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(_MANIFEST_PATH)


def run_backup() -> dict:
    """Run all 3 backups + retention purge. Returns the manifest entry."""
    started = datetime.now(timezone.utc)
    now_ts = started.strftime("%Y%m%dT%H%M%SZ")

    pg = _backup_postgres(now_ts)
    n4j = _backup_neo4j(now_ts)
    chr_ = _backup_chromadb(now_ts)
    purge = _purge_old_archives(_retention_days())

    completed = datetime.now(timezone.utc)
    entry = {
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "duration_s": round((completed - started).total_seconds(), 2),
        "source": "gateway",
        "host_managed": _host_managed(),
        "postgres": pg,
        "neo4j": n4j,
        "chromadb": chr_,
        "purge": purge,
        "all_ok": bool(pg["ok"] and n4j["ok"] and chr_["ok"]),
    }
    try:
        _update_manifest(entry)
    except Exception:
        logger.debug("db_backup: manifest write failed", exc_info=True)

    # Audit (best-effort).
    try:
        from app.life_companion._common import audit_event
        audit_event(
            "db_backup_run",
            all_ok=entry["all_ok"],
            postgres_ok=pg["ok"],
            neo4j_ok=n4j["ok"],
            chromadb_ok=chr_["ok"],
            postgres_bytes=pg["bytes"],
            neo4j_bytes=n4j["bytes"],
            chromadb_bytes=chr_["bytes"],
            duration_s=entry["duration_s"],
            purged=purge["deleted"],
        )
    except Exception:
        logger.debug("db_backup: audit failed", exc_info=True)
    return entry
