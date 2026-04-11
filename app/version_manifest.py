"""
version_manifest.py — Composite versioning across all system state layers.

Treats the entire agent system state as a single versionable artifact:
  code (git SHA) + prompts (registry versions) + mem0 (PostgreSQL snapshot)
  + chromadb (collection hashes) + neo4j (metadata) + config (hash)

Every promotion creates a tagged manifest. Rollback coverage varies by layer:

  RESTORABLE (actual rollback):
    - Prompt versions: active.txt pointers restored instantly
    - Mem0/PostgreSQL: restored from pg_dump snapshot (if available)
    - Config: runtime settings restored from manifest snapshot

  VERIFIABLE (integrity check only):
    - Soul files: SHA-256 compared, violation raised if tampered
    - ChromaDB: collection counts compared, drift flagged

  RECORDED (audit trail only, requires manual intervention):
    - Git/code: SHA recorded; code rollback requires redeployment
    - Neo4j: node/relationship counts recorded; no automated restore

IMMUTABLE — infrastructure-level module.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFESTS_DIR = Path("/app/workspace/manifests")
SNAPSHOTS_DIR = Path("/app/workspace/snapshots")
WORKSPACE = Path("/app/workspace")

# ── Manifest schema ──────────────────────────────────────────────────────────


def create_manifest(
    promoted_by: str = "system",
    reason: str = "",
    evaluation_run_id: str = "",
) -> dict:
    """Capture current state of all system layers into a version manifest.

    Snapshots vary by layer (see module docstring for restore coverage).
    Returns the manifest dict and writes it to workspace/manifests/.
    """
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc)
    ts_str = ts.strftime("%Y%m%dT%H%M%SZ")

    manifest = {
        "version": _next_version(),
        "timestamp": ts.isoformat(),
        "components": {
            "agent_code": _snapshot_code(),
            "prompts": _snapshot_prompts(),
            "soul_md": _hash_soul_files(),
            "mem0_state": _snapshot_mem0(ts_str),
            "chromadb_knowledge": _hash_chromadb(),
            "neo4j_graph": _snapshot_neo4j(ts_str),
            "config": _hash_config(),
        },
        "_config_snapshot": _snapshot_config_values(),
        "promoted_by": promoted_by,
        "evaluation_run_id": evaluation_run_id,
        "reason": reason,
        "parent_version": _current_version(),
    }

    # Write manifest
    manifest_path = MANIFESTS_DIR / f"{manifest['version']}.json"
    from app.safe_io import safe_write_json, safe_write
    safe_write_json(manifest_path, manifest)

    # Update current pointer
    safe_write(MANIFESTS_DIR / "current.txt", manifest["version"])

    logger.info(f"version_manifest: created {manifest['version']} ({reason})")
    return manifest


def get_current_manifest() -> dict | None:
    """Load the currently active manifest."""
    pointer = MANIFESTS_DIR / "current.txt"
    if not pointer.exists():
        return None
    version = pointer.read_text().strip()
    manifest_path = MANIFESTS_DIR / f"{version}.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text())


def list_manifests() -> list[str]:
    """Return all manifest versions, newest first."""
    if not MANIFESTS_DIR.exists():
        return []
    versions = []
    for f in sorted(MANIFESTS_DIR.glob("v*.json"), reverse=True):
        versions.append(f.stem)
    return versions


def restore_from_manifest(version: str) -> dict:
    """Restore restorable layers and verify integrity of others.

    Restorable: prompts, mem0/PostgreSQL, config.
    Verified: soul files (raises on mismatch), ChromaDB (warns on drift).
    Recorded only: git/code, neo4j (logged, no automated restore).

    Returns: {restored: bool, duration_seconds: float, errors: list, warnings: list}
    """
    start = time.monotonic()
    errors = []
    warnings = []

    manifest_path = MANIFESTS_DIR / f"{version}.json"
    if not manifest_path.exists():
        return {"restored": False, "duration_seconds": 0,
                "errors": [f"Manifest {version} not found"], "warnings": []}

    manifest = json.loads(manifest_path.read_text())
    components = manifest.get("components", {})

    # 1. Restore prompts (instant — just update active.txt pointers)
    try:
        _restore_prompts(components.get("prompts", {}))
    except Exception as e:
        errors.append(f"prompts: {e}")

    # 2. Verify soul.md integrity (raises on tampering)
    try:
        _verify_soul_integrity(components.get("soul_md", {}))
    except Exception as e:
        errors.append(f"soul_md: {e}")

    # 3. Restore mem0 PostgreSQL
    try:
        _restore_mem0(components.get("mem0_state", {}))
    except Exception as e:
        errors.append(f"mem0: {e}")

    # 4. Restore config settings
    try:
        _restore_config(components.get("config", {}), manifest)
    except Exception as e:
        warnings.append(f"config: {e}")

    # 5. Check ChromaDB drift (warn only — embeddings not restorable)
    try:
        drift = _check_chromadb_drift(components.get("chromadb_knowledge", {}))
        if drift:
            warnings.append(f"chromadb: {drift}")
    except Exception as e:
        warnings.append(f"chromadb check: {e}")

    # 6. Log neo4j state (audit only — no automated restore)
    try:
        _restore_neo4j(components.get("neo4j_graph", {}))
    except Exception as e:
        warnings.append(f"neo4j: {e}")

    # 7. Flag code drift (audit only — code requires redeployment)
    code_state = components.get("agent_code", {})
    current_code = _snapshot_code()
    if code_state.get("git_sha") and current_code.get("git_sha") != code_state.get("git_sha"):
        warnings.append(
            f"code: current git SHA {current_code.get('git_sha', '?')[:8]} differs from "
            f"manifest {code_state.get('git_sha', '?')[:8]} — manual redeployment needed"
        )

    # 8. Update current pointer
    from app.safe_io import safe_write
    safe_write(MANIFESTS_DIR / "current.txt", version)

    duration = time.monotonic() - start
    logger.info(f"version_manifest: restored to {version} in {duration:.1f}s "
                f"({len(errors)} errors, {len(warnings)} warnings)")

    return {
        "restored": len(errors) == 0,
        "duration_seconds": duration,
        "errors": errors,
        "warnings": warnings,
    }


def rollback_to_previous() -> dict:
    """Rollback to the parent of the current manifest."""
    current = get_current_manifest()
    if not current:
        return {"restored": False, "errors": ["No current manifest"]}
    parent = current.get("parent_version", "")
    if not parent:
        return {"restored": False, "errors": ["No parent version to rollback to"]}
    return restore_from_manifest(parent)


def tag_git_version(version: str, message: str = "") -> bool:
    """Create a git tag for this version (if in a git repo)."""
    try:
        result = subprocess.run(
            ["git", "tag", "-a", version, "-m", message or f"Auto-tagged {version}"],
            capture_output=True, timeout=10,
            cwd="/app",
        )
        return result.returncode == 0
    except Exception:
        return False


# ── Snapshot helpers ─────────────────────────────────────────────────────────


def _next_version() -> str:
    """Generate next semantic version from manifest history."""
    versions = list_manifests()
    if not versions:
        return "v0.1.0"

    latest = versions[0]
    # Parse v{major}.{minor}.{patch}
    try:
        parts = latest.lstrip("v").split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        return f"v{major}.{minor}.{patch + 1}"
    except (IndexError, ValueError):
        return f"v0.1.{len(versions)}"


def _current_version() -> str:
    """Return current active version string."""
    pointer = MANIFESTS_DIR / "current.txt"
    if pointer.exists():
        return pointer.read_text().strip()
    return ""


def _snapshot_code() -> dict:
    """Capture git state. Tries /app first, then /app/workspace (mounted volume)."""
    for cwd in ["/app", "/app/workspace"]:
        try:
            sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5, cwd=cwd,
            ).stdout.strip()
            if sha:
                branch = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True, text=True, timeout=5, cwd=cwd,
                ).stdout.strip()
                dirty = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True, text=True, timeout=5, cwd=cwd,
                ).stdout.strip()
                return {"git_sha": sha, "branch": branch, "dirty": bool(dirty)}
        except Exception:
            continue

    # Fallback: read from BUILD_SHA env var or Dockerfile label
    build_sha = os.environ.get("BUILD_SHA", "")
    if build_sha:
        return {"git_sha": build_sha, "branch": "main", "dirty": False}

    return {"git_sha": "unknown", "branch": "unknown", "dirty": True}


def _snapshot_prompts() -> dict:
    """Capture all active prompt versions."""
    try:
        from app.prompt_registry import get_prompt_versions_map
        return get_prompt_versions_map()
    except Exception:
        return {}


def _hash_soul_files() -> dict:
    """SHA-256 hash of all soul/constitution files."""
    hashes = {}
    soul_dir = Path("/app/app/souls")
    if soul_dir.exists():
        for f in sorted(soul_dir.glob("*.md")):
            hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    return hashes


def _hash_config() -> dict:
    """Hash of current configuration (non-secret fields)."""
    try:
        from app.config import get_settings
        s = get_settings()
        # Hash non-secret config values
        config_str = json.dumps({
            "cost_mode": s.cost_mode,
            "llm_mode": s.llm_mode,
            "evolution_iterations": s.evolution_iterations,
            "local_model_default": s.local_model_default,
            "vetting_enabled": s.vetting_enabled,
            "feedback_enabled": s.feedback_enabled,
            "modification_enabled": s.modification_enabled,
        }, sort_keys=True)
        return {"config_hash": hashlib.sha256(config_str.encode()).hexdigest()}
    except Exception:
        return {"config_hash": "unknown"}


def _snapshot_config_values() -> dict:
    """Capture restorable config values (non-secret behavioral settings)."""
    try:
        from app.config import get_settings
        s = get_settings()
        return {
            "cost_mode": s.cost_mode,
            "llm_mode": s.llm_mode,
            "vetting_enabled": s.vetting_enabled,
            "feedback_enabled": s.feedback_enabled,
            "modification_enabled": s.modification_enabled,
            "evolution_iterations": s.evolution_iterations,
        }
    except Exception:
        return {}


def _hash_chromadb() -> dict:
    """Hash of ChromaDB collection metadata (not full embeddings).

    Uses PersistentClient (same as rest of system) — avoids the
    HttpClient API version mismatch with chromadb server 0.5.23.
    """
    try:
        from app.memory.chromadb_manager import get_client
        client = get_client()
        collections = client.list_collections()
        hashes = {}
        for col in collections:
            count = col.count()
            hashes[col.name] = {
                "count": count,
                "hash": hashlib.sha256(f"{col.name}:{count}".encode()).hexdigest()[:16],
            }
        return {"collections": hashes, "total_collections": len(hashes)}
    except Exception:
        return {"collections": {}}


def _snapshot_mem0(ts_str: str) -> dict:
    """Capture PostgreSQL state metadata via psycopg2 query.

    Uses direct SQL queries instead of pg_dump (which isn't in the slim image).
    Records row counts per schema/table for snapshot comparison.
    """
    try:
        from app.config import get_settings
        import psycopg2
        s = get_settings()
        pg_url = s.mem0_postgres_url
        if not pg_url:
            return {"snapshot_id": "", "error": "no postgres URL"}

        conn = psycopg2.connect(pg_url)
        conn.autocommit = True
        with conn.cursor() as cur:
            # Get row counts per table across all application schemas
            cur.execute("""
                SELECT schemaname, tablename
                FROM pg_tables
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY schemaname, tablename
            """)
            tables = cur.fetchall()

            table_counts = {}
            total_rows = 0
            for schema, table in tables:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
                    count = cur.fetchone()[0]
                    table_counts[f"{schema}.{table}"] = count
                    total_rows += count
                except Exception:
                    table_counts[f"{schema}.{table}"] = -1

        conn.close()

        return {
            "snapshot_id": ts_str,
            "method": "row_counts",
            "total_tables": len(table_counts),
            "total_rows": total_rows,
            "tables": table_counts,
        }
    except Exception as e:
        return {"snapshot_id": ts_str, "error": str(e)[:200]}


def _snapshot_neo4j(ts_str: str) -> dict:
    """Capture Neo4j state metadata."""
    try:
        from neo4j import GraphDatabase
        from app.config import get_settings
        s = get_settings()
        pw = s.mem0_neo4j_password.get_secret_value()
        if not pw:
            return {"snapshot_id": "", "error": "no neo4j password"}

        driver = GraphDatabase.driver(s.mem0_neo4j_url, auth=("neo4j", pw))
        with driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
        driver.close()

        return {
            "snapshot_id": ts_str,
            "node_count": node_count,
            "relationship_count": rel_count,
        }
    except Exception as e:
        return {"snapshot_id": ts_str, "error": str(e)[:200]}


# ── Restore helpers ──────────────────────────────────────────────────────────


def _restore_prompts(prompt_versions: dict) -> None:
    """Restore prompt versions from manifest."""
    if not prompt_versions:
        return
    try:
        from app.prompt_registry import promote_version, get_active_version
        for role, version in prompt_versions.items():
            if isinstance(version, int):
                current = get_active_version(role)
                if current != version:
                    promote_version(role, version)
                    logger.info(f"version_manifest: restored {role} prompt to v{version:03d}")
    except Exception as e:
        raise RuntimeError(f"Failed to restore prompts: {e}")


def _verify_soul_integrity(soul_hashes: dict) -> None:
    """Verify soul files haven't been tampered with."""
    soul_dir = Path("/app/app/souls")
    for filename, expected_hash in soul_hashes.items():
        path = soul_dir / filename
        if path.exists():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected_hash:
                raise RuntimeError(
                    f"SOUL INTEGRITY VIOLATION: {filename} hash mismatch "
                    f"(expected {expected_hash[:16]}..., got {actual[:16]}...)"
                )


def _restore_mem0(mem0_state: dict) -> None:
    """Restore PostgreSQL from snapshot."""
    dump_path = mem0_state.get("dump_path", "")
    if not dump_path or not Path(dump_path).exists():
        logger.debug("version_manifest: no mem0 dump to restore")
        return

    try:
        from app.config import get_settings
        s = get_settings()
        result = subprocess.run(
            ["psql", "-h", s.mem0_postgres_host, "-p", str(s.mem0_postgres_port),
             "-U", s.mem0_postgres_user, "-d", s.mem0_postgres_db,
             "-f", dump_path],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "PGPASSWORD": s.mem0_postgres_password.get_secret_value()},
        )
        if result.returncode != 0:
            logger.warning(f"version_manifest: mem0 restore had issues: {result.stderr[:200]}")
    except Exception as e:
        raise RuntimeError(f"Failed to restore mem0: {e}")


def _restore_config(config_state: dict, manifest: dict) -> None:
    """Restore runtime config settings from manifest.

    Only restores safe, non-secret fields that affect agent behavior.
    Does NOT touch API keys, database URLs, or infrastructure settings.
    """
    config_snapshot = manifest.get("_config_snapshot")
    if not config_snapshot:
        logger.debug("version_manifest: no config snapshot in manifest — skipping config restore")
        return

    try:
        from app.config import get_settings
        s = get_settings()
        restored = []
        for key, value in config_snapshot.items():
            if hasattr(s, key) and key in _RESTORABLE_CONFIG_KEYS:
                current = getattr(s, key)
                if current != value:
                    setattr(s, key, value)
                    restored.append(f"{key}: {current} → {value}")
        if restored:
            logger.info(f"version_manifest: restored config: {', '.join(restored)}")
    except Exception as e:
        raise RuntimeError(f"Config restore failed: {e}")


# Config keys that are safe to restore automatically (non-secret, non-infrastructure)
_RESTORABLE_CONFIG_KEYS = {
    "cost_mode", "llm_mode", "vetting_enabled",
    "feedback_enabled", "modification_enabled",
    "evolution_iterations",
}


def _check_chromadb_drift(chromadb_state: dict) -> str:
    """Compare current ChromaDB state against manifest snapshot.

    Returns a drift description string, or empty string if no drift.
    ChromaDB embeddings are not restorable — this is informational only.
    """
    saved_collections = chromadb_state.get("collections", {})
    if not saved_collections:
        return ""

    try:
        current = _hash_chromadb().get("collections", {})
        drifts = []
        for name, saved in saved_collections.items():
            cur = current.get(name)
            if not cur:
                drifts.append(f"{name}: missing (was {saved.get('count', '?')} docs)")
            elif cur.get("count", 0) != saved.get("count", 0):
                drifts.append(f"{name}: {saved.get('count')} → {cur.get('count')} docs")
        if drifts:
            return f"collection drift: {'; '.join(drifts)}"
    except Exception:
        pass
    return ""


def _restore_neo4j(neo4j_state: dict) -> None:
    """Neo4j state is recorded for audit only — no automated restore.

    Logs the expected node/relationship counts for manual comparison.
    """
    if not neo4j_state or neo4j_state.get("error"):
        return
    logger.debug(f"version_manifest: neo4j state at manifest time: "
                 f"nodes={neo4j_state.get('node_count', '?')}, "
                 f"rels={neo4j_state.get('relationship_count', '?')}")


# ── Cleanup ──────────────────────────────────────────────────────────────────


def cleanup_old_snapshots(keep_latest: int = 10) -> int:
    """Remove old snapshot directories to reclaim disk space."""
    if not SNAPSHOTS_DIR.exists():
        return 0

    removed = 0
    for subdir in ("mem0", "neo4j"):
        snap_dir = SNAPSHOTS_DIR / subdir
        if not snap_dir.exists():
            continue
        dirs = sorted(snap_dir.iterdir(), reverse=True)
        for d in dirs[keep_latest:]:
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
    return removed
