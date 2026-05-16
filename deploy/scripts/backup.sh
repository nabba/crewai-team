#!/usr/bin/env bash
# AndrusAI host-side DB backup script.
# Wave 0/1 closure (#A1, 2026-05-09).
#
# Backs up Postgres + Neo4j + ChromaDB to ./workspace/backups/.
# Run from the repo root: `bash deploy/scripts/backup.sh`.
#
# This is the operator-runnable equivalent of app/healing/db_backup.py.
# Use this when the gateway is down or you'd rather not let the gateway
# do the backups itself. In production K8s, prefer a CronJob; this is
# the laptop-dev path.
#
# Env knobs (2026-05-16 split):
#   BACKUP_SKIP_POSTGRES=1   skip the pg_dump step
#   BACKUP_SKIP_NEO4J=1      skip the neo4j-admin dump step
#   BACKUP_SKIP_CHROMADB=1   skip the chromadb tar step (set by the
#                            host LaunchAgent when the gateway already
#                            owns chromadb under DB_BACKUP_HOST_MANAGED)
#
# Skipped components write {"ok": true, "skipped": true,
# "skipped_reason": "operator_skipped"} into the manifest so the
# freshness monitor's per-component check treats them as deferred
# rather than failed.
#
# Exits non-zero if any non-skipped step fails. Manifest entry is
# still written so the freshness monitor sees the partial run.

set -euo pipefail

# Resolve repo root (script is at deploy/scripts/backup.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f docker-compose.yml ]]; then
    echo "ERROR: not a crewai-team checkout (docker-compose.yml missing)" >&2
    exit 2
fi

# --- Compose detection ---------------------------------------------------
if command -v docker >/dev/null 2>&1; then
    if docker compose version >/dev/null 2>&1; then
        COMPOSE="docker compose"
    elif command -v docker-compose >/dev/null 2>&1; then
        COMPOSE="docker-compose"
    else
        echo "ERROR: docker compose not found" >&2
        exit 2
    fi
else
    echo "ERROR: docker not found" >&2
    exit 2
fi

# --- Container resolution ------------------------------------------------
PG_CONTAINER="$($COMPOSE ps -q postgres 2>/dev/null || true)"
NEO_CONTAINER="$($COMPOSE ps -q neo4j 2>/dev/null || true)"

# --- Output paths --------------------------------------------------------
TS="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_ROOT="${REPO_ROOT}/workspace/backups"
mkdir -p "${BACKUP_ROOT}/postgres" "${BACKUP_ROOT}/neo4j" "${BACKUP_ROOT}/chromadb"

PG_OUT="${BACKUP_ROOT}/postgres/postgres-${TS}.sql.gz"
NEO_OUT="${BACKUP_ROOT}/neo4j/neo4j-${TS}.dump"
CHR_OUT="${BACKUP_ROOT}/chromadb/chromadb-${TS}.tar.gz"
MANIFEST="${BACKUP_ROOT}/manifest.json"

PG_OK=0
NEO_OK=0
CHR_OK=0
PG_SKIPPED=0
NEO_SKIPPED=0
CHR_SKIPPED=0
PG_ERR=""
NEO_ERR=""
CHR_ERR=""
PG_BYTES=0
NEO_BYTES=0
CHR_BYTES=0

# --- Postgres ------------------------------------------------------------
echo ">> Postgres"
if [[ -n "${BACKUP_SKIP_POSTGRES:-}" ]]; then
    PG_SKIPPED=1
    PG_OK=1   # skipped counts as "no gateway failure" — see freshness monitor
    echo "  SKIP: BACKUP_SKIP_POSTGRES set"
elif [[ -z "$PG_CONTAINER" ]]; then
    PG_ERR="postgres container not running"
    echo "  FAIL: $PG_ERR" >&2
else
    PG_DB="${MEM0_PG_DB:-mem0}"
    PG_USER="${MEM0_PG_USER:-mem0}"
    PG_PASS="${MEM0_PG_PASSWORD:-${POSTGRES_PASSWORD:-}}"
    if docker exec -e PGPASSWORD="$PG_PASS" "$PG_CONTAINER" \
        pg_dump --username "$PG_USER" --dbname "$PG_DB" \
        --clean --if-exists --no-owner --no-privileges \
        2>/tmp/pg_dump_err | gzip > "$PG_OUT"; then
        PG_BYTES=$(stat -c%s "$PG_OUT" 2>/dev/null || stat -f%z "$PG_OUT")
        PG_OK=1
        echo "  OK ($PG_BYTES bytes -> $PG_OUT)"
    else
        PG_ERR="$(head -c 500 /tmp/pg_dump_err 2>/dev/null || echo 'pg_dump failed')"
        rm -f "$PG_OUT"
        echo "  FAIL: $PG_ERR" >&2
    fi
fi

# --- Neo4j ---------------------------------------------------------------
# Neo4j Community refuses `neo4j-admin database dump` while the database
# is mounted in a running server (see `neo4j-admin database dump --help`:
# "It is not possible to dump a database that is mounted in a running
# Neo4j server"). The supported path: stop neo4j → dump from a one-shot
# ephemeral container against the same data volume → restart neo4j.
# Daily downtime is ~10–15s at the configured cron hour.
echo ">> Neo4j"
if [[ -n "${BACKUP_SKIP_NEO4J:-}" ]]; then
    NEO_SKIPPED=1
    NEO_OK=1
    echo "  SKIP: BACKUP_SKIP_NEO4J set"
elif [[ -z "$NEO_CONTAINER" ]]; then
    NEO_ERR="neo4j container not running"
    echo "  FAIL: $NEO_ERR" >&2
else
    echo "  Stopping neo4j (brief downtime, dump requires offline DB)..."
    if ! $COMPOSE stop neo4j >/tmp/neo4j_stop_err 2>&1; then
        NEO_ERR="failed to stop neo4j: $(head -c 200 /tmp/neo4j_stop_err)"
        echo "  FAIL: $NEO_ERR" >&2
    else
        # From here on, neo4j MUST be restarted, even if dump fails.
        NEO_DUMP_OK=0
        if $COMPOSE run --rm --no-deps --entrypoint /bin/bash neo4j \
            -c "neo4j-admin database dump neo4j --to-stdout" \
            > "$NEO_OUT" 2>/tmp/neo4j_err; then
            NEO_DUMP_OK=1
        else
            NEO_ERR="$(head -c 500 /tmp/neo4j_err 2>/dev/null || echo 'neo4j-admin failed')"
            rm -f "$NEO_OUT"
        fi
        echo "  Starting neo4j back up..."
        if ! $COMPOSE start neo4j >/tmp/neo4j_start_err 2>&1; then
            # Append the start failure to the dump-error string so the
            # manifest reflects both — but a stuck-down neo4j is worse
            # than a missed backup, so make it loud on stderr.
            echo "  CRITICAL: failed to restart neo4j: $(head -c 200 /tmp/neo4j_start_err)" >&2
            NEO_ERR="${NEO_ERR}; restart_failed: $(head -c 200 /tmp/neo4j_start_err)"
        fi
        if [[ $NEO_DUMP_OK -eq 1 ]]; then
            NEO_BYTES=$(stat -c%s "$NEO_OUT" 2>/dev/null || stat -f%z "$NEO_OUT")
            NEO_OK=1
            echo "  OK ($NEO_BYTES bytes -> $NEO_OUT)"
        else
            echo "  FAIL: $NEO_ERR" >&2
        fi
    fi
fi

# --- ChromaDB ------------------------------------------------------------
echo ">> ChromaDB"
CHR_SRC="${REPO_ROOT}/workspace/memory"
if [[ -n "${BACKUP_SKIP_CHROMADB:-}" ]]; then
    CHR_SKIPPED=1
    CHR_OK=1
    echo "  SKIP: BACKUP_SKIP_CHROMADB set (gateway owns chromadb)"
elif [[ ! -d "$CHR_SRC" ]]; then
    CHR_ERR="chroma source dir not found: $CHR_SRC"
    echo "  FAIL: $CHR_ERR" >&2
else
    if tar --exclude='*.lock' --exclude='*-journal' \
        -czf "$CHR_OUT" -C "${REPO_ROOT}/workspace" memory \
        2>/tmp/chr_err; then
        CHR_BYTES=$(stat -c%s "$CHR_OUT" 2>/dev/null || stat -f%z "$CHR_OUT")
        CHR_OK=1
        echo "  OK ($CHR_BYTES bytes -> $CHR_OUT)"
    else
        CHR_ERR="$(head -c 500 /tmp/chr_err 2>/dev/null || echo 'tar failed')"
        rm -f "$CHR_OUT"
        echo "  FAIL: $CHR_ERR" >&2
    fi
fi

# --- Manifest update -----------------------------------------------------
ALL_OK=$([[ $PG_OK -eq 1 && $NEO_OK -eq 1 && $CHR_OK -eq 1 ]] && echo true || echo false)
COMPLETED_AT="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"

# Build manifest entry via python3 — the previous bash-heredoc form
# had quoting bugs and didn't carry the new ``skipped`` flag.
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is required to update the manifest" >&2
    exit 2
fi

python3 - "$MANIFEST" "$COMPLETED_AT" "$TS" \
    "$PG_OK" "$PG_SKIPPED" "$PG_BYTES" "$PG_ERR" \
    "$NEO_OK" "$NEO_SKIPPED" "$NEO_BYTES" "$NEO_ERR" \
    "$CHR_OK" "$CHR_SKIPPED" "$CHR_BYTES" "$CHR_ERR" \
    "$ALL_OK" <<'PYEOF'
import json
import sys
from pathlib import Path

(manifest_path_str, completed_at, ts,
 pg_ok, pg_skipped, pg_bytes, pg_err,
 neo_ok, neo_skipped, neo_bytes, neo_err,
 chr_ok, chr_skipped, chr_bytes, chr_err,
 all_ok_str) = sys.argv[1:18]
manifest_path = Path(manifest_path_str)


def component(ok: str, skipped: str, path: str, size: str, err: str) -> dict:
    out: dict = {
        "ok": ok == "1",
        "path": path if ok == "1" and skipped != "1" else None,
        "bytes": int(size) if size.isdigit() else 0,
        "error": err if err else None,
    }
    if skipped == "1":
        out["skipped"] = True
        out["skipped_reason"] = "operator_skipped"
        out["error"] = None
    return out


new_entry = {
    "started_at": completed_at,
    "completed_at": completed_at,
    "all_ok": all_ok_str == "true",
    "source": "operator_script",
    "postgres": component(pg_ok, pg_skipped,
                          f"backups/postgres/postgres-{ts}.sql.gz",
                          pg_bytes, pg_err),
    "neo4j": component(neo_ok, neo_skipped,
                       f"backups/neo4j/neo4j-{ts}.dump",
                       neo_bytes, neo_err),
    "chromadb": component(chr_ok, chr_skipped,
                          f"backups/chromadb/chromadb-{ts}.tar.gz",
                          chr_bytes, chr_err),
}

if manifest_path.exists():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        manifest = {"runs": []}
else:
    manifest = {"runs": []}

runs = manifest.setdefault("runs", [])
runs.append(new_entry)
if len(runs) > 200:
    manifest["runs"] = runs[-200:]
manifest["last_updated_at"] = completed_at

tmp = manifest_path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
tmp.replace(manifest_path)
PYEOF

# --- Retention purge -----------------------------------------------------
RETENTION_DAYS="${DB_BACKUP_RETENTION_DAYS:-30}"
echo ">> Purging archives older than ${RETENTION_DAYS} days"
PURGED=0
for sub in postgres neo4j chromadb; do
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        rm -f "$f" && PURGED=$((PURGED + 1))
    done < <(find "${BACKUP_ROOT}/${sub}" -type f -mtime +${RETENTION_DAYS} 2>/dev/null)
done
echo "  Purged: $PURGED file(s)"

_status_label() {
    # $1=ok flag (0/1) $2=skipped flag (0/1) — order matters for the OK output.
    if [[ "$2" -eq 1 ]]; then echo SKIPPED
    elif [[ "$1" -eq 1 ]]; then echo OK
    else echo FAIL
    fi
}

# --- Summary -------------------------------------------------------------
echo
echo "Summary:"
echo "  Postgres:  $(_status_label $PG_OK $PG_SKIPPED)"
echo "  Neo4j:     $(_status_label $NEO_OK $NEO_SKIPPED)"
echo "  ChromaDB:  $(_status_label $CHR_OK $CHR_SKIPPED)"
echo "  Manifest:  $MANIFEST"

if [[ "$ALL_OK" == "true" ]]; then
    exit 0
else
    exit 1
fi
